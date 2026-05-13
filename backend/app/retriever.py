from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from sklearn.feature_extraction.text import HashingVectorizer
except Exception:  # pragma: no cover
    HashingVectorizer = None  # type: ignore

from .compat import Document

from .config import (
    FAISS_INDEX_PATH,
    LOCAL_EMBEDDING_DIM,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    USE_OPENAI_EMBEDDINGS,
    VECTOR_DIR,
    VECTOR_MATRIX_PATH,
    VECTOR_METADATA_PATH,
)
from .knowledge import doc_category, get_doc_name
from .nlp import tokenize
from .schemas import Intent


@dataclass
class RetrievedDocument:
    document: Document
    score: float
    vector_score: float
    retrieval_method: str

    @property
    def raw(self) -> Dict[str, Any]:
        raw = self.document.metadata.get("raw", {})
        return raw if isinstance(raw, dict) else {}


class LocalHashEmbeddings:
    """Deterministic local embeddings for offline demos.

    This is not as semantically rich as OpenAI embeddings, but it still creates dense vectors and uses FAISS.
    It keeps the assessment project runnable without paid keys or internet access.
    """

    provider_name = "local-hashing-embeddings"

    def __init__(self, dimension: int = LOCAL_EMBEDDING_DIM):
        self.dimension = dimension
        self.last_provider_name = self.provider_name
        if HashingVectorizer is None:
            self.vectorizer = None
        else:
            self.vectorizer = HashingVectorizer(
                n_features=dimension,
                alternate_sign=False,
                norm=None,
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=True,
            )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is not None:
            matrix = self.vectorizer.transform(list(texts)).astype(np.float32)
            dense = matrix.toarray().astype("float32")
        else:
            dense = np.zeros((len(texts), self.dimension), dtype="float32")
            for row, text in enumerate(texts):
                for token in tokenize(text):
                    dense[row, hash(token) % self.dimension] += 1.0
        return normalize_matrix(dense)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class OpenAIEmbeddingProvider:
    provider_name = "openai-text-embedding-3-large"

    def __init__(self, fallback: LocalHashEmbeddings):
        self.fallback = fallback
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OpenAI is not None and OPENAI_API_KEY else None
        self.provider_name = f"openai-{OPENAI_EMBEDDING_MODEL}"
        self.last_provider_name = self.provider_name

    def _embed_with_openai(self, texts: Sequence[str]) -> np.ndarray:
        if self.client is None:
            raise RuntimeError("OpenAI client is not configured")
        response = self.client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=list(texts))
        vectors = [item.embedding for item in response.data]
        self.last_provider_name = self.provider_name
        return normalize_matrix(np.array(vectors, dtype="float32"))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        try:
            return self._embed_with_openai(texts)
        except Exception as exc:
            print(f"OpenAI embeddings failed; falling back to local embeddings: {exc}")
            self.last_provider_name = self.fallback.provider_name
            return self.fallback.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype("float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def lexical_overlap(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for token in q_tokens if token in text_lower)
    return hits / max(len(q_tokens), 1)


def entity_boost(doc_text: str, entities: Dict[str, Any]) -> float:
    score = 0.0
    lower = doc_text.lower()
    for key, weight in {
        "event": 0.25,
        "preferred_color": 0.22,
        "fabric": 0.2,
        "mood": 0.16,
        "time_of_day": 0.08,
    }.items():
        value = entities.get(key)
        if value and str(value).lower() in lower:
            score += weight
    return score


def intent_boost(intent: Intent, category: str, doc_text: str) -> float:
    if intent == "faq" and category == "faq":
        return 0.35
    if intent == "blouse_info" and category in {"blouse", "fashion_rule"}:
        return 0.25
    if intent == "saree_info" and category in {"saree", "fashion_rule"}:
        return 0.25
    if intent in {"style_recommendation", "color_recommendation", "clarification"} and category in {"saree", "blouse", "fashion_rule"}:
        return 0.18
    if intent == "greeting":
        return -0.2
    return 0.0


class FashionRAGStore:
    """FAISS-backed semantic retriever over fashion and FAQ documents."""

    def __init__(self, documents: List[Document], force_rebuild: bool = False):
        self.documents = documents
        self.local_embeddings = LocalHashEmbeddings()
        self.embedding_provider = self._select_embedding_provider()
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.index_loaded_from_disk = False
        self.actual_embedding_provider_name = getattr(self.embedding_provider, "provider_name", "unknown")
        self.vector_store_name = "faiss"
        if force_rebuild or not self._load_cached_index(documents):
            self.rebuild(documents)

    def _select_embedding_provider(self):
        if USE_OPENAI_EMBEDDINGS and OPENAI_API_KEY and OpenAI is not None:
            return OpenAIEmbeddingProvider(self.local_embeddings)
        return self.local_embeddings

    @property
    def embedding_provider_name(self) -> str:
        return self.actual_embedding_provider_name

    def _documents_fingerprint(self, documents: List[Document]) -> str:
        payload = [
            {
                "id": doc.metadata.get("id"),
                "category": doc.metadata.get("category"),
                "content": doc.page_content,
            }
            for doc in documents
        ]
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _requested_provider_name(self) -> str:
        return getattr(self.embedding_provider, "provider_name", "unknown")

    def _load_cached_index(self, documents: List[Document]) -> bool:
        if not VECTOR_METADATA_PATH.exists() or not VECTOR_MATRIX_PATH.exists():
            return False

        try:
            metadata = json.loads(VECTOR_METADATA_PATH.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != self._documents_fingerprint(documents):
                return False
            if metadata.get("provider") != self._requested_provider_name():
                return False

            vectors = np.load(VECTOR_MATRIX_PATH).astype("float32")
            if vectors.ndim != 2 or vectors.shape[0] != len(documents):
                return False

            self.documents = documents
            self.vectors = vectors
            self.actual_embedding_provider_name = str(metadata.get("provider", self._requested_provider_name()))

            if faiss is not None and FAISS_INDEX_PATH.exists():
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                if self.index.d != vectors.shape[1] or self.index.ntotal != vectors.shape[0]:
                    return False
                self.vector_store_name = "faiss.IndexFlatIP"
            elif faiss is not None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
                self.index.add(vectors)
                self.vector_store_name = "faiss.IndexFlatIP"
                faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            else:
                self.index = None
                self.vector_store_name = "numpy-cosine-fallback"

            self.index_loaded_from_disk = True
            return True
        except Exception as exc:
            print(f"Cached vector index could not be loaded; rebuilding: {exc}")
            return False

    def _save_cached_index(self) -> None:
        if self.vectors is None:
            return

        try:
            VECTOR_DIR.mkdir(parents=True, exist_ok=True)
            np.save(VECTOR_MATRIX_PATH, self.vectors)
            if faiss is not None and self.index is not None:
                faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            VECTOR_METADATA_PATH.write_text(
                json.dumps(
                    {
                        "fingerprint": self._documents_fingerprint(self.documents),
                        "provider": self.actual_embedding_provider_name,
                        "dimension": int(self.vectors.shape[1]),
                        "documents": len(self.documents),
                        "vector_store": self.vector_store_name,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"Vector index could not be saved; continuing with in-memory index: {exc}")

    def rebuild(self, documents: List[Document]) -> None:
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.vectors = self.embedding_provider.embed_documents(texts)
        self.actual_embedding_provider_name = getattr(
            self.embedding_provider,
            "last_provider_name",
            self._requested_provider_name(),
        )
        if self.vectors.ndim != 2:
            raise RuntimeError("Embedding provider returned invalid vector matrix")

        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.vector_store_name = "faiss.IndexFlatIP"
        else:
            self.index = None
            self.vector_store_name = "numpy-cosine-fallback"
        self.index_loaded_from_disk = False
        self._save_cached_index()

    def search(self, query: str, intent: Intent, entities: Dict[str, Any], top_k: int = 8) -> List[RetrievedDocument]:
        if not self.documents:
            return []

        query_vector = self.embedding_provider.embed_query(query).astype("float32")
        if self.vectors is not None and query_vector.shape[0] != self.vectors.shape[1]:
            print("Query embedding dimension changed; rebuilding index with local fallback embeddings.")
            self.embedding_provider = self.local_embeddings
            self.actual_embedding_provider_name = self.local_embeddings.provider_name
            self.rebuild(self.documents)
            query_vector = self.embedding_provider.embed_query(query).astype("float32")
        search_k = min(max(top_k * 3, top_k), len(self.documents))

        if self.index is not None:
            scores, indices = self.index.search(np.expand_dims(query_vector, axis=0), search_k)
            raw_hits = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if int(idx) >= 0]
            method = "faiss"
        else:
            assert self.vectors is not None
            scores = np.dot(self.vectors, query_vector)
            sorted_indices = np.argsort(scores)[::-1][:search_k]
            raw_hits = [(int(idx), float(scores[idx])) for idx in sorted_indices]
            method = "numpy-cosine-fallback"

        reranked: List[RetrievedDocument] = []
        for idx, vector_score in raw_hits:
            doc = self.documents[idx]
            category = str(doc.metadata.get("category", "knowledge"))
            text = doc.page_content
            combined_score = (
                vector_score
                + lexical_overlap(query, text) * 0.16
                + entity_boost(text, entities)
                + intent_boost(intent, category, text)
            )
            reranked.append(
                RetrievedDocument(
                    document=doc,
                    score=round(float(combined_score), 4),
                    vector_score=round(float(vector_score), 4),
                    retrieval_method=method,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]


def recommendation_from_doc(doc: Dict[str, Any]) -> Optional[Tuple[str, str, str, str]]:
    category = doc_category(doc)
    if category not in {"saree", "blouse"}:
        return None
    name = get_doc_name(doc)
    reason = doc.get("description") or ", ".join(doc.get("styling_tips", doc.get("pairing_tips", [])))
    return category, name, str(reason)[:260], doc.get("id", "unknown")
