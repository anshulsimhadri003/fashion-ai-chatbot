"""
Fashion-Tech RAG Conversational AI Chatbot Backend

FastAPI backend for a domain-focused saree/blouse styling assistant.

- LangChain Document objects for KB ingestion
- FAISS vector search over embedded fashion/FAQ knowledge
- OpenAI embeddings when configured, deterministic local embeddings otherwise
- LangGraph state-machine orchestration for context -> intent -> retrieval -> generation -> memory
- OpenAI LLM response generation with grounded RAG context and local fallback
"""

from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import APP_NAME, APP_VERSION, CORS_ORIGINS, OPENAI_API_KEY, USE_OPENAI
from app.graph import FashionChatGraph
from app.knowledge import flatten_knowledge_to_documents, load_knowledge_base, total_knowledge_items
from app.memory import clear_session, conversation_memory
from app.retriever import FashionRAGStore
from app.schemas import ChatRequest, ChatResponse, HealthResponse

# -----------------------------
# Startup resources
# -----------------------------
knowledge_base = load_knowledge_base()
documents = flatten_knowledge_to_documents(knowledge_base)
rag_store = FashionRAGStore(documents)
chat_graph = FashionChatGraph(rag_store)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Domain-focused RAG conversational AI chatbot for sarees, blouses, styling, and platform FAQs. "
        "Uses FastAPI, FAISS, embeddings, LangChain documents, LangGraph orchestration, and optional OpenAI generation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def health_payload() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=APP_NAME,
        version=APP_VERSION,
        llm_enabled=bool(OPENAI_API_KEY and USE_OPENAI),
        embedding_provider=rag_store.embedding_provider_name,
        vector_store=rag_store.vector_store_name,
        graph_enabled=chat_graph.graph_enabled,
        knowledge_items=total_knowledge_items(knowledge_base),
    )


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return health_payload()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return health_payload()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if not request.session_id:
        request.session_id = str(uuid.uuid4())
    return chat_graph.invoke(request)


@app.get("/memory/{session_id}")
def read_memory(session_id: str) -> Dict[str, Any]:
    if session_id not in conversation_memory:
        raise HTTPException(status_code=404, detail="Session not found")
    return conversation_memory[session_id]


@app.delete("/memory/{session_id}")
def delete_memory(session_id: str) -> Dict[str, str]:
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/kb/rebuild")
def rebuild_knowledge_base() -> Dict[str, Any]:
    """Reload JSON KB and rebuild the FAISS index without restarting the server."""
    global knowledge_base, documents, rag_store, chat_graph
    knowledge_base = load_knowledge_base()
    documents = flatten_knowledge_to_documents(knowledge_base)
    rag_store = FashionRAGStore(documents, force_rebuild=True)
    chat_graph = FashionChatGraph(rag_store)
    return {
        "status": "rebuilt",
        "counts": {key: len(value) for key, value in knowledge_base.items()},
        "embedding_provider": rag_store.embedding_provider_name,
        "vector_store": rag_store.vector_store_name,
        "graph_enabled": chat_graph.graph_enabled,
    }
