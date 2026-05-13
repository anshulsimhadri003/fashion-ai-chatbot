from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .compat import Document

from .config import KB_DIR


def load_json_file(filename: str) -> List[Dict[str, Any]]:
    path = KB_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Knowledge file missing: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_knowledge_base() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "sarees": load_json_file("sarees.json"),
        "blouses": load_json_file("blouses.json"),
        "fashion_rules": load_json_file("fashion_rules.json"),
        "faqs": load_json_file("faqs.json"),
    }


def get_doc_name(doc: Dict[str, Any]) -> str:
    return doc.get("name") or doc.get("question") or doc.get("topic") or doc.get("id", "Knowledge Item")


def doc_category(doc: Dict[str, Any]) -> str:
    if doc.get("category"):
        return str(doc["category"])
    if doc.get("rule"):
        return "fashion_rule"
    return "knowledge"


def document_text(doc: Dict[str, Any]) -> str:
    category = doc_category(doc)
    name = get_doc_name(doc)
    chunks = [f"Category: {category}", f"Title: {name}"]

    ordered_keys = [
        "fabric", "style", "best_for", "colors", "mood", "neckline", "sleeves", "description", "styling_tips",
        "pairing_tips", "topic", "applies_to", "rule", "recommendations", "question", "keywords", "answer",
    ]
    for key in ordered_keys:
        value = doc.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        chunks.append(f"{key}: {value}")
    return "\n".join(chunks)


def flatten_knowledge_to_documents(knowledge_base: Dict[str, List[Dict[str, Any]]]) -> List[Document]:
    documents: List[Document] = []
    for collection_name, docs in knowledge_base.items():
        for doc in docs:
            metadata = {
                "id": doc.get("id", get_doc_name(doc)),
                "category": doc_category(doc),
                "name": get_doc_name(doc),
                "collection": collection_name,
                "raw": doc,
            }
            documents.append(Document(page_content=document_text(doc), metadata=metadata))
    return documents


def total_knowledge_items(knowledge_base: Dict[str, List[Dict[str, Any]]]) -> int:
    return sum(len(items) for items in knowledge_base.values())
