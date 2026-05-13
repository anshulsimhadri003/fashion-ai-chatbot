from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "knowledge_base"
VECTOR_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_DIR / "fashion_kb.faiss"
VECTOR_MATRIX_PATH = VECTOR_DIR / "fashion_kb_vectors.npy"
VECTOR_METADATA_PATH = VECTOR_DIR / "fashion_kb_metadata.json"

load_dotenv(BASE_DIR / ".env")

APP_NAME = "Fashion-Tech RAG Conversational AI Chatbot"
APP_VERSION = "2.0.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large").strip()
USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() in {"1", "true", "yes"}
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() in {"1", "true", "yes"}

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM", "384"))

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
