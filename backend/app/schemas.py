from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal[
    "greeting",
    "saree_info",
    "blouse_info",
    "style_recommendation",
    "color_recommendation",
    "faq",
    "clarification",
    "fallback",
]


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Client-generated or server-generated session ID")
    message: str = Field(..., min_length=1, description="Natural language user message")


class Recommendation(BaseModel):
    category: str
    name: str
    reason: str
    source_id: str


class SourceDocument(BaseModel):
    id: str
    category: str
    name: str
    score: float
    retrieval_method: str = "faiss"


class ChatResponse(BaseModel):
    session_id: str
    intent: Intent
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    follow_up_question: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    sources: List[SourceDocument] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    used_llm: bool = False
    rag_enabled: bool = True
    graph_enabled: bool = True
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    llm_enabled: bool
    embedding_provider: str
    vector_store: str
    graph_enabled: bool
    knowledge_items: int
