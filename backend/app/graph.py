from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = None  # type: ignore
    StateGraph = None  # type: ignore

from .config import RAG_TOP_K
from .knowledge import get_doc_name
from .llm import build_local_reply, call_openai_reply
from .memory import compact_history, get_session, update_session
from .nlp import context_query, detect_intent, extract_entities
from .retriever import FashionRAGStore, RetrievedDocument, recommendation_from_doc
from .schemas import ChatRequest, ChatResponse, Recommendation, SourceDocument


class ChatState(TypedDict, total=False):
    session_id: str
    message: str
    session: Dict[str, Any]
    history: List[Dict[str, str]]
    intent: str
    entities: Dict[str, Any]
    retrieval_query: str
    retrieved: List[RetrievedDocument]
    recommendations: List[Recommendation]
    sources: List[SourceDocument]
    reply: str
    follow_up_question: Optional[str]
    used_llm: bool


class FashionChatGraph:
    """LangGraph orchestration for the conversation pipeline.

    Flow:
    load_memory -> understand_intent -> extract_preferences -> retrieve_context -> generate_response -> update_memory
    """

    def __init__(self, retriever: FashionRAGStore):
        self.retriever = retriever
        self.graph_enabled = StateGraph is not None
        self.compiled_graph = self._compile_graph() if self.graph_enabled else None

    def _compile_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("load_memory", self.load_memory)
        graph.add_node("understand_intent", self.understand_intent)
        graph.add_node("extract_preferences", self.extract_preferences)
        graph.add_node("retrieve_context", self.retrieve_context)
        graph.add_node("generate_response", self.generate_response)
        graph.add_node("update_memory", self.update_memory)

        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "understand_intent")
        graph.add_edge("understand_intent", "extract_preferences")
        graph.add_edge("extract_preferences", "retrieve_context")
        graph.add_edge("retrieve_context", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)
        return graph.compile()

    def load_memory(self, state: ChatState) -> ChatState:
        session_id = state.get("session_id") or str(uuid.uuid4())
        session = get_session(session_id)
        return {"session_id": session_id, "session": session, "history": compact_history(session)}

    def understand_intent(self, state: ChatState) -> ChatState:
        message = state["message"]
        session = state.get("session", {})
        previous_intent = session.get("last_intent")
        intent = detect_intent(message, previous_intent)
        if intent == "clarification" and previous_intent:
            intent = previous_intent
        return {"intent": intent}

    def extract_preferences(self, state: ChatState) -> ChatState:
        message = state["message"]
        session = state.get("session", {})
        entities = extract_entities(message, session.get("entities", {}))
        retrieval_query = context_query(message, entities)
        return {"entities": entities, "retrieval_query": retrieval_query}

    def retrieve_context(self, state: ChatState) -> ChatState:
        intent = state.get("intent", "fallback")
        entities = state.get("entities", {})
        retrieval_query = state.get("retrieval_query", state["message"])
        retrieved = self.retriever.search(retrieval_query, intent, entities, top_k=RAG_TOP_K)  # type: ignore[arg-type]

        recommendations: List[Recommendation] = []
        seen = set()
        for item in retrieved:
            rec = recommendation_from_doc(item.raw)
            if rec is None:
                continue
            category, name, reason, source_id = rec
            if source_id in seen:
                continue
            seen.add(source_id)
            recommendations.append(Recommendation(category=category, name=name, reason=reason, source_id=source_id))
            if len(recommendations) >= 4:
                break

        sources = [
            SourceDocument(
                id=item.raw.get("id", str(item.document.metadata.get("id", "unknown"))),
                category=str(item.document.metadata.get("category", "knowledge")),
                name=get_doc_name(item.raw),
                score=round(item.score, 3),
                retrieval_method=item.retrieval_method,
            )
            for item in retrieved[:6]
        ]
        return {"retrieved": retrieved, "recommendations": recommendations, "sources": sources}

    def generate_response(self, state: ChatState) -> ChatState:
        intent = state.get("intent", "fallback")
        message = state["message"]
        entities = state.get("entities", {})
        retrieved = state.get("retrieved", [])
        history = state.get("history", [])

        llm_result = call_openai_reply(intent, message, entities, retrieved, history)  # type: ignore[arg-type]
        if llm_result:
            reply, follow_up = llm_result
            return {"reply": reply, "follow_up_question": follow_up, "used_llm": True}

        reply, follow_up = build_local_reply(intent, message, entities, retrieved)  # type: ignore[arg-type]
        return {"reply": reply, "follow_up_question": follow_up, "used_llm": False}

    def update_memory(self, state: ChatState) -> ChatState:
        update_session(
            state["session_id"],
            str(state.get("intent", "fallback")),
            state.get("entities", {}),
            state["message"],
            state.get("reply", ""),
        )
        return {}

    # Backward-compatible aliases for older code/tests that imported the original method names.
    load_context = load_memory
    understand = understand_intent
    retrieve = retrieve_context
    generate = generate_response
    persist = update_memory

    def invoke(self, request: ChatRequest) -> ChatResponse:
        initial_state: ChatState = {
            "session_id": request.session_id or str(uuid.uuid4()),
            "message": request.message.strip(),
        }
        if self.compiled_graph is not None:
            final_state = self.compiled_graph.invoke(initial_state)
        else:
            # Manual fallback keeps the app running if LangGraph is not installed.
            state = dict(initial_state)
            for node in [
                self.load_memory,
                self.understand_intent,
                self.extract_preferences,
                self.retrieve_context,
                self.generate_response,
                self.update_memory,
            ]:
                state.update(node(state))
            final_state = state

        context = {
            "entities": final_state.get("entities", {}),
            "memory": {
                "last_intent": final_state.get("intent", "fallback"),
                "history_turns": len(final_state.get("history", [])) // 2,
            },
            "retrieval_query": final_state.get("retrieval_query", request.message.strip()),
            "source_count": len(final_state.get("sources", [])),
        }

        return ChatResponse(
            session_id=final_state["session_id"],
            intent=final_state.get("intent", "fallback"),  # type: ignore[arg-type]
            reply=final_state.get("reply", "I could not generate a response."),
            recommendations=final_state.get("recommendations", []),
            follow_up_question=final_state.get("follow_up_question"),
            entities=final_state.get("entities", {}),
            sources=final_state.get("sources", []),
            context=context,
            used_llm=bool(final_state.get("used_llm", False)),
            rag_enabled=True,
            graph_enabled=self.graph_enabled,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
