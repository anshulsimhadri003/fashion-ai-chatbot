from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .config import MAX_HISTORY

conversation_memory: Dict[str, Dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in conversation_memory:
        conversation_memory[session_id] = {
            "entities": {},
            "last_intent": None,
            "chat_history": [],
            "created_at": now_iso(),
            "updated_at": now_iso(),
        }
    return conversation_memory[session_id]


def compact_history(session: Dict[str, Any]) -> List[Dict[str, str]]:
    history = session.get("chat_history", [])
    return history[-MAX_HISTORY:]


def update_session(session_id: str, intent: str, entities: Dict[str, Any], user_message: str, assistant_reply: str) -> None:
    session = get_session(session_id)
    session["entities"] = entities
    session["last_intent"] = intent
    session["updated_at"] = now_iso()
    session["chat_history"].append({"role": "user", "content": user_message})
    session["chat_history"].append({"role": "assistant", "content": assistant_reply})
    session["chat_history"] = session["chat_history"][-MAX_HISTORY:]


def clear_session(session_id: str) -> None:
    conversation_memory.pop(session_id, None)
