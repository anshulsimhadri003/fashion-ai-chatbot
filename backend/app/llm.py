from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from .config import OPENAI_API_KEY, OPENAI_MODEL, USE_OPENAI
from .knowledge import get_doc_name
from .retriever import RetrievedDocument
from .schemas import Intent


def retrieved_to_prompt_payload(retrieved: List[RetrievedDocument]) -> List[Dict[str, Any]]:
    payload = []
    for item in retrieved:
        raw = item.raw
        payload.append(
            {
                "source_id": raw.get("id", item.document.metadata.get("id")),
                "category": item.document.metadata.get("category"),
                "name": get_doc_name(raw),
                "score": item.score,
                "content": item.document.page_content,
            }
        )
    return payload


def build_llm_input(
    intent: Intent,
    message: str,
    entities: Dict[str, Any],
    retrieved: List[RetrievedDocument],
    history: List[Dict[str, str]],
) -> str:
    return json.dumps(
        {
            "role": "fashion_tech_chatbot",
            "goal": "Respond naturally like a helpful stylist and platform support assistant.",
            "intent": intent,
            "user_message": message,
            "known_user_context": entities,
            "recent_chat_history": history[-10:],
            "retrieved_knowledge": retrieved_to_prompt_payload(retrieved),
            "style_rules": [
                "Be warm, human, and practical. Avoid robotic wording.",
                "Use the retrieved context as the source of truth for product/style/FAQ details.",
                "For styling requests, give one clear main recommendation and 1-2 supporting alternatives when useful.",
                "For platform FAQ requests, do not invent policies beyond the retrieved FAQ content.",
                "Maintain continuity with prior context. If the user says 'make it modern', remember the earlier occasion/color.",
                "Ask only one follow-up question when a critical detail is missing.",
                "Keep the reply concise: normally 5-8 lines maximum.",
            ],
            "output_contract": {
                "reply": "user-facing answer only",
                "follow_up_question": "one optional question or null",
            },
        },
        ensure_ascii=False,
    )


def call_openai_reply(
    intent: Intent,
    message: str,
    entities: Dict[str, Any],
    retrieved: List[RetrievedDocument],
    history: List[Dict[str, str]],
) -> Optional[Tuple[str, Optional[str]]]:
    if not USE_OPENAI or not OPENAI_API_KEY or OpenAI is None:
        return None

    system_instructions = (
        "You are Nira, a premium fashion-tech conversational assistant for sarees and blouse styling. "
        "You combine stylist-level advice with platform support. You are contextual, warm, concise, and grounded in retrieved knowledge. "
        "Return valid JSON only with keys: reply and follow_up_question. follow_up_question can be null."
    )
    prompt = build_llm_input(intent, message, entities, retrieved, history)

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=system_instructions,
            input=prompt,
        )
        text = getattr(response, "output_text", None)
        if not text:
            return None
        text = text.strip()
        try:
            parsed = json.loads(text)
            reply = str(parsed.get("reply", "")).strip()
            follow_up = parsed.get("follow_up_question")
            follow_up = str(follow_up).strip() if follow_up else None
            if reply:
                return reply, follow_up
        except json.JSONDecodeError:
            return text, None
    except Exception as exc:
        print(f"OpenAI response generation failed; using local fallback: {exc}")
    return None


def build_local_reply(
    intent: Intent,
    message: str,
    entities: Dict[str, Any],
    retrieved: List[RetrievedDocument],
) -> Tuple[str, Optional[str]]:
    if intent == "greeting":
        return (
            "Hi, I’m Nira. I can help you pick sarees, blouse designs, color pairings, and also answer order, delivery, return, and customization questions.",
            "What are you dressing for — a wedding, festive function, party, office look, or something casual?",
        )

    if intent == "faq" and retrieved:
        for item in retrieved:
            raw = item.raw
            if raw.get("answer"):
                return raw["answer"], None

    sarees = []
    blouses = []
    rules = []
    for item in retrieved:
        raw = item.raw
        category = raw.get("category") or item.document.metadata.get("category")
        if category == "saree":
            sarees.append(raw)
        elif category == "blouse":
            blouses.append(raw)
        elif category == "fashion_rule":
            rules.append(raw)
        elif raw.get("rule"):
            rules.append(raw)

    event = entities.get("event")
    color = entities.get("preferred_color")
    mood = entities.get("mood")
    fabric = entities.get("fabric")

    context = []
    if color:
        context.append(str(color))
    if mood:
        context.append(str(mood))
    if event:
        context.append(f"{event} look")
    elif fabric:
        context.append(f"{fabric} saree look")
    context_phrase = " ".join(context) if context else "your look"

    if sarees or blouses:
        primary_saree = sarees[0] if sarees else None
        primary_blouse = blouses[0] if blouses else None
        rule_text = rules[0].get("rule") if rules else None

        parts = []
        if primary_saree:
            parts.append(f"For {context_phrase}, I’d start with a {primary_saree.get('name')}. {primary_saree.get('description')}")
        if primary_blouse:
            parts.append(f"Pair it with a {primary_blouse.get('name')} — {primary_blouse.get('description')}")
        if rule_text:
            parts.append(str(rule_text))
        if not primary_saree and primary_blouse:
            parts.insert(0, f"A strong blouse direction for {context_phrase} is the {primary_blouse.get('name')}.")

        follow_up = None
        if not event and intent in {"style_recommendation", "saree_info", "clarification"}:
            follow_up = "Which occasion is this for — wedding, reception, haldi, office, party, or casual wear?"
        elif not color and intent in {"style_recommendation", "color_recommendation", "clarification"}:
            follow_up = "Do you already have a color family in mind, like pastel, red, black, gold, or green?"
        elif not mood:
            follow_up = "Do you want the vibe to be traditional, modern, minimal, royal, or bold?"

        return "\n\n".join(parts[:3]), follow_up

    return (
        "I can help, but I need one styling detail to make this accurate. Share the occasion, color, saree fabric, or the mood you want, and I’ll suggest a complete direction.",
        "What occasion are you dressing for?",
    )
