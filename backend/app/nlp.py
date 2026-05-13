from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .schemas import Intent

STOPWORDS = {
    "a", "an", "the", "for", "to", "me", "my", "i", "need", "want", "with", "and", "or", "of", "in", "on",
    "is", "are", "can", "you", "please", "suggest", "recommend", "show", "tell", "about", "something", "look",
}

EVENT_KEYWORDS = {
    "wedding": ["wedding", "marriage", "shaadi", "day wedding", "night wedding"],
    "reception": ["reception"],
    "engagement": ["engagement", "ring ceremony"],
    "haldi": ["haldi"],
    "mehendi": ["mehendi", "mehndi"],
    "sangeet": ["sangeet"],
    "party": ["party", "cocktail", "night out", "date night"],
    "office": ["office", "work", "meeting", "formal"],
    "casual": ["casual", "daily", "regular", "college"],
    "festival": ["festival", "festive", "diwali", "pongal", "navratri", "puja", "pooja"],
}

COLOR_KEYWORDS = [
    "royal blue", "pastel pink", "powder blue", "red", "maroon", "gold", "yellow", "mustard", "green", "emerald",
    "blue", "navy", "black", "white", "ivory", "beige", "grey", "silver", "champagne", "pink", "pastel",
    "lavender", "mint", "peach", "olive", "wine", "orange", "lime",
]

FABRIC_KEYWORDS = [
    "cotton linen", "silk", "banarasi", "kanjivaram", "organza", "georgette", "chiffon", "cotton", "linen", "handloom",
]

MOOD_KEYWORDS = [
    "royal", "traditional", "modern", "minimal", "elegant", "bold", "glam", "soft", "romantic", "festive",
    "simple", "classy", "premium", "comfortable", "playful", "ethnic", "light", "lighter", "lightweight",
]

BLOUSE_KEYWORDS = [
    "blouse", "neck", "neckline", "sleeve", "sleeves", "back design", "high neck", "boat neck", "corset", "halter",
    "sleeveless", "embroidered", "pearl", "mirror work", "mirror-work",
]

FAQ_KEYWORDS = [
    "track", "order", "return", "refund", "exchange", "delivery", "shipping", "payment", "upi", "cod", "customize",
    "measurement", "size", "account", "where is", "how long", "cancel",
]

GREETING_KEYWORDS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def contains_any(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def detect_intent(message: str, previous_intent: Optional[str] = None) -> Intent:
    text = normalize(message)

    if text in GREETING_KEYWORDS or any(text == item for item in GREETING_KEYWORDS):
        return "greeting"

    if contains_any(text, FAQ_KEYWORDS):
        return "faq"

    if contains_any(text, BLOUSE_KEYWORDS):
        return "blouse_info"

    has_event = any(keyword in text for variants in EVENT_KEYWORDS.values() for keyword in variants)
    has_color = any(color in text for color in COLOR_KEYWORDS)
    has_fabric = any(fabric in text for fabric in FABRIC_KEYWORDS)
    has_mood = any(mood in text for mood in MOOD_KEYWORDS)

    if has_event or contains_any(text, ["style", "outfit", "wear", "occasion", "function", "look"]):
        return "style_recommendation"

    if "saree" in text or has_fabric:
        return "saree_info"

    if has_color:
        return "color_recommendation"

    # Important for natural follow-ups like: "modern and pastel", "make it more festive", "what about black?"
    if previous_intent in {"style_recommendation", "saree_info", "blouse_info", "color_recommendation"} and (has_color or has_mood or has_fabric):
        return previous_intent  # type: ignore[return-value]

    if len(tokenize(text)) <= 3 and previous_intent in {"style_recommendation", "saree_info", "blouse_info", "color_recommendation"}:
        return "clarification"

    return "fallback"


def extract_entities(message: str, existing_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    text = normalize(message)
    entities: Dict[str, Any] = dict(existing_context or {})

    for event, variants in EVENT_KEYWORDS.items():
        if contains_any(text, variants):
            entities["event"] = event
            break

    detected_colors = [color for color in COLOR_KEYWORDS if color in text]
    if detected_colors:
        entities["preferred_color"] = detected_colors[-1]

    detected_fabrics = [fabric for fabric in FABRIC_KEYWORDS if fabric in text]
    if detected_fabrics:
        entities["fabric"] = detected_fabrics[-1]

    detected_moods = [mood for mood in MOOD_KEYWORDS if mood in text]
    if detected_moods:
        mood = detected_moods[-1]
        entities["mood"] = "lightweight" if mood in {"light", "lighter"} else mood

    budget_match = re.search(r"(?:under|below|budget|less than)\s*(?:rs\.?|₹|inr)?\s*(\d{3,6})", text)
    if budget_match:
        entities["budget"] = int(budget_match.group(1))

    if any(term in text for term in ["day", "morning", "afternoon"]):
        entities["time_of_day"] = "day"
    elif any(term in text for term in ["night", "evening"]):
        entities["time_of_day"] = "evening"

    return entities


def context_query(message: str, entities: Dict[str, Any]) -> str:
    parts = [message]
    for key in ["event", "preferred_color", "fabric", "mood", "time_of_day"]:
        if entities.get(key):
            parts.append(f"{key}: {entities[key]}")
    return " | ".join(parts)
