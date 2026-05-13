from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

try:
    from langchain_core.documents import Document as LangChainDocument

    Document = LangChainDocument
    LANGCHAIN_CORE_AVAILABLE = True
except Exception:  # pragma: no cover
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: Dict[str, Any] = field(default_factory=dict)

    LANGCHAIN_CORE_AVAILABLE = False
