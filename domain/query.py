from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ConversationContext:
    """Prior turns, if the caller wants the router/synthesizer to be context-aware."""
    history: tuple[str, ...] = ()


@dataclass(frozen=True)
class Query:
    """A user's natural-language request plus anything that constrains the answer."""
    text: str
    context: ConversationContext | None = None
    filters: Mapping[str, Any] = field(default_factory=dict)
