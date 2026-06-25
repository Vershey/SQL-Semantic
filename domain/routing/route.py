from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    SQL = "sql"
    SEMANTIC = "semantic"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class RouteDecision:
    """Which strategies should run. More than one route == a hybrid answer."""
    routes: tuple[Route, ...]
    confidence: float
    rationale: str
