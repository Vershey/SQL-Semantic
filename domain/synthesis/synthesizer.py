from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from domain.query import Query
from domain.retrieval.evidence import Evidence
from domain.routing.route import RouteDecision


@dataclass(frozen=True)
class Citation:
    source: str
    locator: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...] = ()
    is_clarification: bool = False
    decision: RouteDecision | None = None

    @staticmethod
    def clarification(text: str) -> "Answer":
        return Answer(text=text, is_clarification=True)


class Synthesizer(Protocol):
    """Port. Turns merged evidence (tables + passages) into a grounded answer."""

    async def synthesize(
        self, query: Query, evidence: Evidence, decision: RouteDecision
    ) -> Answer: ...
