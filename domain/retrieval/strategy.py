from __future__ import annotations

from typing import Protocol

from domain.query import Query
from domain.retrieval.evidence import Evidence
from domain.routing.route import Route


class RetrievalStrategy(Protocol):
    """Port. The common shape of SQL retrieval, semantic retrieval, and anything
    you add later. `description` is consumed by the router to decide routing."""

    route: Route
    description: str

    async def retrieve(self, query: Query) -> Evidence: ...
