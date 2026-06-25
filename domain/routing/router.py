from __future__ import annotations

from typing import Protocol

from domain.query import Query
from domain.routing.route import RouteDecision


class Router(Protocol):
    """Port. Maps a query to a routing decision. Implementations live in infrastructure."""

    async def route(self, query: Query) -> RouteDecision: ...
