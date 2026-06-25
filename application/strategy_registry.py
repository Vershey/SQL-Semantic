from __future__ import annotations

from domain.retrieval.strategy import RetrievalStrategy
from domain.routing.route import Route


def build_registry(*strategies: RetrievalStrategy) -> dict[Route, RetrievalStrategy]:
    """Index strategies by their own declared `route`. Adding a new strategy is a
    one-line change here (or at the call site) and nothing else needs to know."""
    registry: dict[Route, RetrievalStrategy] = {}
    for strategy in strategies:
        if strategy.route in registry:
            raise ValueError(f"Duplicate strategy registered for route {strategy.route}.")
        registry[strategy.route] = strategy
    return registry
