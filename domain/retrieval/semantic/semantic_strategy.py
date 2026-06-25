from __future__ import annotations

from domain.ports.embedder import Embedder
from domain.ports.vector_store import VectorStore
from domain.query import Query
from domain.retrieval.evidence import Evidence, Passage, Provenance
from domain.routing.route import Route


class SemanticStrategy:
    """Embed the query, search the vector store, wrap hits as cited passages."""

    route = Route.SEMANTIC
    description = (
        "Answers questions about unstructured documents: policies, guides, FAQs, "
        "definitions, and 'what is / how do I' explanations."
    )

    def __init__(self, embedder: Embedder, store: VectorStore, k: int = 6) -> None:
        self._embedder = embedder
        self._store = store
        self._k = k

    async def retrieve(self, query: Query) -> Evidence:
        vector = await self._embedder.embed(query.text)
        hits = await self._store.search(vector, k=self._k, filters=query.filters)
        passages = tuple(
            Passage(
                text=h.text,
                provenance=Provenance(source=h.source, locator=h.id, score=h.score),
            )
            for h in hits
        )
        return Evidence(passages=passages, routes_used=(Route.SEMANTIC,))
