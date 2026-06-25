from __future__ import annotations

from typing import Any, Mapping, Sequence

from domain.ports.embedder import Embedder
from domain.ports.vector_store import ScoredChunk


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # both vectors are L2-normalized


class InMemoryVectorStore:
    """Process-local vector store. Embeds documents on add(); brute-force cosine on
    search(). Replace with a pgvector/Pinecone/Qdrant adapter for real workloads."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._items: list[tuple[str, str, str, Sequence[float]]] = []

    async def add(self, doc_id: str, text: str, source: str) -> None:
        # Embed the source/title alongside the body so title keywords help ranking.
        vector = await self._embedder.embed(f"{source} {text}")
        self._items.append((doc_id, text, source, vector))

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        wanted_source = (filters or {}).get("source")
        scored = [
            ScoredChunk(id=doc_id, text=text, source=source, score=_cosine(embedding, vec))
            for (doc_id, text, source, vec) in self._items
            if wanted_source is None or source == wanted_source
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]
