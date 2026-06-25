from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    text: str
    source: str
    score: float


class VectorStore(Protocol):
    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 8,
        filters: Mapping[str, Any] | None = None,
    ) -> list[ScoredChunk]: ...
