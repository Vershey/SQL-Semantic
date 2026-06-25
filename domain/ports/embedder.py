from __future__ import annotations

from typing import Protocol, Sequence


class Embedder(Protocol):
    async def embed(self, text: str) -> Sequence[float]: ...
