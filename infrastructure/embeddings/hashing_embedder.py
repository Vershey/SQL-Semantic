from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "of", "to", "in", "on",
    "for", "and", "or", "with", "our", "your", "my", "we", "you", "it", "that",
    "this", "what", "how", "do", "does", "did", "can", "at", "by", "as",
})


def _tokenize(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


class HashingEmbedder:
    """Dependency-free, deterministic embedder (signed hashing trick). Good enough
    for an offline demo. Swap for a real embedding model adapter in production."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    async def embed(self, text: str) -> Sequence[float]:
        vec = [0.0] * self._dim
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 1.0 if (h >> 8) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]
