from __future__ import annotations

from typing import Protocol


class LanguageModel(Protocol):
    """Port for any chat/completion model used for routing, SQL gen, synthesis."""

    async def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> str: ...
