from __future__ import annotations

from typing import Protocol

from domain.retrieval.evidence import TabularResult


class SqlExecutor(Protocol):
    """Executes a validated, read-only SQL string and returns normalized rows."""

    async def execute(self, sql: str) -> TabularResult: ...
