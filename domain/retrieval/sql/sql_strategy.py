from __future__ import annotations

import re

from domain.ports.llm import LanguageModel
from domain.ports.schema_provider import SchemaProvider
from domain.ports.sql_executor import SqlExecutor
from domain.query import Query
from domain.retrieval.evidence import Evidence
from domain.retrieval.sql.sql_guard import SqlGuard
from domain.routing.route import Route

_SYSTEM = (
    "You translate a user's question into a single read-only SQLite SELECT "
    "query. Use only the provided tables and columns. Return SQL only, no prose."
)
_FENCE_RE = re.compile(r"^```(?:sql)?|```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


class SqlStrategy:
    """Text-to-SQL. Coordinates ports only; knows nothing about which LLM or
    database is behind them. Order: read schema -> generate -> guard -> execute."""

    route = Route.SQL
    description = (
        "Answers questions needing counts, sums, averages, filtering, or grouping "
        "over structured tables (orders, customers, revenue, dates)."
    )

    def __init__(
        self,
        llm: LanguageModel,
        schema: SchemaProvider,
        executor: SqlExecutor,
        guard: SqlGuard,
    ) -> None:
        self._llm = llm
        self._schema = schema
        self._executor = executor
        self._guard = guard

    async def retrieve(self, query: Query) -> Evidence:
        catalog = await self._schema.schema()
        prompt = f"Schema:\n{catalog.to_prompt()}\n\nQuestion: {query.text}\n\nSQL:"
        raw = await self._llm.complete(prompt, system=_SYSTEM)
        sql = self._guard.enforce(_strip_fences(raw))
        result = await self._executor.execute(sql)
        return Evidence(tables=(result,), routes_used=(Route.SQL,))
