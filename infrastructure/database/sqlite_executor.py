from __future__ import annotations

import sqlite3

from domain.retrieval.evidence import Provenance, TabularResult


class SqliteExecutor:
    """Executes SQL against SQLite in read-only mode (defense in depth alongside
    SqlGuard). For production, use a thread pool / aiosqlite so it doesn't block
    the event loop, and a DB role with SELECT-only grants."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    async def execute(self, sql: str) -> TabularResult:
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            cursor = conn.execute(sql)
            columns = tuple(d[0] for d in (cursor.description or ()))
            rows = tuple(tuple(r) for r in cursor.fetchall())
        finally:
            conn.close()
        return TabularResult(
            columns=columns,
            rows=rows,
            generated_sql=sql,
            provenance=Provenance(source="database", locator=sql),
        )
