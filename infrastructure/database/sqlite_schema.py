from __future__ import annotations

import sqlite3

from domain.retrieval.sql.schema_catalog import ColumnInfo, SchemaCatalog, TableInfo


class SqliteSchemaProvider:
    """Introspects the live SQLite schema and caches it (schema rarely changes
    within a process; re-fetching it on every query is wasteful)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._cache: SchemaCatalog | None = None

    async def schema(self) -> SchemaCatalog:
        if self._cache is not None:
            return self._cache
        conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            tables = []
            for name in names:
                cols = tuple(
                    ColumnInfo(name=c[1], type=(c[2] or "TEXT"))
                    for c in conn.execute(f"PRAGMA table_info({name})")
                )
                tables.append(TableInfo(name=name, columns=cols))
        finally:
            conn.close()
        self._cache = SchemaCatalog(tuple(tables))
        return self._cache
