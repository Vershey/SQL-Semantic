from __future__ import annotations

from typing import Protocol

from domain.retrieval.sql.schema_catalog import SchemaCatalog


class SchemaProvider(Protocol):
    async def schema(self) -> SchemaCatalog: ...
