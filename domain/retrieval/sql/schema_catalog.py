from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[ColumnInfo, ...]


@dataclass(frozen=True)
class SchemaCatalog:
    tables: tuple[TableInfo, ...]

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tables)

    def to_prompt(self) -> str:
        """Compact rendering handed to the LLM during SQL generation."""
        return "\n".join(
            f"{t.name}({', '.join(f'{c.name} {c.type}' for c in t.columns)})"
            for t in self.tables
        )
