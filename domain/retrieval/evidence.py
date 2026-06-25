from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Iterable

from domain.routing.route import Route


@dataclass(frozen=True)
class Provenance:
    """Where a piece of evidence came from, so the synthesizer can cite it."""
    source: str               # table name, document id, ...
    locator: str              # the SQL string, the chunk id, a page number
    score: float | None = None


@dataclass(frozen=True)
class Passage:
    text: str
    provenance: Provenance


@dataclass(frozen=True)
class TabularResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    generated_sql: str
    provenance: Provenance


@dataclass(frozen=True)
class Evidence:
    """The normalized output of *any* retrieval strategy.

    Both Text-to-SQL and semantic retrieval produce this single type, which is
    what lets the synthesizer stay strategy-agnostic and what makes hybrid
    answers a simple concatenation rather than a special case.
    """
    passages: tuple[Passage, ...] = ()
    tables: tuple[TabularResult, ...] = ()
    routes_used: tuple[Route, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.passages and not self.tables

    def merged_with(self, other: "Evidence") -> "Evidence":
        return Evidence(
            passages=self.passages + other.passages,
            tables=self.tables + other.tables,
            routes_used=self.routes_used + other.routes_used,
        )

    @staticmethod
    def merge(items: Iterable["Evidence"]) -> "Evidence":
        return reduce(lambda a, b: a.merged_with(b), items, Evidence())
