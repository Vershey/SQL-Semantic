from __future__ import annotations

import re
from dataclasses import dataclass


class SqlGuardError(Exception):
    """Raised when generated SQL fails validation. The strategy lets this propagate."""


_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "replace", "grant", "revoke", "attach", "detach", "pragma", "vacuum",
    "merge", "exec",
)
_TABLE_RE = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)
_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


@dataclass
class SqlGuard:
    """Security boundary for LLM-generated SQL.

    Enforces: single statement, read-only (SELECT/WITH), no DDL/DML keywords,
    table allowlist, and a mandatory row limit. This is intentionally a
    first-class domain object rather than a prompt instruction the model can
    ignore. A production version should parse the AST (e.g. sqlglot) instead of
    relying on regexes for table extraction.
    """

    allowed_tables: frozenset[str]
    max_rows: int = 1000

    def enforce(self, sql: str) -> str:
        cleaned = _COMMENT_RE.sub(" ", sql).strip().rstrip(";").strip()
        if not cleaned:
            raise SqlGuardError("Empty query.")
        if ";" in cleaned:
            raise SqlGuardError("Only a single statement is allowed.")

        lowered = cleaned.lower()
        if not (lowered.startswith("select") or lowered.startswith("with")):
            raise SqlGuardError("Only read-only SELECT queries are allowed.")
        for kw in _FORBIDDEN:
            if re.search(rf"\b{kw}\b", lowered):
                raise SqlGuardError(f"Disallowed keyword: {kw}.")

        referenced = {m.group(1).lower() for m in _TABLE_RE.finditer(cleaned)}
        allowed = {t.lower() for t in self.allowed_tables}
        unknown = referenced - allowed
        if unknown:
            raise SqlGuardError(
                f"Query references tables outside the allowlist: {sorted(unknown)}."
            )

        if not _LIMIT_RE.search(lowered):
            cleaned = f"{cleaned} LIMIT {self.max_rows}"
        return cleaned
