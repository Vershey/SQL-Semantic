from __future__ import annotations

import json
import re

# Signals the deterministic router uses. In production the real LLM replaces all
# of this; here it lets the whole pipeline run end-to-end with no API key.
_SQL_SIGNALS = (
    "how many", "count", "number of", "total", "sum", "revenue", "sales",
    "average", "avg", "mean", "orders", "order", "customers", "customer",
    "amount", "top", "highest", "lowest", "most", "least", "per month",
    "by month", "monthly", "in 20", "since 20", "this year",
)
_DOC_SIGNALS = (
    "policy", "policies", "refund", "return", "returns", "shipping", "ship",
    "delivery", "deliver", "warranty", "guide", "documentation", "docs",
    "faq", "eligibility", "eligible", "explain", "definition", "define",
    "sku", "how do i", "what does",
)


def _question(prompt: str) -> str:
    m = re.search(r"Question:\s*(.+)", prompt)
    return (m.group(1).strip() if m else prompt).lower()


class FakeLLM:
    """Deterministic stand-in for a real model. Dispatches on the system prompt:
    routing -> JSON decision, SQL gen -> a SELECT, synthesis -> a short answer."""

    async def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024
    ) -> str:
        s = (system or "").lower()
        if "router" in s:
            return self._route(prompt)
        if "sqlite" in s or "select" in s:
            return self._sql(prompt)
        return self._synthesize(prompt)

    # --- routing ---
    def _route(self, prompt: str) -> str:
        q = _question(prompt)
        sql_hits = sum(1 for k in _SQL_SIGNALS if k in q)
        doc_hits = sum(1 for k in _DOC_SIGNALS if k in q)
        if sql_hits and doc_hits:
            routes, rationale = ["sql", "semantic"], "Question spans structured and document data."
        elif sql_hits:
            routes, rationale = ["sql"], "Looks like an aggregate/filter over tables."
        elif doc_hits:
            routes, rationale = ["semantic"], "Looks like a documentation/FAQ question."
        else:
            routes, rationale = ["clarify"], "Not enough signal to pick a data source."
        confidence = 0.85 if (sql_hits or doc_hits) else 0.2
        return json.dumps({"routes": routes, "confidence": confidence, "rationale": rationale})

    # --- text-to-sql ---
    def _sql(self, prompt: str) -> str:
        q = _question(prompt)
        ym = re.search(r"\b(20\d\d)\b", q)
        where = ""
        if ym:
            y = int(ym.group(1))
            where = f" WHERE created_at >= '{y}-01-01' AND created_at < '{y + 1}-01-01'"
        if any(k in q for k in ("by month", "per month", "monthly")):
            return (
                "SELECT strftime('%Y-%m', created_at) AS month, "
                f"SUM(amount) AS revenue FROM orders{where} "
                "GROUP BY month ORDER BY month"
            )
        if any(k in q for k in ("how many", "count", "number of")):
            return f"SELECT COUNT(*) AS order_count FROM orders{where}"
        if any(k in q for k in ("total", "sum", "revenue", "sales")):
            return f"SELECT SUM(amount) AS total_revenue FROM orders{where}"
        if any(k in q for k in ("average", "avg", "mean")):
            return f"SELECT AVG(amount) AS avg_order_value FROM orders{where}"
        if any(k in q for k in ("top", "highest", "most", "largest")):
            return f"SELECT id, customer, amount FROM orders{where} ORDER BY amount DESC"
        return f"SELECT id, customer, amount, created_at FROM orders{where} ORDER BY created_at DESC"

    # --- synthesis ---
    def _synthesize(self, prompt: str) -> str:
        lines = prompt.splitlines()
        tables = [ln.split("|", 1)[1].strip() for ln in lines if ln.startswith("TABLE |")]
        docs = [ln.split("|", 1)[1].strip() for ln in lines if ln.startswith("DOC |")]
        parts: list[str] = []
        if tables:
            parts.append("From the data: " + "; ".join(tables) + ".")
        if docs:
            parts.append("From the documentation: " + docs[0])
        return " ".join(parts) if parts else "I couldn't find relevant information."
