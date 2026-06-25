from __future__ import annotations

from domain.ports.llm import LanguageModel
from domain.query import Query
from domain.retrieval.evidence import Evidence
from domain.routing.route import RouteDecision
from domain.synthesis.synthesizer import Answer, Citation

_SYSTEM = (
    "You are an assistant. Answer the question using only the supplied evidence. "
    "If the evidence is insufficient, say so plainly. Be concise and cite sources."
)


class LlmSynthesizer:
    """Composes an answer from merged evidence. Citations are derived directly from
    the Evidence (deterministic provenance), not from the model's prose."""

    def __init__(self, llm: LanguageModel) -> None:
        self._llm = llm

    async def synthesize(
        self, query: Query, evidence: Evidence, decision: RouteDecision
    ) -> Answer:
        if evidence.is_empty:
            return Answer(
                text="I don't have enough information to answer that.",
                decision=decision,
            )
        text = (await self._llm.complete(self._render(query, evidence), system=_SYSTEM)).strip()
        return Answer(text=text, citations=self._citations(evidence), decision=decision)

    @staticmethod
    def _render(query: Query, evidence: Evidence) -> str:
        lines = [f"Question: {query.text}", "", "Evidence:"]
        for t in evidence.tables:
            if len(t.rows) == 1 and len(t.columns) == 1:
                lines.append(f"TABLE | {t.columns[0]}={t.rows[0][0]}")
            elif len(t.rows) == 1:
                pairs = ", ".join(f"{c}={v}" for c, v in zip(t.columns, t.rows[0]))
                lines.append(f"TABLE | {pairs}")
            else:
                head = ", ".join(t.columns)
                sample = "; ".join(str(r) for r in t.rows[:5])
                lines.append(f"TABLE | {len(t.rows)} rows ({head}): {sample}")
        for p in evidence.passages[:4]:
            text = " ".join(p.text.split())
            if len(text) > 240:
                text = text[:240] + "..."
            lines.append(f"DOC | ({p.provenance.source}) {text}")
        return "\n".join(lines)

    @staticmethod
    def _citations(evidence: Evidence) -> tuple[Citation, ...]:
        seen: set[tuple[str, str]] = set()
        out: list[Citation] = []
        for t in evidence.tables:
            key = (t.provenance.source, t.provenance.locator)
            if key not in seen:
                seen.add(key)
                out.append(Citation(*key))
        for p in evidence.passages:
            key = (p.provenance.source, p.provenance.locator)
            if key not in seen:
                seen.add(key)
                out.append(Citation(*key))
        return tuple(out)
