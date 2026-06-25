from __future__ import annotations

import json
import re
from typing import Mapping

from domain.ports.llm import LanguageModel
from domain.retrieval.strategy import RetrievalStrategy
from domain.query import Query
from domain.routing.route import Route, RouteDecision

_SYSTEM = "You are a query router. Decide which data sources can answer the question."
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> str:
    m = _JSON_RE.search(raw)
    return m.group(0) if m else raw


class LlmRouter:
    """Default Router adapter. Builds a routing prompt from each strategy's
    `description` and asks the model to pick one or more routes (or clarify)."""

    def __init__(
        self, llm: LanguageModel, strategies: Mapping[Route, RetrievalStrategy]
    ) -> None:
        self._llm = llm
        self._strategies = strategies

    async def route(self, query: Query) -> RouteDecision:
        sources = "\n".join(
            f"- {route.value}: {strategy.description}"
            for route, strategy in self._strategies.items()
        )
        prompt = (
            f"Question: {query.text}\n\n"
            f"Available data sources:\n{sources}\n\n"
            'Respond with JSON only: {"routes": [...], "confidence": <0-1>, '
            '"rationale": "..."}. Use multiple routes if both help. '
            'Use ["clarify"] if the question is too vague to answer.'
        )
        raw = await self._llm.complete(prompt, system=_SYSTEM)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> RouteDecision:
        try:
            data = json.loads(_extract_json(raw))
            valid = Route._value2member_map_
            routes = tuple(Route(r) for r in data.get("routes", []) if r in valid)
            if not routes:
                routes = (Route.CLARIFY,)
            return RouteDecision(
                routes=routes,
                confidence=float(data.get("confidence", 0.5)),
                rationale=str(data.get("rationale", "")),
            )
        except Exception:
            return RouteDecision(
                routes=(Route.CLARIFY,),
                confidence=0.0,
                rationale="Could not parse the routing decision.",
            )
