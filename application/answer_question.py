from __future__ import annotations

import asyncio
from typing import Mapping

from domain.query import Query
from domain.retrieval.evidence import Evidence
from domain.retrieval.strategy import RetrievalStrategy
from domain.routing.route import Route
from domain.routing.router import Router
from domain.synthesis.synthesizer import Answer, Synthesizer


class AnswerQuestionUseCase:
    """The orchestrator: route -> gather evidence (in parallel for hybrid) ->
    synthesize. Deliberately thin; all dependencies are ports."""

    def __init__(
        self,
        router: Router,
        strategies: Mapping[Route, RetrievalStrategy],
        synthesizer: Synthesizer,
    ) -> None:
        self._router = router
        self._strategies = strategies
        self._synthesizer = synthesizer

    async def execute(self, query: Query) -> Answer:
        decision = await self._router.route(query)
        if Route.CLARIFY in decision.routes:
            return Answer.clarification(
                decision.rationale or "Could you add detail so I know where to look?"
            )

        runnable = [r for r in decision.routes if r in self._strategies]
        if not runnable:
            return Answer.clarification("I couldn't match this question to a data source.")

        results = await asyncio.gather(
            *(self._strategies[route].retrieve(query) for route in runnable)
        )
        evidence = Evidence.merge(results)
        return await self._synthesizer.synthesize(query, evidence, decision)
