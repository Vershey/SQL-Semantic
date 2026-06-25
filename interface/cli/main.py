from __future__ import annotations

import asyncio
import sys

from application.config import Config
from bootstrap.container import build_use_case
from domain.query import Query


async def _run(question: str) -> None:
    use_case = await build_use_case(Config.from_env())
    answer = await use_case.execute(Query(text=question))
    print(answer.text)
    if answer.decision:
        routes = ", ".join(r.value for r in answer.decision.routes)
        print(f"\n[routes: {routes} | confidence {answer.decision.confidence:.2f}]")
    if answer.citations:
        print("[citations]")
        for c in answer.citations:
            print(f"  - {c.source}: {c.locator}")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How many orders did we get in 2024?"
    asyncio.run(_run(question))


if __name__ == "__main__":
    main()
