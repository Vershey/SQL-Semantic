from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from application.config import Config
from bootstrap.container import build_use_case
from domain.query import Query


class AskRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    source: str
    locator: str


class AskResponse(BaseModel):
    answer: str
    is_clarification: bool
    routes: list[str]
    citations: list[CitationOut]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.use_case = await build_use_case(Config.from_env())
    yield


app = FastAPI(title="Router + Text-to-SQL + Semantic Retrieval", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    answer = await app.state.use_case.execute(Query(text=req.question))
    routes = [r.value for r in answer.decision.routes] if answer.decision else []
    return AskResponse(
        answer=answer.text,
        is_clarification=answer.is_clarification,
        routes=routes,
        citations=[CitationOut(source=c.source, locator=c.locator) for c in answer.citations],
    )
