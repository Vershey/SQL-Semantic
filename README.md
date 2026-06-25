# Router + Text-to-SQL + Semantic Retrieval

A reference architecture for a system that answers natural-language questions by
**routing** each query to the right retrieval strategy: **Text-to-SQL** for
structured data, **semantic retrieval** for unstructured documents, or both
(hybrid). It runs end-to-end out of the box with no external services, using
SQLite, an in-memory vector store, and a deterministic fake LLM. Swapping in a
real model, database, or vector DB is a one-adapter change.

## The idea

The router is a thin dispatcher. Text-to-SQL and semantic retrieval are two
interchangeable retrieval strategies behind one interface, and each emits the
same normalized `Evidence`. A separate synthesizer turns evidence into the final
answer with citations. Because everything produces `Evidence`, adding a strategy,
running strategies in parallel, and combining them for hybrid answers all come
for free.

```
Query -> Router -> { SQL strategy | Semantic strategy } -> Evidence -> Synthesizer -> Answer
```

## Layout (hexagonal / ports & adapters)

`domain` holds the pure logic and the ports (interfaces); it imports no
infrastructure. `infrastructure` implements those ports. `application` is thin
orchestration. `interface` is the entry points. `bootstrap` wires everything.

```
domain/
  query.py                     Query, ConversationContext
  routing/route.py             Route, RouteDecision
  routing/router.py            Router (port)
  retrieval/strategy.py        RetrievalStrategy (port)
  retrieval/evidence.py        Evidence, Passage, TabularResult, Provenance
  retrieval/sql/               SqlStrategy, SchemaCatalog, SqlGuard
  retrieval/semantic/          SemanticStrategy
  synthesis/synthesizer.py     Synthesizer (port), Answer, Citation
  ports/                       LanguageModel, Embedder, VectorStore,
                               SqlExecutor, SchemaProvider
application/
  answer_question.py           AnswerQuestionUseCase (the orchestrator)
  strategy_registry.py         Route -> RetrievalStrategy
  config.py                    Config (env-driven)
infrastructure/
  llm/                         FakeLLM (default), AnthropicLLM
  embeddings/                  HashingEmbedder
  vector_store/                InMemoryVectorStore
  database/                    SqliteExecutor, SqliteSchemaProvider
  routing/                     LlmRouter
  synthesis/                   LlmSynthesizer
interface/
  api/app.py                   FastAPI, POST /ask
  cli/main.py                  command-line entry
bootstrap/
  container.py                 composition root: build_use_case()
  seed.py                      demo orders table + demo documents
```

## Run it

```bash
pip install -r requirements.txt

# CLI (run from the project root)
python -m interface.cli.main "How many orders did we get in 2024?"
python -m interface.cli.main "What is our refund policy?"
python -m interface.cli.main "Show revenue by month"

# HTTP API
uvicorn interface.api.app:app --reload
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was our total revenue in 2024?"}'

# Tests
pytest
```

## Swapping in real infrastructure

Each piece is a port with an adapter; replace the adapter in `bootstrap/container.py`.

- **Real model**: `pip install anthropic`, set `ANTHROPIC_API_KEY`, and run with
  `USE_REAL_ANTHROPIC=1`. The container then uses `AnthropicLLM` instead of `FakeLLM`.
- **Postgres** instead of SQLite: write `PostgresExecutor` (SELECT-only role) and
  `PostgresSchemaProvider` (introspect `information_schema`) against the existing
  `SqlExecutor` / `SchemaProvider` ports.
- **pgvector / Pinecone / Qdrant**: write a `VectorStore` adapter; the
  `SemanticStrategy` and use case don't change.
- **Real embeddings**: write an `Embedder` adapter calling your embedding model.

## Design notes

- **`SqlGuard` is a security boundary**, not a prompt instruction: single
  statement, read-only, no DDL/DML, table allowlist, mandatory `LIMIT`. The
  SQLite adapter also opens the connection read-only as defense in depth. A
  production guard should parse the SQL AST (e.g. sqlglot) rather than use regex.
- **Routing quality depends on the strategy `description`s** the router is given,
  plus the schema summary. A router that doesn't know a table exists will never
  route to it.
- **Citations come from `Evidence` provenance**, derived deterministically rather
  than from model prose.
- **Evaluation** should score three failure modes independently: did it route
  correctly, did the SQL return the right rows, did retrieval surface the right
  documents.
