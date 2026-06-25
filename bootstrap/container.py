from __future__ import annotations

from application.answer_question import AnswerQuestionUseCase
from application.config import Config
from application.strategy_registry import build_registry
from bootstrap.seed import seed_sqlite, seed_vector_store
from domain.ports.llm import LanguageModel
from domain.retrieval.semantic.semantic_strategy import SemanticStrategy
from domain.retrieval.sql.sql_guard import SqlGuard
from domain.retrieval.sql.sql_strategy import SqlStrategy
from infrastructure.database.sqlite_executor import SqliteExecutor
from infrastructure.database.sqlite_schema import SqliteSchemaProvider
from infrastructure.embeddings.hashing_embedder import HashingEmbedder
from infrastructure.routing.llm_router import LlmRouter
from infrastructure.synthesis.llm_synthesizer import LlmSynthesizer
from infrastructure.vector_store.in_memory_store import InMemoryVectorStore


def _make_llm(config: Config) -> LanguageModel:
    if config.use_real_anthropic:
        from infrastructure.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(model=config.anthropic_model)
    from infrastructure.llm.fake_llm import FakeLLM

    return FakeLLM()


async def build_use_case(config: Config | None = None) -> AnswerQuestionUseCase:
    """Composition root: pick adapters, wire them to domain services, return the
    use case. This is the ONLY place that knows the concrete infrastructure."""
    config = config or Config.from_env()
    llm = _make_llm(config)

    # Structured side: seed, introspect schema, derive the SQL allowlist from it.
    seed_sqlite(config.db_path)
    schema_provider = SqliteSchemaProvider(config.db_path)
    catalog = await schema_provider.schema()
    guard = SqlGuard(
        allowed_tables=frozenset(catalog.table_names), max_rows=config.sql_max_rows
    )
    sql_strategy = SqlStrategy(
        llm=llm,
        schema=schema_provider,
        executor=SqliteExecutor(config.db_path),
        guard=guard,
    )

    # Unstructured side: embed and seed the vector store.
    embedder = HashingEmbedder(dim=config.embedding_dim)
    store = InMemoryVectorStore(embedder)
    await seed_vector_store(store)
    semantic_strategy = SemanticStrategy(
        embedder=embedder, store=store, k=config.semantic_top_k
    )

    registry = build_registry(sql_strategy, semantic_strategy)
    router = LlmRouter(llm, registry)
    synthesizer = LlmSynthesizer(llm)
    return AnswerQuestionUseCase(router=router, strategies=registry, synthesizer=synthesizer)
