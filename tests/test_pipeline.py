import asyncio

from application.config import Config
from bootstrap.container import build_use_case
from domain.query import Query
from domain.retrieval.sql.sql_guard import SqlGuard, SqlGuardError


def test_guard_blocks_writes():
    guard = SqlGuard(allowed_tables=frozenset({"orders"}))
    for bad in ("DELETE FROM orders", "DROP TABLE orders", "UPDATE orders SET amount = 0"):
        try:
            guard.enforce(bad)
            assert False, f"expected SqlGuardError for: {bad}"
        except SqlGuardError:
            pass


def test_guard_blocks_unknown_table():
    guard = SqlGuard(allowed_tables=frozenset({"orders"}))
    try:
        guard.enforce("SELECT * FROM secrets")
        assert False, "expected SqlGuardError for unknown table"
    except SqlGuardError:
        pass


def test_guard_injects_limit():
    guard = SqlGuard(allowed_tables=frozenset({"orders"}), max_rows=50)
    out = guard.enforce("SELECT id FROM orders")
    assert "limit 50" in out.lower()


def test_sql_route_end_to_end(tmp_path):
    config = Config(db_path=str(tmp_path / "test.db"))
    use_case = asyncio.run(build_use_case(config))
    answer = asyncio.run(use_case.execute(Query("How many orders did we get in 2024?")))
    assert answer.decision is not None
    assert "sql" in [r.value for r in answer.decision.routes]
    assert not answer.is_clarification
    assert answer.citations


def test_semantic_route_end_to_end(tmp_path):
    config = Config(db_path=str(tmp_path / "test.db"))
    use_case = asyncio.run(build_use_case(config))
    answer = asyncio.run(use_case.execute(Query("What is our refund policy?")))
    assert answer.decision is not None
    assert "semantic" in [r.value for r in answer.decision.routes]
    assert answer.citations
