from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    db_path: str = "demo.db"
    semantic_top_k: int = 6
    sql_max_rows: int = 1000
    use_real_anthropic: bool = False
    anthropic_model: str = "claude-sonnet-4-6"
    embedding_dim: int = 256

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            db_path=os.getenv("DB_PATH", "demo.db"),
            semantic_top_k=int(os.getenv("SEMANTIC_TOP_K", "6")),
            sql_max_rows=int(os.getenv("SQL_MAX_ROWS", "1000")),
            use_real_anthropic=os.getenv("USE_REAL_ANTHROPIC", "0").lower() in ("1", "true", "yes"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "256")),
        )
