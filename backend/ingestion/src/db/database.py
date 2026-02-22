"""Database backend abstraction & factory.

Currently only a MongoDB backend is implemented. Additional backends (e.g., PostgreSQL)
can add support by implementing `upsert_contract_chunks` and wiring into
`get_database_backend`.
"""

from __future__ import annotations

from typing import Protocol, List, Dict, Any

from ..util.logger import get_logger

logger = get_logger("db.factory")


class DatabaseBackend(Protocol):  # pragma: no cover - structural protocol
    def upsert_contract_chunks(self, docs: List[Dict[str, Any]]) -> None: ...


def get_database_backend() -> DatabaseBackend:
    from .mongo import MongoDatabase, have_mongo  # lazy import

    if not have_mongo():
        raise RuntimeError("MongoDB backend required: MONGODB_ATLAS_URI not set.")
    return MongoDatabase()


__all__ = ["DatabaseBackend", "get_database_backend"]
