"""Centralized singleton accessors for shared clients.

Expose stable, cached instances for expensive resources so the API layer can import from one place.
All getters are idempotent and safe for concurrent reads (locking handled in source modules).
"""

from __future__ import annotations

# Re-export existing singletons implemented in their own modules to avoid circular imports.
from ..db.mongo import get_mongo_client  # noqa: F401
from ..models.embeddings import get_embedding_client  # noqa: F401
from ..models.llm import get_llm  # noqa: F401
from .phi_redact import redact_phi  # noqa: F401
from .langsmith import get_client as get_langsmith_client  # noqa: F401

__all__ = [
    "get_mongo_client",
    "get_embedding_client",
    "get_llm",
    "get_langsmith_client",
    "redact_phi",
]
