"""Vector snippet retrieval helpers.

Thin wrappers over Mongo vector search (sync + async) with consistent logging and
formatted join helper for downstream LLM consumption.
"""

from typing import List, Dict
from ..db.mongo import search, search_async
from ..util.logger import get_logger

logger = get_logger("retriever")

import os


def _default_top_k() -> int:
    try:
        return int(os.getenv("AE_EMBEDDING_TOP_K", "6"))
    except Exception:
        return 6


def retrieve_contract_snippets(therapy: str, query: str, k: int = None) -> List[Dict]:
    """Synchronous vector search for contract snippets."""
    if k is None:
        k = _default_top_k()
    logger.info(f"retrieve snippets therapy={therapy} q='{query[:80]}...' k={k}")
    return search(query=query, therapy=therapy, k=k)


async def retrieve_contract_snippets_async(
    therapy: str, query: str, k: int = None
) -> List[Dict]:
    """Async vector search for contract snippets."""
    if k is None:
        k = _default_top_k()
    logger.info(f"retrieve async snippets therapy={therapy} q='{query[:80]}...' k={k}")
    return await search_async(query=query, therapy=therapy, k=k)


def join_snippets(snips: List[Dict]) -> str:
    """Join top snippet texts into a readable context block."""
    return "\n".join(
        f"[{s.get('contract_id')}#{s.get('chunk_id')} score={round(s.get('_score', 0.0), 3)}] {s.get('text', '').strip()}"
        for s in snips[:10]
    )


__all__ = [
    "retrieve_contract_snippets",
    "retrieve_contract_snippets_async",
    "join_snippets",
]
