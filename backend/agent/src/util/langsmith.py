"""LangSmith/LangChain tracing helpers for optional instrumentation. All functions are safe-noops if tracing env vars are not set."""

from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List, Optional

_LS_CLIENT = None  # cached LangSmith client


def trace_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING_V2", "false").lower() == "true"


def decision_logging_enabled() -> bool:
    return os.getenv("AE_LANGSMITH_LOG_DECISIONS", "false").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


def get_client():
    """Return cached LangSmith client if tracing & logging enabled and API key present.

    Safe for repeated calls; returns None if not configured. Errors during creation are swallowed.
    """
    global _LS_CLIENT
    # Fast path
    if _LS_CLIENT is not None:
        return _LS_CLIENT
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return None
    if not (trace_enabled() or decision_logging_enabled()):
        return None
    try:
        from langsmith import Client  # type: ignore

        _LS_CLIENT = Client(
            api_key=api_key,
            api_url=os.getenv("LANGSMITH_ENDPOINT") or None,
        )
    except Exception:
        _LS_CLIENT = None
    return _LS_CLIENT


def safe_log_decision(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Emit a custom run representing a decision event.

    Returns the created run id or None.
    """
    if not decision_logging_enabled():
        return None
    client = get_client()
    if client is None:
        return None
    try:
        run_name = "clinical_ae_decision"
        run_id = str(uuid.uuid4())
        client.create_run(
            name=run_name,
            run_type="chain",
            inputs=inputs,
            outputs=outputs,
            tags=tags or ["decision"],
            metadata=metadata or {},
            id=run_id,
            project_name=os.getenv("LANGSMITH_PROJECT") or None,
        )
        return run_id
    except Exception:
        return None


__all__ = [
    "safe_log_decision",
    "decision_logging_enabled",
    "trace_enabled",
    "get_client",
]
