from typing import TypedDict, List, Dict, Any
import os
import json
from pathlib import Path
from datetime import datetime
from langgraph.graph import StateGraph, END
from ..util.phi_redact import redact_phi
from ..rag.retriever import retrieve_contract_snippets, join_snippets
from ..rag.prompts import BASE_DECISION_HINT
from ..models.llm import decide_with_llm, Decision
from ..util.logger import get_logger

logger = get_logger("graph")


class AEState(TypedDict, total=False):
    therapy: str
    assessment_summary: str
    contract_context: str
    decision: Dict[str, Any]
    errors: List[str]


def node_parse_assessment(state: AEState) -> AEState:
    summary = state.get("assessment_summary", "")
    if os.getenv("AE_REDACT_PHI", "true").lower() == "true" and summary:
        red, _ = redact_phi(summary)
        summary = red
    return {**state, "assessment_summary": summary}


def node_retrieve(state: AEState) -> AEState:
    therapy = state.get("therapy", "")
    query = (state.get("assessment_summary", "") + "\n" + BASE_DECISION_HINT).strip()
    try:
        snips = retrieve_contract_snippets(therapy=therapy, query=query, k=6)
    except Exception as e:
        logger.error(f"Retrieval failed: {type(e).__name__}: {e}", exc_info=True)
        errs = state.get("errors", []) + [
            f"retrieval:{type(e).__name__}:{str(e)[:200]}"
        ]
        return {**state, "contract_context": "(retrieval error)", "errors": errs}
    return {
        **state,
        "contract_context": join_snippets(snips) if snips else "(no snippets)",
    }


def node_decide(state: AEState) -> AEState:
    # Fail fast: if retrieval had an error or produced no usable snippets, abort before LLM.
    errs = state.get("errors", [])
    ctx = state.get("contract_context", "")
    if any(e.startswith("retrieval:") for e in errs):
        raise RuntimeError("retrieval failure; aborting before LLM")
    if ctx in ("(retrieval error)", "(no snippets)"):
        raise RuntimeError("no contract snippets; aborting before LLM")

    d = decide_with_llm(
        therapy=state.get("therapy", ""),
        assessment_summary=state.get("assessment_summary", ""),
        contract_context=ctx,
    )
    decision_dict = d.model_dump()
    new_state = {**state, "decision": decision_dict}
    if os.getenv(
        "AE_PERSIST_DECISIONS", "true"
    ).lower() == "true" and not new_state.get("errors"):
        _persist_decision(new_state)
    return new_state


def _persist_decision(state: AEState) -> None:
    try:
        therapy = (state.get("therapy") or "unknown").replace("/", "-")
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(__file__).resolve().parents[2] / "test" / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "therapy": therapy,
            "timestamp_utc": ts,
            "decision": state.get("decision", {}),
            "contract_context_preview": (state.get("contract_context", "")[:2000]),
            "errors": state.get("errors", []),
        }
        fname = out_dir / f"decision_{therapy}_{ts}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"persist decision: {fname}")
    except Exception as exc:
        logger.error(f"persist error: {type(exc).__name__}: {exc}")


def build_graph():
    g = StateGraph(AEState)
    g.add_node("parse", node_parse_assessment)
    g.add_node("retrieve", node_retrieve)
    g.add_node("decide", node_decide)
    g.set_entry_point("parse")
    g.add_edge("parse", "retrieve")
    g.add_edge("retrieve", "decide")
    g.add_edge("decide", END)
    return g.compile()
