import os
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from anyio import to_thread
import asyncio

from ..util.fhir import extract_therapy_type, summarize_assessment, load_assessment
from ..util.assessment_rules import match_assessment
from ..util.env_loader import load_env
from ..rag.graph import build_graph
from ..util.logger import get_logger
from ..util.langsmith import safe_log_decision
from ..db.mongo import get_mongo_client
from ..util.oauth import is_azure_auth_configured

_env_file = load_env()
logger = get_logger("api")
if _env_file:
    logger.info(f"env loaded: {_env_file}")


async def wait_for_dependencies():
    """Wait for Mongo & LLM readiness.

    Env:
      AE_SKIP_STARTUP_WAIT=true  -> skip all checks
      AE_STARTUP_MAX_RETRIES=0   -> infinite retry
      AE_STARTUP_MAX_RETRIES>0   -> abort after N attempts
      AE_STARTUP_RETRY_SECONDS   -> seconds between attempts
      AE_LLM_STARTUP_PROMPT      -> optional custom probe text (default built-in)
    """
    if env_bool("AE_SKIP_STARTUP_WAIT", False):
        logger.info("startup wait skipped (AE_SKIP_STARTUP_WAIT=true)")
        return
    max_retries = int(os.getenv("AE_STARTUP_MAX_RETRIES", "0") or 0)
    delay = float(os.getenv("AE_STARTUP_RETRY_SECONDS", "5") or 5)
    provider = env_str("AE_LLM_PROVIDER", "azure").lower()
    mongo_configured = bool(os.getenv("AE_MONGO_URI", "").strip())
    attempt = 0
    probe_text = os.getenv("AE_LLM_STARTUP_PROMPT", "health readiness probe")

    def _check_mongo() -> bool:
        if not mongo_configured:
            return True
        try:
            client = get_mongo_client()
            client.admin.command("ping")
            return True
        except Exception as e:
            logger.warning(f"mongo ping failed attempt={attempt}: {type(e).__name__}")
            return False

    def _check_llm() -> tuple[bool, str]:
        try:
            from ..models.llm import get_llm
            from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

            temperature = float(os.getenv("AE_LLM_TEMPERATURE", "1"))
            max_tokens = int(os.getenv("AE_LLM_MAX_TOKENS", "128"))
            llm = get_llm(
                provider=provider,
                temperature=temperature,
                max_tokens=max_tokens,
                force_json=True,
            )
            try:
                response = llm.invoke(
                    [
                        SystemMessage(
                            content='Startup readiness check. Reply with a short valid JSON object like {"ok": true}.'
                        ),
                        HumanMessage(content=probe_text),
                    ]
                )
                if response is None:
                    raise ValueError("Forced JSON probe returned None")
                return True, "json"
            except Exception as e1:
                et1 = type(e1).__name__
                logger.error(f"Forced JSON probe failed: {et1} - {str(e1)}")
                # Fallback attempt: non-JSON client
                try:
                    llm_plain = get_llm(
                        provider=provider,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        force_json=False,
                    )
                    response = llm_plain.invoke(
                        [
                            HumanMessage(content=probe_text),
                        ]
                    )
                    if response is None or not isinstance(response, str):
                        raise ValueError("Fallback probe returned invalid response")
                    logger.warning(
                        f"LLM probe succeeded only after fallback (first failure {et1})"
                    )
                    return True, f"fallback:{et1}"
                except Exception as e2:
                    et2 = type(e2).__name__
                    logger.error(f"Fallback probe failed: {et2} - {str(e2)}")
                    return False, f"fail:{et1}->{et2}"
        except Exception as e:
            return False, f"construct:{type(e).__name__}"

    while True:
        attempt += 1
        mongo_ok = await to_thread.run_sync(_check_mongo)
        llm_ok, llm_detail = await to_thread.run_sync(_check_llm)
        if mongo_ok and llm_ok:
            logger.info(
                f"startup dependencies ready attempt={attempt} llm={llm_detail}"
            )
            return
        if max_retries and attempt >= max_retries:
            raise RuntimeError(
                f"Startup dependencies not ready after {attempt} attempts: mongo_ok={mongo_ok} llm_ok={llm_ok} detail={llm_detail}"
            )
        logger.info(
            f"dependencies not ready attempt={attempt} mongo_ok={mongo_ok} llm_ok={llm_ok} detail={llm_detail}; retrying in {delay}s"
        )
        await asyncio.sleep(delay)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await wait_for_dependencies()
    yield


app = FastAPI(
    title="Clinical Adverse Event Detection API", version="0.3.1", lifespan=lifespan
)

# Mount static files for UI
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

"""
API Endpoints:

POST /assessment
    Classify a FHIR QuestionnaireResponse as an adverse event. Returns decision, contract context, and trace info.

POST /assessment/source
    Classify an assessment from inline, file, or S3 source. Returns same structure as /assessment.

GET /health
    Liveness check. Returns {"status": "alive"}.

GET /ready
    Readiness check for Mongo, embeddings, LLM, and auth. Returns status and subsystem checks.

GET /
    Root info: API name, version, endpoints, and persistence flag.
"""


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")


def env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


class QuestionnaireResponse(BaseModel):
    """FHIR QuestionnaireResponse input for assessment endpoints."""

    resourceType: str = Field(default="QuestionnaireResponse")
    id: str | None = None
    questionnaireTitle: str | None = None
    status: str | None = None
    partOf: Dict[str, Any] | None = None
    item: list[Dict[str, Any]] | None = None

    class Config:
        extra = "allow"


class DecisionOut(BaseModel):
    """API response: decision, contract context, and trace info."""

    therapy: str
    decision: Dict[str, Any]
    contract_context_preview: str
    source: str
    filtered_important_questions: bool = False
    trace_run_id: str | None = Field(default=None, description="LangSmith run id")


class AssessmentSource(BaseModel):
    """Input for /assessment/source: supports inline, file, or S3 assessment sources."""

    source_type: str = Field(..., description="inline | file | s3")
    assessment: Dict[str, Any] | None = None
    path: str | None = Field(None, description="File path if source_type=file")
    s3_bucket: str | None = None
    s3_key: str | None = None
    filter_important: bool = False
    use_keywords: bool = True

    class Config:
        extra = "forbid"


_graph = None


def get_graph():
    """Singleton accessor for the workflow graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _validate_and_summarize(
    data: Dict[str, Any], filter_important: bool, use_keywords: bool
) -> tuple[str, str]:
    """Common validation + summarization shared by sync/async pipelines."""
    provider = env_str("AE_LLM_PROVIDER", "azure").lower()
    if provider == "azure" and not is_azure_auth_configured():
        raise HTTPException(
            status_code=500,
            detail="LLM credentials missing (OAuth or AE_AZURE_API_KEY)",
        )
    therapy = extract_therapy_type(data)
    if not therapy:
        raise HTTPException(
            status_code=400, detail="therapyType not found in partOf.identifier"
        )
    # DISABLED: Assessment validation check
    # if not match_assessment(data):
    #     raise HTTPException(
    #         status_code=422, detail="Assessment does not match processing rules"
    #     )
    summary = summarize_assessment(
        data, filter_important=filter_important, therapy_type=therapy
    )
    return therapy, summary or ""


async def process_assessment(
    data: Dict[str, Any],
    filter_important: bool,
    use_keywords: bool,
    async_pipeline: bool,
) -> DecisionOut:
    """Unified assessment processing selecting graph (sync) or async manual pipeline.

    async_pipeline=True executes retrieval + LLM directly with async primitives.
    Otherwise uses the pre-built LangGraph workflow (invoked in a worker thread).
    """
    # Run common validation/summarization in thread if async to avoid blocking loop
    if async_pipeline:
        therapy, summary = await to_thread.run_sync(
            _validate_and_summarize, data, filter_important, use_keywords
        )
    else:
        therapy, summary = _validate_and_summarize(data, filter_important, use_keywords)

    if async_pipeline:
        # Async pipeline: manual retrieval + LLM decision
        from ..rag.retriever import retrieve_contract_snippets_async, join_snippets

        try:
            snips = await retrieve_contract_snippets_async(
                therapy=therapy, query=summary, k=6
            )
            contract_ctx = join_snippets(snips) if snips else "(no snippets)"
        except Exception as e:  # retrieval failure
            raise HTTPException(status_code=500, detail=f"retrieval:{type(e).__name__}")
        if contract_ctx in ("(no snippets)"):
            raise HTTPException(
                status_code=503, detail="No contract snippets available"
            )
        from ..models.llm import decide_with_llm_async

        try:
            decision_obj = await decide_with_llm_async(
                therapy=therapy,
                assessment_summary=summary,
                contract_context=contract_ctx,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"llm:{type(e).__name__}")
        decision_payload = decision_obj.model_dump()
        run_id = safe_log_decision(
            inputs={
                "therapy": therapy,
                "filter_important": filter_important,
                "use_keywords": use_keywords,
                "summary_len": len(summary),
            },
            outputs=decision_payload,
            tags=["endpoint", "assessment", "async"],
            metadata={
                "path": "/assessment",
                "version": app.version,
                "async_pipeline": True,
            },
        )
        return DecisionOut(
            therapy=therapy,
            decision=decision_payload,
            contract_context_preview=contract_ctx,
            source="inline",
            filtered_important_questions=filter_important,
            trace_run_id=run_id,
        )

    # Sync graph pipeline: invoke LangGraph to handle retrieval + decision
    graph = get_graph()
    state = {
        "therapy": therapy,
        "assessment_summary": summary,
        "contract_context": "",
        "decision": {},
        "errors": [],
    }
    try:
        result = await to_thread.run_sync(graph.invoke, state)
    except Exception as e:
        logger.exception("Graph invocation failed")
        raise HTTPException(
            status_code=500, detail=f"Processing error: {type(e).__name__}"
        )
    decision_payload = result.get("decision", {})
    run_id = safe_log_decision(
        inputs={
            "therapy": therapy,
            "filter_important": filter_important,
            "use_keywords": use_keywords,
            "summary_len": len(summary),
        },
        outputs=decision_payload,
        tags=["endpoint", "assessment"],
        metadata={"path": "/assessment", "version": app.version},
    )
    return DecisionOut(
        therapy=therapy,
        decision=decision_payload,
        contract_context_preview=result.get("contract_context", ""),
        source="inline",
        filtered_important_questions=filter_important,
        trace_run_id=run_id,
    )


@app.post("/assessment", response_model=DecisionOut, tags=["assessment"])
async def assess(
    payload: QuestionnaireResponse,
    filter_important: Optional[bool] = None,
    use_keywords: Optional[bool] = None,
    request: Request = None,
):
    """POST /assessment: classify a FHIR QuestionnaireResponse as an adverse event."""
    fi = (
        env_bool("AE_FILTER_IMPORTANT_DEFAULT", True)
        if filter_important is None
        else filter_important
    )
    uk = (
        env_bool("AE_FILTER_USE_KEYWORDS_DEFAULT", True)
        if use_keywords is None
        else use_keywords
    )
    async_pipeline = env_bool("AE_ASYNC_PIPELINE", False)
    return await process_assessment(payload.model_dump(), fi, uk, async_pipeline)


@app.post("/assessment/source", response_model=DecisionOut, tags=["assessment"])
async def assess_source(src: AssessmentSource, request: Request = None):
    """
    POST /assessment/source
    Classify an assessment from inline, file, or S3 source.
    Request: AssessmentSource JSON
    Response: DecisionOut JSON
    """
    provider = env_str("AE_LLM_PROVIDER", "azure").lower()
    if provider == "azure" and not is_azure_auth_configured():
        raise HTTPException(
            status_code=500,
            detail="LLM credentials missing (OAuth or AE_AZURE_API_KEY)",
        )
    if src.source_type == "inline":
        if not src.assessment:
            raise HTTPException(
                status_code=400,
                detail="assessment payload required for inline source_type",
            )
        data = src.assessment
        origin = "inline"
    elif src.source_type == "file":
        if not src.path:
            raise HTTPException(
                status_code=400, detail="path required for file source_type"
            )
        p = Path(src.path)
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {src.path}")
        data = load_assessment(str(p))
        origin = f"file:{p.name}"
    elif src.source_type == "s3":
        if not src.s3_bucket or not src.s3_key:
            raise HTTPException(
                status_code=400,
                detail="s3_bucket and s3_key required for s3 source_type",
            )
        try:
            import boto3  # optional dependency
        except ImportError:
            raise HTTPException(
                status_code=500, detail="boto3 not installed; cannot fetch from S3"
            )
        s3 = boto3.client("s3")
        try:
            obj = s3.get_object(Bucket=src.s3_bucket, Key=src.s3_key)
            import json

            data = json.loads(obj["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.exception("Error fetching S3 object")
            raise HTTPException(
                status_code=500, detail=f"S3 fetch failed: {type(e).__name__}"
            )
        origin = f"s3:{src.s3_bucket}/{src.s3_key}"
    else:
        raise HTTPException(
            status_code=400, detail="Unsupported source_type. Use inline | file | s3"
        )

    async_pipeline = env_bool("AE_ASYNC_PIPELINE", False)
    out = await process_assessment(
        data, src.filter_important, src.use_keywords, async_pipeline
    )
    out.source = origin
    return out


@app.get("/health", tags=["ops"])
async def health():
    """
    GET /health
    Liveness check. Returns {"status": "alive"}.
    """
    return {"status": "alive"}


@app.get("/ready", tags=["ops"])
async def ready():
    """
    GET /ready
    Readiness check for Mongo, embeddings, LLM, and auth.
    Returns status and subsystem checks.
    """
    checks = {}
    # Mongo readiness
    uri = os.getenv("AE_MONGO_URI")
    if uri:
        try:
            client = get_mongo_client()
            await to_thread.run_sync(lambda: client.admin.command("ping"))
            checks["mongo"] = "ok"
        except Exception as e:
            checks["mongo"] = f"error:{type(e).__name__}"
    else:
        checks["mongo"] = "not-configured"
    # Embeddings readiness
    checks["embeddings"] = (
        "ok"
        if os.getenv("AE_EMBED_DEPLOYMENT") and is_azure_auth_configured()
        else "not-configured"
    )
    # LLM readiness
    provider = env_str("AE_LLM_PROVIDER", "azure")
    if provider == "azure":
        llm_ok = (
            all([os.getenv("AE_AZURE_CHAT_DEPLOYMENT"), os.getenv("AE_AZURE_ENDPOINT")])
            and is_azure_auth_configured()
        )
        checks["llm"] = "ok" if llm_ok else "not-configured"
    else:
        checks["llm"] = "ok"  # assume local llama
    # Auth open (internal-only)
    checks["auth"] = "open"
    overall = all(v == "ok" or v == "open" for v in checks.values())
    return {"status": "ready" if overall else "degraded", "checks": checks}


# UI Routes
@app.get("/ui", tags=["ui"])
async def ui_home():
    """GET /ui - Serve the assessment UI home page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="UI not found")


@app.get("/api/test-assessments", tags=["ui"])
async def list_test_assessments():
    """GET /api/test-assessments - List all available test assessment files."""
    test_data_dir = (
        Path(__file__).parent.parent.parent / "test" / "data" / "assessments"
    )

    if not test_data_dir.exists():
        return {"assessments": []}

    try:
        files = sorted(
            [
                {
                    "name": f.stem,
                    "file": f.name,
                    "description": f"Test assessment: {f.stem}",
                }
                for f in test_data_dir.glob("*.json")
            ],
            key=lambda x: x["file"],
        )
        return {"assessments": files}
    except Exception as e:
        logger.error(f"Error listing test assessments: {e}")
        return {"assessments": []}


@app.get("/api/test-assessment/{filename}", tags=["ui"])
async def get_test_assessment(filename: str):
    """GET /api/test-assessment/{filename} - Load a test assessment file."""
    # Validate filename to prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    test_data_dir = (
        Path(__file__).parent.parent.parent / "test" / "data" / "assessments"
    )
    file_path = test_data_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Test file not found: {filename}")

    if not str(file_path).startswith(str(test_data_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        return load_assessment(str(file_path))
    except Exception as e:
        logger.error(f"Error loading test assessment: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error loading assessment: {type(e).__name__}"
        )


# Suppress Chrome devtools warnings
@app.get("/.well-known/appspecific/{path:path}", tags=["ignore"])
async def well_known(path: str):
    """Suppress well-known metadata requests from browsers."""
    return {"status": "not-found"}


# Root - serve UI
@app.get("/", tags=["ops"])
async def root():
    """GET / - Serve the assessment UI home page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    # Fallback to API info if UI not found
    return {
        "name": app.title,
        "version": app.version,
        "endpoints": ["/assessment", "/assessment/source", "/health", "/ready", "/ui"],
        "persist_decisions": env_bool("AE_PERSIST_DECISIONS", True),
    }


# Launch example: uvicorn src.app.server:app --host 0.0.0.0 --port 8080
