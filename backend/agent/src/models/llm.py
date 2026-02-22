import os
from typing import Dict, Any, Tuple, Optional
from anyio import to_thread
from pydantic import BaseModel, Field
from ..util.phi_redact import redact_phi
from ..util.langsmith import safe_log_decision
from ..util.oauth import get_azure_api_token
from ..util.logger import get_logger

logger = get_logger("llm")


class Decision(BaseModel):
    """Structured decision output from the LLM."""

    is_adverse_event: bool = Field(
        ..., description="True if assessment meets adverse event definition"
    )
    matched_criteria: list[str] = Field(
        default_factory=list, description="Contract criteria matched"
    )
    explanation: str = Field(..., description="Brief explanation for decision")
    reasoning: str = Field(default="", description="Step-by-step reasoning chain")
    recommended_next_steps: list[str] = Field(
        default_factory=list, description="Follow up actions"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Subjective confidence 0..1")


def _maybe_redact(text: str) -> Tuple[str, Dict[str, str]]:
    """Redact PHI if enabled; return redacted text and mapping."""
    if os.getenv("AE_REDACT_PHI", "true").lower() == "true" and text:
        red, mapping = redact_phi(text)
        return red, mapping
    return text, {}


# --- Env helpers (unified) -------------------------------------------------
def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v and v.strip() else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v and v.strip() else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.lower() in ("1", "true", "yes", "on")


def _env_provider() -> str:
    return os.getenv("AE_LLM_PROVIDER", "azure").lower()


# --- Internal composition helpers -----------------------------------------
def _build_chain(provider: str, temperature: float, max_tokens: int, force_json: bool):
    """Return a langchain runnable that produces a Decision object."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from ..rag.prompts import DECISION_SYSTEM_PROMPT, DECISION_USER_PROMPT_TEMPLATE
    from langchain_core.runnables import RunnableSequence  # type: ignore

    parser = JsonOutputParser(pydantic_object=Decision)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DECISION_SYSTEM_PROMPT),
            ("user", DECISION_USER_PROMPT_TEMPLATE),
        ]
    )
    llm = get_llm(
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        force_json=force_json,
    )
    return prompt | llm | parser


def _log_llm_decision(
    decision_obj: Decision,
    provider: str,
    therapy: str,
    summary_len: int,
    contract_len: int,
    is_async: bool,
):
    safe_log_decision(
        inputs={
            "therapy": therapy,
            "summary_len": summary_len,
            "contract_len": contract_len,
            "provider": provider,
        },
        outputs=decision_obj.model_dump(),
        tags=["decision", "llm"],
        metadata={"heuristic": False, "async": is_async},
    )


def decide_with_llm(
    therapy: str, assessment_summary: str, contract_context: str
) -> Decision:
    """Synchronous decision via configured LLM."""
    provider = _env_provider()
    assessment_summary_red, _ = _maybe_redact(assessment_summary)
    contract_context_red, _ = _maybe_redact(contract_context)
    temperature = _env_float("AE_LLM_TEMPERATURE", 0.2)
    max_tokens = _env_int("AE_LLM_MAX_TOKENS", 800)
    force_json = _env_bool("AE_FORCE_JSON", True)
    chain = _build_chain(provider, temperature, max_tokens, force_json)
    out = chain.invoke(
        {
            "therapy": therapy,
            "summary": assessment_summary_red,
            "contract": contract_context_red,
        }
    )
    decision_obj = out if isinstance(out, Decision) else Decision.model_validate(out)
    _log_llm_decision(
        decision_obj,
        provider,
        therapy,
        len(assessment_summary_red or ""),
        len(contract_context_red or ""),
        False,
    )
    return decision_obj


async def decide_with_llm_async(
    therapy: str, assessment_summary: str, contract_context: str
) -> Decision:
    """Async decision via configured LLM (uses native async if available)."""
    provider = _env_provider()
    assessment_summary_red, _ = _maybe_redact(assessment_summary)
    contract_context_red, _ = _maybe_redact(contract_context)
    temperature = _env_float("AE_LLM_TEMPERATURE", 0.2)
    max_tokens = _env_int("AE_LLM_MAX_TOKENS", 800)
    force_json = _env_bool("AE_FORCE_JSON", True)
    chain = _build_chain(provider, temperature, max_tokens, force_json)
    if hasattr(chain, "ainvoke"):
        out = await chain.ainvoke(
            {
                "therapy": therapy,
                "summary": assessment_summary_red,
                "contract": contract_context_red,
            }
        )
    else:  # Fallback to thread for sync-only runtimes
        out = await to_thread.run_sync(
            chain.invoke,
            {
                "therapy": therapy,
                "summary": assessment_summary_red,
                "contract": contract_context_red,
            },
        )
    decision_obj = out if isinstance(out, Decision) else Decision.model_validate(out)
    _log_llm_decision(
        decision_obj,
        provider,
        therapy,
        len(assessment_summary_red or ""),
        len(contract_context_red or ""),
        True,
    )
    return decision_obj


_LLM_CACHE: Dict[str, Any] = {}


def _llm_cache_key(
    provider: str, temperature: float, max_tokens: int, force_json: bool
) -> str:
    """Stable cache key (include temperature even if provider may ignore it)."""
    return f"{provider}|{temperature}|{max_tokens}|json={force_json}"


def get_llm(provider: str, temperature: float, max_tokens: int, force_json: bool):
    """Return cached LLM client. Creates on first request per unique configuration.

    Supported providers:
    - 'azure': Azure OpenAI (requires AE_AZURE_* env vars)
    - 'huggingface': HuggingFace models via transformers pipeline
    - 'llama' / 'ollama': Local Ollama server (requires Ollama running)
    """
    key = _llm_cache_key(provider, temperature, max_tokens, force_json)
    existing = _LLM_CACHE.get(key)
    if existing is not None:
        return existing

    if provider == "azure":
        from langchain_openai import AzureChatOpenAI

        model_kwargs: Optional[Dict[str, Any]] = None
        if force_json:
            model_kwargs = {"response_format": {"type": "json_object"}}
        api_key = get_azure_api_token()
        client = AzureChatOpenAI(
            azure_deployment=os.getenv("AE_AZURE_CHAT_DEPLOYMENT"),
            api_version=os.getenv("AE_AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AE_AZURE_ENDPOINT"),
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )
    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        except ImportError:
            try:
                from langchain_community.llms import HuggingFacePipeline

                ChatHuggingFace = None
            except ImportError:
                raise RuntimeError(
                    "HuggingFace LLM requested but langchain-huggingface not installed. "
                    "Run: pip install langchain-huggingface transformers torch"
                )

        model_id = os.getenv(
            "AE_HUGGINGFACE_LLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct"
        )
        device = os.getenv("AE_HUGGINGFACE_DEVICE", "auto")

        # Check for quantization settings
        load_in_4bit = os.getenv("AE_HUGGINGFACE_LOAD_4BIT", "false").lower() == "true"
        load_in_8bit = os.getenv("AE_HUGGINGFACE_LOAD_8BIT", "false").lower() == "true"

        logger.info(
            f"Loading HuggingFace LLM: {model_id} (device={device}, 4bit={load_in_4bit}, 8bit={load_in_8bit})"
        )

        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                pipeline,
                BitsAndBytesConfig,
            )
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Configure quantization if requested
            quantization_config = None
            if load_in_4bit or load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                quantization_config=quantization_config,
                torch_dtype=(
                    torch.float16 if not (load_in_4bit or load_in_8bit) else None
                ),
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                return_full_text=False,
            )

            hf_pipeline = HuggingFacePipeline(pipeline=pipe)

            # Use ChatHuggingFace wrapper if available for better chat support
            if ChatHuggingFace is not None:
                client = ChatHuggingFace(llm=hf_pipeline)
            else:
                client = hf_pipeline

        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise RuntimeError(f"Failed to load HuggingFace model '{model_id}': {e}")

    elif provider == "groq":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("AE_GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq provider requires AE_GROQ_API_KEY environment variable"
            )

        model_name = os.getenv("AE_GROQ_MODEL", "mixtral-8x7b-32768")

        model_kwargs: Optional[Dict[str, Any]] = None
        if force_json:
            model_kwargs = {"response_format": {"type": "json_object"}}

        client = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )
        logger.info(f"Using Groq LLM: {model_name}")

    elif provider in ("llama", "ollama"):
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError as e:
            raise RuntimeError(
                "LLM provider 'llama/ollama' requested but ChatOllama not available"
            ) from e

        model_name = os.getenv("AE_LLAMA_MODEL", "llama3.2")
        base_url = os.getenv("AE_OLLAMA_BASE_URL", "http://localhost:11434")

        client = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
        )
        logger.info(f"Using Ollama LLM: {model_name} at {base_url}")
    else:
        raise ValueError(
            f"Unsupported AE_LLM_PROVIDER='{provider}' (supported: azure, huggingface, groq, llama, ollama)"
        )

    _LLM_CACHE[key] = client
    return client


__all__ = ["Decision", "decide_with_llm", "decide_with_llm_async", "get_llm"]
