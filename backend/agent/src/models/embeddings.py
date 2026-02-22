import os
from typing import List, Optional, Any, Union
from threading import Lock
from ..util.phi_redact import redact_phi
from ..util.logger import get_logger

logger = get_logger("embeddings")

_EMB_CLIENT_SINGLETON: Optional[Any] = None
_EMB_LOCK = Lock()


class HuggingFaceEmbeddings:
    """HuggingFace sentence-transformers embedding client for local embeddings.

    Uses sentence-transformers library to run embeddings locally without
    requiring external API calls. Useful for development, testing, or
    environments without Azure access.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """Initialize HuggingFace embedding model.

        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2)
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto-detect)
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._device = device
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(
            f"Loaded HuggingFace model: {model_name} on device: {self._model.device}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        embedding = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.tolist()


class EmbeddingClient:
    """Embedding client supporting Azure OpenAI or HuggingFace providers.

    Provider selection via AE_EMBED_PROVIDER env var:
    - 'azure' (default): Uses Azure OpenAI embeddings
    - 'huggingface': Uses local sentence-transformers models

    HuggingFace configuration:
    - AE_HUGGINGFACE_EMBED_MODEL: Model name (default: all-MiniLM-L6-v2)
    - AE_HUGGINGFACE_DEVICE: Device to use (cpu/cuda/mps, default: auto-detect)
    - AE_EMBED_DIM: Optional dimension hint

    Azure configuration:
    - AE_EMBED_DEPLOYMENT, AE_AZURE_ENDPOINT, AE_AZURE_API_VERSION
    - Uses OAuth or AE_AZURE_API_KEY for authentication
    """

    def __init__(self):
        self._logger = get_logger("embeddings")
        self._client = get_embedding_client()
        dim_hint = os.getenv("AE_EMBED_DIM")
        self._dim: Optional[int] = int(dim_hint) if dim_hint else None
        self._dim_lock = Lock()

    @property
    def dim(self) -> Optional[int]:
        """Return the embedding vector dimension."""
        if self._dim is not None:
            return self._dim
        with self._dim_lock:
            if self._dim is not None:
                return self._dim
            try:
                probe_vec = self._client.embed_query("dimension probe")
                if probe_vec:
                    self._dim = len(probe_vec)
                    self._logger.debug(f"Detected embedding dimension: {self._dim}")
                else:
                    self._logger.warning(
                        "Probe embedding returned empty vector; dimension unknown"
                    )
            except Exception as e:
                self._logger.warning(f"Failed to auto-detect embedding dimension: {e}")
            return self._dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        _ = self.dim  # triggers detection if needed
        if os.getenv("AE_REDACT_PHI", "true").lower() == "true":
            redacted = []
            for t in texts:
                r, _ = redact_phi(t)
                redacted.append(r)
            texts = redacted
        return self._client.embed_documents(texts)


def _create_azure_client():
    """Create Azure OpenAI embeddings client."""
    from langchain_openai import AzureOpenAIEmbeddings
    from ..util.oauth import get_azure_api_token

    api_key = get_azure_api_token()

    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AE_EMBED_DEPLOYMENT"),
        api_version=os.getenv("AE_AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AE_AZURE_ENDPOINT"),
        openai_api_key=api_key,
    )


def _create_huggingface_client() -> HuggingFaceEmbeddings:
    """Create HuggingFace sentence-transformers embeddings client."""
    model_name = os.getenv("AE_HUGGINGFACE_EMBED_MODEL", "all-MiniLM-L6-v2")
    device = os.getenv("AE_HUGGINGFACE_DEVICE")

    logger.info(f"Creating HuggingFace embeddings client with model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name, device=device)


def get_embedding_client() -> Union[Any, HuggingFaceEmbeddings]:
    """Return a process-wide singleton embedding client.

    Provider selection via AE_EMBED_PROVIDER env var:
    - 'azure' (default): Azure OpenAI embeddings
    - 'huggingface': Local sentence-transformers embeddings
    """
    global _EMB_CLIENT_SINGLETON
    if _EMB_CLIENT_SINGLETON is not None:
        return _EMB_CLIENT_SINGLETON

    with _EMB_LOCK:
        if _EMB_CLIENT_SINGLETON is None:
            provider = os.getenv("AE_EMBED_PROVIDER", "azure").lower().strip()

            if provider == "huggingface":
                _EMB_CLIENT_SINGLETON = _create_huggingface_client()
            elif provider == "azure":
                _EMB_CLIENT_SINGLETON = _create_azure_client()
            else:
                logger.warning(
                    f"Unknown AE_EMBED_PROVIDER '{provider}', defaulting to azure"
                )
                _EMB_CLIENT_SINGLETON = _create_azure_client()

    return _EMB_CLIENT_SINGLETON


__all__ = ["EmbeddingClient", "HuggingFaceEmbeddings", "get_embedding_client"]
