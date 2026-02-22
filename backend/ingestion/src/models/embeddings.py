import os
from typing import List, Optional, Union, Any
from threading import Lock
from langchain_openai import AzureOpenAIEmbeddings
from ..util.logger import get_logger
from ..util.oauth import get_azure_api_token, is_azure_auth_configured

_EMB_CLIENT_SINGLETON: Optional[Any] = None
_EMB_LOCK = Lock()
_logger = get_logger("embeddings")


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
        _logger.info(
            f"Loaded HuggingFace model: {model_name} on device: {self._model.device}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors as lists of floats
        """
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Single text string to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return embedding.tolist()


class EmbeddingClient:
    """Embedding client supporting Azure OpenAI or HuggingFace providers.

    Provider selection via EMBEDDINGS_PROVIDER env var:
    - 'azure' (default): Uses Azure OpenAI embeddings with OAuth/API key auth
    - 'huggingface': Uses local sentence-transformers models

    HuggingFace configuration:
    - HUGGINGFACE_MODEL: Model name (default: all-MiniLM-L6-v2)
    - HUGGINGFACE_DEVICE: Device to use (cpu/cuda/mps, default: auto-detect)

    Azure configuration:
    - EMBEDDINGS_DEPLOYMENT, EMBEDDINGS_ENDPOINT, EMBEDDINGS_API_VERSION
    - Uses OAuth for AI COE authentication with fallback to EMBEDDINGS_API_KEY
    """

    def __init__(self):
        self._client = get_embedding_client()
        dim_hint = os.getenv("EMBEDDINGS_DIM")
        self.dim = int(dim_hint) if dim_hint else None

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._client.embed_documents(texts)
        if self.dim is None and embeddings:
            self.dim = len(embeddings[0])
            _logger.info(f"embedding_dim_inferred={self.dim}")
        return embeddings


def _create_azure_client() -> AzureOpenAIEmbeddings:
    """Create Azure OpenAI embeddings client."""
    deployment = os.getenv("EMBEDDINGS_DEPLOYMENT")
    endpoint = os.getenv("EMBEDDINGS_ENDPOINT")
    api_version = os.getenv("EMBEDDINGS_API_VERSION")

    # Use OAuth token or fallback to legacy API key
    if is_azure_auth_configured():
        api_key = get_azure_api_token()
        _logger.info("Using OAuth authentication for Azure embeddings")
    else:
        api_key = os.getenv("EMBEDDINGS_API_KEY")
        if api_key:
            _logger.info("Using legacy API key for Azure embeddings")

    if not all([deployment, endpoint, api_version, api_key]):
        missing = [
            n
            for n, v in [
                ("EMBEDDINGS_DEPLOYMENT", deployment),
                ("EMBEDDINGS_ENDPOINT", endpoint),
                ("EMBEDDINGS_API_VERSION", api_version),
                ("EMBEDDINGS_API_KEY or AE_AICOE_OAUTH_*", api_key),
            ]
            if not v
        ]
        raise RuntimeError(
            f"Missing required Azure embedding environment vars: {', '.join(missing)}"
        )

    return AzureOpenAIEmbeddings(
        azure_deployment=deployment,
        api_version=api_version,
        azure_endpoint=endpoint,
        openai_api_key=api_key,
    )


def _create_huggingface_client() -> HuggingFaceEmbeddings:
    """Create HuggingFace sentence-transformers embeddings client."""
    model_name = os.getenv("HUGGINGFACE_MODEL", "all-MiniLM-L6-v2")
    device = os.getenv("HUGGINGFACE_DEVICE")  # None = auto-detect

    _logger.info(f"Creating HuggingFace embeddings client with model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name, device=device)


def get_embedding_client() -> Union[AzureOpenAIEmbeddings, HuggingFaceEmbeddings]:
    """Return a process-wide singleton embedding client.

    Provider selection via EMBEDDINGS_PROVIDER env var:
    - 'azure' (default): Azure OpenAI embeddings
    - 'huggingface': Local sentence-transformers embeddings

    Safe for concurrent reads; creation guarded by a lock.
    """
    global _EMB_CLIENT_SINGLETON
    if _EMB_CLIENT_SINGLETON is not None:
        return _EMB_CLIENT_SINGLETON

    with _EMB_LOCK:
        if _EMB_CLIENT_SINGLETON is None:
            provider = os.getenv("EMBEDDINGS_PROVIDER", "azure").lower().strip()

            if provider == "huggingface":
                _EMB_CLIENT_SINGLETON = _create_huggingface_client()
            elif provider == "azure":
                _EMB_CLIENT_SINGLETON = _create_azure_client()
            else:
                _logger.warning(
                    f"Unknown EMBEDDINGS_PROVIDER '{provider}', defaulting to azure"
                )
                _EMB_CLIENT_SINGLETON = _create_azure_client()

    return _EMB_CLIENT_SINGLETON


__all__ = ["EmbeddingClient", "HuggingFaceEmbeddings", "get_embedding_client"]
