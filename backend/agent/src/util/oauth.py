"""Generic OAuth 2.0 token manager using client credentials flow.

Simple, reusable OAuth 2.0 implementation using the `requests` library.
Supports multiple named clients for different APIs.

Usage:
    # Register a client from environment variables
    register_oauth_client_from_env("azure_openai", env_prefix="AE_AZURE_OAUTH")

    # Get tokens
    token = get_oauth_token("azure_openai")
"""

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict

import requests

from .logger import get_logger

_logger = get_logger("oauth")


@dataclass
class TokenInfo:
    """Cached OAuth token with metadata."""

    access_token: str
    expires_at: float
    token_type: str = "Bearer"
    scope: Optional[str] = None


class OAuthClient:
    """Thread-safe OAuth 2.0 client credentials token manager."""

    def __init__(
        self,
        name: str,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: Optional[str] = None,
        buffer_seconds: int = 300,
    ):
        self.name = name
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._buffer_seconds = buffer_seconds
        self._token: Optional[TokenInfo] = None
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return all([self._token_url, self._client_id, self._client_secret])

    def _is_token_valid(self) -> bool:
        if not self._token:
            return False
        return time.time() < (self._token.expires_at - self._buffer_seconds)

    def _fetch_token(self) -> TokenInfo:
        """Fetch token using requests library."""
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scope:
            data["scope"] = self._scope

        _logger.debug(f"[{self.name}] Fetching OAuth token")

        try:
            # Use SSL cert if configured
            verify = os.getenv("SSL_CERT_FILE", True)

            response = requests.post(
                self._token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
                verify=verify,
            )
            response.raise_for_status()
            result = response.json()

        except requests.RequestException as e:
            _logger.error(f"[{self.name}] OAuth request failed: {e}")
            raise RuntimeError(
                f"OAuth token request failed for '{self.name}': {e}"
            ) from e

        access_token = result.get("access_token")
        if not access_token:
            raise RuntimeError(f"OAuth response missing access_token for '{self.name}'")

        expires_in = result.get("expires_in", 3600)
        token = TokenInfo(
            access_token=access_token,
            expires_at=time.time() + expires_in,
            token_type=result.get("token_type", "Bearer"),
            scope=result.get("scope"),
        )

        _logger.info(f"[{self.name}] OAuth token obtained, expires in {expires_in}s")
        return token

    def get_token(self) -> str:
        """Get a valid access token, refreshing if needed (thread-safe)."""
        if self._is_token_valid():
            return self._token.access_token

        with self._lock:
            if self._is_token_valid():
                return self._token.access_token
            self._token = self._fetch_token()
            return self._token.access_token

    def invalidate(self) -> None:
        """Force token refresh on next get_token() call."""
        with self._lock:
            self._token = None


# =============================================================================
# Client Registry
# =============================================================================

_REGISTRY: Dict[str, OAuthClient] = {}
_REGISTRY_LOCK = threading.Lock()


def register_oauth_client(
    name: str,
    token_url: str,
    client_id: str,
    client_secret: str,
    scope: Optional[str] = None,
    buffer_seconds: int = 300,
) -> OAuthClient:
    """Register an OAuth client."""
    client = OAuthClient(
        name=name,
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
        buffer_seconds=buffer_seconds,
    )
    with _REGISTRY_LOCK:
        _REGISTRY[name] = client
    _logger.info(f"Registered OAuth client: {name}")
    return client


def register_oauth_client_from_env(name: str, env_prefix: str) -> Optional[OAuthClient]:
    """Register an OAuth client from environment variables.

    Looks for: {PREFIX}_TOKEN_URL, {PREFIX}_CLIENT_ID, {PREFIX}_CLIENT_SECRET,
    {PREFIX}_SCOPE, {PREFIX}_TOKEN_BUFFER_SECONDS
    """
    token_url = os.getenv(f"{env_prefix}_TOKEN_URL", "")
    client_id = os.getenv(f"{env_prefix}_CLIENT_ID", "")
    client_secret = os.getenv(f"{env_prefix}_CLIENT_SECRET", "")

    if not all([token_url, client_id, client_secret]):
        return None

    return register_oauth_client(
        name=name,
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        scope=os.getenv(f"{env_prefix}_SCOPE"),
        buffer_seconds=int(os.getenv(f"{env_prefix}_TOKEN_BUFFER_SECONDS", "300")),
    )


def get_oauth_client(name: str) -> Optional[OAuthClient]:
    """Get a registered OAuth client."""
    return _REGISTRY.get(name)


def get_oauth_token(name: str) -> str:
    """Get an access token from a registered client."""
    client = _REGISTRY.get(name)
    if not client:
        raise ValueError(f"OAuth client '{name}' not registered")
    return client.get_token()


def is_oauth_client_configured(name: str) -> bool:
    """Check if an OAuth client is registered and configured."""
    client = _REGISTRY.get(name)
    return client is not None and client.is_configured


# =============================================================================
# Azure OpenAI Helpers (convenience functions)
# =============================================================================

_AZURE_CLIENT_NAME = "azure_openai"
_AZURE_INIT_LOCK = threading.Lock()
_AZURE_INITIALIZED = False


def _ensure_azure_initialized() -> None:
    """Lazy-initialize Azure OAuth client from environment."""
    global _AZURE_INITIALIZED
    if _AZURE_INITIALIZED:
        return
    with _AZURE_INIT_LOCK:
        if not _AZURE_INITIALIZED:
            register_oauth_client_from_env(_AZURE_CLIENT_NAME, "AE_AZURE_OAUTH")
            _AZURE_INITIALIZED = True


def get_azure_api_token() -> str:
    """Get API token for Azure OpenAI (OAuth or legacy API key fallback)."""
    _ensure_azure_initialized()

    if is_oauth_client_configured(_AZURE_CLIENT_NAME):
        return get_oauth_token(_AZURE_CLIENT_NAME)

    # Fallback to legacy API key
    legacy_key = os.getenv("AE_AZURE_API_KEY")
    if legacy_key:
        return legacy_key

    raise ValueError(
        "Azure API credentials not configured. Set AE_AZURE_OAUTH_* env vars or AE_AZURE_API_KEY"
    )


def is_azure_auth_configured() -> bool:
    """Check if Azure auth is configured (OAuth or legacy key)."""
    _ensure_azure_initialized()
    return is_oauth_client_configured(_AZURE_CLIENT_NAME) or bool(
        os.getenv("AE_AZURE_API_KEY")
    )


__all__ = [
    # Generic OAuth
    "OAuthClient",
    "register_oauth_client",
    "register_oauth_client_from_env",
    "get_oauth_client",
    "get_oauth_token",
    "is_oauth_client_configured",
    # Azure OpenAI helpers
    "get_azure_api_token",
    "is_azure_auth_configured",
]
