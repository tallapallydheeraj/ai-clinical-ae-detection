""".env loader: loads first found .env from ENV_PATH, cwd, parent dirs, or agent root."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Core keys: OAuth credentials OR legacy API key, plus other required configs
_CORE_KEYS = [
    "AE_AZURE_OAUTH_CLIENT_ID",
    "AE_AZURE_API_KEY",
    "AE_LLM_PROVIDER",
    "AE_EMBED_DEPLOYMENT",
    "AE_MONGO_URI",
]


def _already_loaded() -> bool:
    return any(os.getenv(k) for k in _CORE_KEYS)


def load_env(force: bool = False) -> Optional[Path]:
    """Load .env from ENV_PATH, cwd, parent dirs, or agent root. Returns loaded path or None."""
    if _already_loaded() and not force:
        return None
    candidates = []
    override = os.getenv("ENV_PATH")
    if override:
        candidates.append(Path(override))
    cwd = Path.cwd()
    candidates.append(cwd / ".env")
    for parent in cwd.parents:
        candidates.append(parent / ".env")
    agent_env = Path(__file__).resolve().parents[2] / ".env"
    candidates.append(agent_env)
    for path in candidates:
        try:
            if path.exists():
                load_dotenv(path)
                return path
        except Exception:
            continue
    return None


__all__ = ["load_env"]
