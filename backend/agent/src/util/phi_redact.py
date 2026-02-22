from __future__ import annotations

"""PHI redaction utilities.

Core behavior:
 - Deterministic regex-based masking (currently UUID) using placeholder tokens [[<TYPE>_n]].
 - Mandatory PHI detection (PERSON, EMAIL, PHONE, etc.) via Presidio (no fallback).
 - Stable placeholders re-used for repeated values within a single call.
 - Company names in a static whitelist are never masked (even if analyzer tags them).
"""

import re
import string
import logging
from typing import Dict, Tuple, List, Optional

LOGGER = logging.getLogger("phi_redact")

# Company name whitelist (static; normalized at module import)
COMPANY_WHITELIST: set[str] = {
    "accredo",
    "cvs specialty",
    "walgreens",
    "optum",
    "express scripts",
    "humana specialty",
    "briova",
    "caremark",
    "biomatrix",
    "amber pharmacy",
    "senderra",
    "pavilion pharmacy",
    "panther rx",
    "vital care",
    "apria",
    "us bioservices",
    "onco360",
    "cura script",
    "diplomat pharmacy",
    "vanderbilt specialty pharmacy",
    "chartwell pharmacy",
    "fairview specialty pharmacy",
    "aurora pharmacy",
    "ascellahealth",
}

# Translation table for stripping punctuation once (shared normalization)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(token: str) -> str:
    """Lowercase + strip punctuation + trim."""
    return token.lower().translate(_PUNCT_TABLE).strip()


# Pre-normalize static company whitelist for fast membership tests
_NORMALIZED_COMPANY_WHITELIST = {_normalize(c) for c in COMPANY_WHITELIST}

PlaceholderMapping = Dict[str, str]

# Lazy singleton for Presidio analyzer
_ANALYZER: Optional[object] = None  # type: ignore

# Centralized deterministic regex patterns (extendable)
REGEX_PATTERNS: dict[str, re.Pattern] = {
    "UUID": re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
}

_ANALYZER_ENTITIES: List[str] = [
    "PERSON",  # comment if lots of false positives
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    # "US_SSN",
    # "US_PASSPORT",
    # "US_DRIVER_LICENSE",
    # "MEDICAL_LICENSE",
    # "IP_ADDRESS",
    # "URL",
]


def _get_analyzer() -> object:  # type: ignore[return-type]
    """Return a cached Presidio AnalyzerEngine instance or raise if unavailable.

    This enforces that PHI detection is always active; absence of Presidio is considered
    a misconfiguration rather than a runtime option.
    """
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    try:
        logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)
        from presidio_analyzer import AnalyzerEngine  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Presidio not available: {type(e).__name__}") from e
    try:
        _ANALYZER = AnalyzerEngine()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Presidio analyzer init failed: {type(e).__name__}") from e
    return _ANALYZER


def redact_phi(text: str) -> Tuple[str, PlaceholderMapping]:
    """Redact PHI from text, preserving whitelisted company names.

    Returns (redacted_text, placeholder_mapping).
    """
    value_to_placeholder: Dict[str, str] = {}
    mapping: PlaceholderMapping = {}
    counters: Dict[str, int] = {}

    whitelist = _NORMALIZED_COMPANY_WHITELIST

    def _placeholder(entity_type: str, value: str) -> str:
        if value in value_to_placeholder:
            return value_to_placeholder[value]
        counters[entity_type] = counters.get(entity_type, 0) + 1
        ph = f"[[{entity_type}_{counters[entity_type]}]]"
        value_to_placeholder[value] = ph
        mapping[ph] = value
        return ph

    redacted = text

    # Phase 1: deterministic regex masking
    for entity_type, pattern in REGEX_PATTERNS.items():
        for m in reversed(list(pattern.finditer(redacted))):
            val = m.group(0)
            if _normalize(val) in whitelist:
                continue
            ph = _placeholder(entity_type, val)
            redacted = redacted[: m.start()] + ph + redacted[m.end() :]

    # Phase 2: PHI entities via Presidio (mandatory)
    analyzer = _get_analyzer()
    results = analyzer.analyze(text=redacted, entities=_ANALYZER_ENTITIES, language="en")  # type: ignore[attr-defined]
    results.sort(key=lambda r: r.start, reverse=True)
    for r in results:
        val = redacted[r.start : r.end]
        if _normalize(val) in whitelist:
            continue
        if val.isupper() and len(val) > 2:  # heuristic skip for acronyms
            continue
        ph = _placeholder(r.entity_type, val)
        redacted = redacted[: r.start] + ph + redacted[r.end :]

    return redacted, mapping


def restore_text(redacted: str, mapping: PlaceholderMapping) -> str:
    def repl(m: re.Match) -> str:
        return mapping.get(m.group(0), m.group(0))

    return re.sub(r"\[\[[A-Z_]+_\d+\]\]", repl, redacted)


__all__ = ["redact_phi", "restore_text"]
