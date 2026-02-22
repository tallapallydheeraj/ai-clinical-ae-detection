"""Minimal text normalization utilities.

Currently limited to whitespace & dash normalization. Keep lean to avoid
semantic drift before embedding.
"""

from __future__ import annotations

import re

__all__ = ["clean_text", "normalize_structure"]


def clean_text(t: str) -> str:
    """Perform lightweight, lossless cleanup of raw contract text.

    Normalizations:
    - Replace em/en dashes with hyphen to simplify downstream tokenization.
    - Collapse all consecutive whitespace (including newlines) to a single space.
    - Strip leading/trailing whitespace.

    The function deliberately avoids aggressive unicode normalization or punctuation
    removal so embeddings preserve as much semantic signal as possible.
    """
    t = t.replace("\u2014", "-").replace("\u2013", "-")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_structure(text: str) -> str:
    # Reserved for future structural adjustments; currently identical to clean_text.
    return clean_text(text)
