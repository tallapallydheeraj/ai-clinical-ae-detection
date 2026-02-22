"""Chunking utilities (section heuristic or semantic sentence grouping).

Production-focused: minimal comments, single env reads, defensive fallbacks.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from .logger import get_logger

_log = get_logger("chunking")

from .text_normalize import clean_text

__all__ = ["chunk_text_sections", "chunk_text_semantic", "chunk_text"]


def _slice_long(text: str, max_chars: int) -> List[str]:
    """Word-safe slice: prefer last whitespace <= max_chars else hard cut."""
    parts: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        remaining = n - i
        if remaining <= max_chars:
            seg = text[i:]
            if seg.strip():
                parts.append(seg.strip())
            break
        window = text[i : i + max_chars]
        # Find last whitespace in window to split; avoid leading/trailing whitespace artifacts.
        cut = window.rfind(" ")
        if (
            cut < max_chars * 0.5
        ):  # Heuristic: if whitespace too early, treat as no good boundary.
            cut = -1
        if cut == -1:
            # fallback hard boundary
            seg = window
            i += max_chars
        else:
            seg = window[:cut]
            i += cut + 1  # skip the space we split on
        if seg.strip():
            parts.append(seg.strip())
    return parts


def _slice_long_tokens(
    text: str, max_tokens: int, encoder, overlap_tokens: int = 0
) -> List[str]:
    """Token-safe slice with optional trailing overlap (overlap_tokens < max_tokens)."""
    if encoder is None:
        return _slice_long(text, max_tokens * 4)  # heuristic char fallback
    tokens = encoder.encode(text)
    parts: List[str] = []
    n = len(tokens)
    if n <= max_tokens:
        decoded = encoder.decode(tokens).strip()
        return [decoded] if decoded else []
    step = (
        max_tokens
        if overlap_tokens <= 0 or overlap_tokens >= max_tokens
        else (max_tokens - overlap_tokens)
    )
    if overlap_tokens > 0 and overlap_tokens < max_tokens:
        _log.debug(
            f"token_overlap applied overlap_tokens={overlap_tokens} max_tokens={max_tokens}"
        )
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        block = tokens[start:end]
        decoded = encoder.decode(block).strip()
        if decoded:
            parts.append(decoded)
        if end == n:
            break
        start += step
    return parts


def _get_token_encoder(
    model_name: Optional[str] = None,
):  # model_name kept for future extension
    """Return tiktoken encoder or None (silent failure)."""
    try:  # pragma: no cover - optional path
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        _log.info("token_encoder=tiktoken cl100k_base active")
        return enc
    except Exception:
        _log.info("token_encoder=disabled (tiktoken import failed)")
        return None


def _count_tokens(text: str, encoder) -> int:
    if encoder is None:
        return 0
    try:  # pragma: no cover
        return len(encoder.encode(text))
    except Exception:
        return 0


def _get_overlap_tokens() -> int:
    """Single env read for overlap configuration."""
    if os.getenv("CHUNK_ENABLE_OVERLAP", "false").lower() != "true":
        return 0
    raw = os.getenv("CHUNK_OVERLAP_TOKENS", "0")
    return int(raw) if raw.isdigit() else 0


def _slice_long_chars_overlap(
    text: str, max_chars: int, overlap_tokens: int
) -> List[str]:
    """Char-based slicing with heuristic token overlap (tokens≈4 chars).

    If overlap_tokens == 0 -> fallback to word-safe slicing.
    Uses step = max_chars - overlap_chars (overlap_chars = overlap_tokens * 4), guarded >0.
    Word boundary preference retained per window.
    """
    if overlap_tokens <= 0:
        return _slice_long(text, max_chars)
    overlap_chars = overlap_tokens * 4
    step = max_chars - overlap_chars
    if step <= 0:
        return _slice_long(text, max_chars)  # safety
    parts: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_chars, n)
        window = text[i:end]
        if len(window) == max_chars and end < n:
            cut = window.rfind(" ")
            if cut > max_chars * 0.5:  # prefer later whitespace
                window = window[:cut]
                end = i + cut
        seg = window.strip()
        if seg:
            parts.append(seg)
        if end == n:
            break
        i += step
    if len(parts) > 1:
        _log.debug(
            f"oversize_char_overlap parts={len(parts)} first_len={len(parts[0])} overlap_tokens={overlap_tokens}"
        )
    return parts


def chunk_text_sections(
    text: str,
    max_chars: int = 2000,
    max_tokens: Optional[int] = None,
    token_encoder=None,
) -> List[str]:
    """Heuristic section chunking (headers + bullets) with size packing."""
    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []
    header_pattern = re.compile(r"^(?:SECTION\s+\d+|[A-Z][A-Z0-9 \-/]{3,}$|\d+\.\s+.+)")

    def is_bullet(line: str) -> bool:
        ls = line.lstrip()
        return bool(re.match(r"^(?:[-*]|\d+\.|[A-Za-z]\))\s+", ls))

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_header = False
        if header_pattern.match(stripped):
            is_header = True
        elif is_bullet(line) and (i == 0 or not lines[i - 1].strip()):
            is_header = True
        if is_header:
            if current:
                sections.append("\n".join(current).strip())
                current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks: List[str] = []
    buf: List[str] = []
    buf_len_chars = 0
    buf_len_tokens = 0
    use_tokens = max_tokens is not None and token_encoder is not None
    overlap_tokens = _get_overlap_tokens() if use_tokens else 0
    for sec in sections:
        sec_clean = clean_text(sec)
        # oversize single section handling (char-based slicing remains for simplicity)
        if (
            use_tokens
            and max_tokens
            and _count_tokens(sec_clean, token_encoder) > max_tokens
        ):
            token_parts = _slice_long_tokens(
                sec_clean, max_tokens, token_encoder, overlap_tokens=overlap_tokens
            )
            if len(token_parts) > 1:
                _log.debug(
                    f"oversize_token_split_section parts={len(token_parts)} first_tokens={_count_tokens(token_parts[0], token_encoder)} total_tokens={_count_tokens(sec_clean, token_encoder)}"
                )
            chunks.extend(token_parts)
            continue
        if len(sec_clean) > max_chars and not use_tokens:
            char_parts = _slice_long_chars_overlap(sec_clean, max_chars, overlap_tokens)
            chunks.extend(char_parts)
            continue
        sec_tokens = _count_tokens(sec_clean, token_encoder) if use_tokens else 0
        projected_chars = buf_len_chars + len(sec_clean) + 1
        projected_tokens = buf_len_tokens + sec_tokens if use_tokens else 0
        limit_exceeded = (use_tokens and projected_tokens > (max_tokens or 0)) or (
            not use_tokens and projected_chars > max_chars
        )
        if limit_exceeded and buf:
            chunks.append("\n".join(buf).strip())
            buf = [sec_clean]
            buf_len_chars = len(sec_clean)
            buf_len_tokens = sec_tokens
        else:
            buf.append(sec_clean)
            buf_len_chars = projected_chars
            if use_tokens:
                buf_len_tokens = projected_tokens
    if buf:
        chunks.append("\n".join(buf).strip())
    return chunks


def _split_sentences(text: str) -> List[str]:
    """Sentence split via spaCy (if available) else regex fallback."""
    try:  # pragma: no cover - optional path
        import spacy  # type: ignore

        # Prefer existing small English model; if unavailable create blank with sentencizer.
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if sents:
            _log.info(f"sentence_splitter=spacy count={len(sents)}")
            return sents
    except Exception:
        _log.info("sentence_splitter=regex_fallback")
        pass
    # Naive fallback (maintenance-light; accepts abbreviation split imperfections).
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_text_semantic(
    text: str,
    model_name: Optional[str] = None,
    max_chars: int = 2000,
    max_sentences: int = 25,
    similarity_threshold: float = 0.40,
    similarity_window: int = 3,
    max_tokens: Optional[int] = None,
    token_encoder=None,
) -> List[str]:
    """Semantic sentence grouping (cosine similarity + size limits)."""
    text = clean_text(text)
    sentences = _split_sentences(text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    _log.info(
        f"semantic_sentences={len(sentences)} max_sentences={max_sentences} threshold_init={similarity_threshold}"
    )
    # Embed all sentences using Azure embeddings in a single batch.
    try:
        from ..models.embeddings import EmbeddingClient  # local azure-only client
        import numpy as np  # type: ignore

        emb_client = EmbeddingClient()
        raw_embs = emb_client.embed(sentences)
        sent_embeddings = np.asarray(raw_embs, dtype=float)
        # Normalize for cosine similarity consistency.
        norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sent_embeddings = sent_embeddings / norms
    except Exception:
        # Fallback: heuristic section chunking if embedding fails entirely.
        return chunk_text_sections(text, max_chars=max_chars)

    # Optional adaptive threshold
    if os.getenv("SEMANTIC_ADAPTIVE_THRESHOLD", "false").lower() == "true":
        try:
            mean_emb = np.mean(sent_embeddings, axis=0)
            sims = sent_embeddings @ mean_emb
            var = float(np.var(sims))
            target_var = float(os.getenv("SEMANTIC_TARGET_VARIANCE", "0.04"))
            scale_min = float(os.getenv("SEMANTIC_THRESHOLD_SCALE_MIN", "0.7"))
            scale_max = float(os.getenv("SEMANTIC_THRESHOLD_SCALE_MAX", "1.3"))
            scale = 1 + (target_var - var)
            scale = max(scale_min, min(scale_max, scale))
            similarity_threshold = round(similarity_threshold * scale, 4)
            _log.info(
                f"semantic_threshold_adapted var={var:.4f} threshold_new={similarity_threshold}"
            )
        except Exception:
            pass

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_embs: List = []
    current_chars = 0
    current_tokens = 0
    use_tokens = max_tokens is not None and token_encoder is not None

    overlap_tokens = _get_overlap_tokens() if (max_tokens and token_encoder) else 0

    def flush():
        nonlocal current_sentences, current_embs, current_chars, current_tokens
        if not current_sentences:
            return
        merged = " ".join(current_sentences).strip()
        if (
            use_tokens
            and max_tokens
            and _count_tokens(merged, token_encoder) > max_tokens
        ):
            parts = _slice_long_tokens(
                merged, max_tokens, token_encoder, overlap_tokens=overlap_tokens
            )
            if len(parts) > 1:
                _log.debug(
                    f"oversize_token_split_semantic parts={len(parts)} first_tokens={_count_tokens(parts[0], token_encoder)} total_tokens={_count_tokens(merged, token_encoder)}"
                )
            chunks.extend(parts)
        elif len(merged) > max_chars:
            parts = _slice_long_chars_overlap(merged, max_chars, overlap_tokens)
            if len(parts) > 1:
                _log.debug(
                    f"oversize_sentence_split parts={len(parts)} first_len={len(parts[0])} total_len={len(merged)}"
                )
            chunks.extend(parts)
        else:
            chunks.append(merged)
        current_sentences.clear()
        current_embs.clear()
        current_chars = 0
        current_tokens = 0

    for sent, emb in zip(sentences, sent_embeddings):
        sent_len = len(sent)
        sent_tokens = _count_tokens(sent, token_encoder) if use_tokens else 0
        if not current_sentences:
            current_sentences.append(sent)
            current_embs.append(emb)
            current_chars += sent_len + 1
            if use_tokens:
                current_tokens += sent_tokens
            continue
        window_embs = current_embs[-similarity_window:]
        mean_vec = sum(window_embs) / len(window_embs)
        import numpy as np  # type: ignore

        sim = float(np.dot(mean_vec, emb))
        projected_chars = current_chars + sent_len + 1
        projected_tokens = current_tokens + sent_tokens if use_tokens else 0
        size_exceeded = (use_tokens and projected_tokens > (max_tokens or 0)) or (
            not use_tokens and projected_chars > max_chars
        )
        len_exceeded = len(current_sentences) >= max_sentences
        similarity_exceeded = sim < similarity_threshold
        if size_exceeded or len_exceeded or similarity_exceeded:
            reason = (
                "size" if size_exceeded else ("len" if len_exceeded else "similarity")
            )
            _log.debug(
                f"semantic_split reason={reason} current_group_size={len(current_sentences)} sim={sim:.4f}"
            )
            flush()
        current_sentences.append(sent)
        current_embs.append(emb)
        current_chars += sent_len + 1
        if use_tokens:
            current_tokens += sent_tokens
    flush()
    return chunks


def chunk_text(text: str, strategy: str = "section", **kwargs) -> List[str]:
    """Entry point selecting strategy ('section' or 'semantic')."""
    s = strategy.lower()
    # Optional token limit via env or kwargs
    max_tokens = kwargs.get("max_tokens")
    if max_tokens is None:
        env_tokens = os.getenv("CHUNK_MAX_TOKENS", "512")
        if env_tokens and env_tokens.isdigit():
            max_tokens = int(env_tokens)
    token_encoder = None
    if max_tokens:
        token_encoder = _get_token_encoder(kwargs.get("model_name"))
        if token_encoder is None:
            # If tokenizer unavailable, we silently fallback to char-only limits.
            max_tokens = None
    if s == "semantic":
        return chunk_text_semantic(
            text,
            model_name=None,
            max_chars=kwargs.get("max_chars", 2000),
            max_sentences=kwargs.get("max_sentences", 25),
            similarity_threshold=kwargs.get("similarity_threshold", 0.40),
            similarity_window=kwargs.get("similarity_window", 3),
            max_tokens=max_tokens,
            token_encoder=token_encoder,
        )
    _log.info("strategy=section")
    return chunk_text_sections(
        text,
        max_chars=kwargs.get("max_chars", 2000),
        max_tokens=max_tokens,
        token_encoder=token_encoder,
    )
