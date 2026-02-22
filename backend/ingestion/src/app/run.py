"""Contract ingestion runner.

dev env uses local fixture files; qa/prod load from S3 bucket.
Chunking strategy: section (default) or semantic.
"""

import os
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from ..util.text_normalize import clean_text
from ..util.chunking import chunk_text
from ..models.embeddings import EmbeddingClient
from ..db.database import get_database_backend
from ..util.logger import get_logger

logger = get_logger("ingest_contracts")


def _load_local_contracts() -> List[Tuple[str, Path]]:
    base = Path(__file__).resolve().parents[2]
    data_dir = base / "test" / "data" / "contracts"
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        logger.warning(f"No local contract .txt files found under {data_dir}")
    else:
        logger.info(f"Found {len(files)} local contract files under {data_dir}")
    return [(f.stem.upper(), f) for f in files]


def _load_s3_contracts() -> List[Tuple[str, str]]:
    bucket = os.getenv("S3_CONTRACT_BUCKET")
    prefix = os.getenv("S3_CONTRACT_PREFIX", "")
    if not bucket:
        logger.error("S3_CONTRACT_BUCKET not set; cannot load contracts from S3.")
        return []
    try:
        import boto3  # type: ignore

        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        texts: List[Tuple[str, str]] = []
        total = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.lower().endswith(".txt"):
                    continue
                therapy = Path(key).stem.upper()
                body = (
                    s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
                )
                texts.append((therapy, body))
                total += 1
        logger.info(
            f"Loaded {total} contract texts from S3 bucket={bucket} prefix='{prefix}'"
        )
        return texts
    except Exception as e:
        logger.exception(f"Failed to load contracts from S3: {e}")
        return []


def main():
    load_dotenv()
    env = os.getenv("ENVIRONMENT", "dev").lower()
    strategy_raw = os.getenv("CHUNK_STRATEGY", "section")
    strategy = "semantic" if strategy_raw.lower() == "semantic" else "section"

    # Unified default max chars for all strategies.
    max_chars_default = 2000
    max_chars = int(os.getenv("CHUNK_MAX_CHARS", str(max_chars_default)))
    semantic_similarity_threshold = float(os.getenv("SEMANTIC_MIN_SIMILARITY", "0.40"))
    semantic_max_sentences = int(os.getenv("SEMANTIC_MAX_SENTENCES", "25"))
    semantic_similarity_window = int(os.getenv("SEMANTIC_SIMILARITY_WINDOW", "3"))

    logger.info(
        f"env={env} strategy={strategy} max_chars={max_chars} "
        f"semantic_threshold={semantic_similarity_threshold if strategy=='semantic' else '-'}"
    )

    if env == "dev":
        contract_refs = _load_local_contracts()
        texts: List[Tuple[str, str]] = []
        for therapy, path in contract_refs:
            try:
                texts.append((therapy, path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.error(f"Failed reading {path}: {e}")
    else:
        texts = _load_s3_contracts()

    if not texts:
        logger.warning("No contract texts to ingest.")
        return

    ec = EmbeddingClient()
    all_docs = []
    for therapy, raw in texts:
        contract_id = therapy
        raw_clean = clean_text(raw)
        if strategy == "semantic":
            chunks = chunk_text(
                raw_clean,
                strategy=strategy,
                max_chars=max_chars,
                similarity_threshold=semantic_similarity_threshold,
                max_sentences=semantic_max_sentences,
                similarity_window=semantic_similarity_window,
            )
        else:
            chunks = chunk_text(raw_clean, strategy=strategy, max_chars=max_chars)
        embeddings = ec.embed(chunks)
        logger.info(
            f"contract={therapy} chunks={len(chunks)} embedding_dim={getattr(ec, 'dim', '?')}"
        )
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            all_docs.append(
                {
                    "contract_id": contract_id,
                    "therapy": therapy,
                    "chunk_id": i,
                    "text": text,
                    "embedding": emb,
                }
            )
    try:
        db = get_database_backend()
    except Exception as e:
        logger.error(f"Database backend unavailable: {e}")
        return
    db.upsert_contract_chunks(all_docs)
    logger.info("Ingestion complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
