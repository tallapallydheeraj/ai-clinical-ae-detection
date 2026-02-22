from typing import List, Dict, Any
import os
import time
from ..util.logger import get_logger

logger = get_logger("mongo")


def have_mongo() -> bool:
    return bool(os.getenv("MONGODB_ATLAS_URI", "").strip())


def _mongo_client():
    """Create a MongoClient.

    Notes:
        If you recently updated the connection string to a single host that is no longer primary,
        writes will raise NotPrimaryError. Prefer a full replica set / Atlas SRV URI.
    Env overrides:
        MONGODB_SERVER_SELECTION_MS (int, default 30000)
    """
    from pymongo import MongoClient

    uri = os.getenv("MONGODB_ATLAS_URI")
    server_sel_ms = int(os.getenv("MONGODB_SERVER_SELECTION_MS", "30000"))
    # Provide appName for observability; rely on driver topology management.
    app_name = os.getenv("PROJECT_NAME", "clinical-adverse-event-ingestion")
    return MongoClient(
        uri,
        appname=app_name,
        serverSelectionTimeoutMS=server_sel_ms,
        retryWrites=True,
    )


def _ensure_collection_indexes(client, db_name: str, coll_name: str):
    """Ensure basic scalar indexes exist (id is implicit)."""
    try:
        coll = client[db_name][coll_name]
        # Therapy filter index (single field). If already exists, PyMongo ignores.
        coll.create_index("therapy")
        # contract_id index (optional for filtering outside vector search).
        coll.create_index("contract_id")
    except Exception as e:
        logger.debug(f"Skipping scalar index creation: {e}")


def _ensure_vector_search_index(
    client, db_name: str, coll_name: str, index_name: str, embedding_dim: int
):
    """Ensure Atlas Search vector index exists; create if missing.

    Uses createSearchIndexes command (MongoDB Atlas / server with search capability).
    If listing fails (no search enabled), function logs and returns silently.
    """
    if embedding_dim <= 0:
        logger.warning("Cannot create vector index: unknown embedding dimension.")
        return
    try:
        coll = client[db_name][coll_name]
        # List existing search indexes
        existing = []
        try:
            existing = [
                d.get("name") for d in coll.aggregate([{"$listSearchIndexes": {}}])
            ]
            logger.debug(f"Found existing search indexes: {existing}")
        except Exception as list_err:
            logger.debug(f"List search indexes not available: {list_err}")
            return
        if index_name in existing:
            logger.debug(
                f"Search index '{index_name}' already exists, skipping creation."
            )
            return  # already present
        logger.info(
            f"Creating search vector index '{index_name}' (dim={embedding_dim})."
        )
        definition = {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": embedding_dim,
                        "similarity": "cosine",
                    },
                    "contract_id": {"type": "token"},
                    "therapy": {"type": "token"},
                },
            }
        }
        client[db_name].command(
            {
                "createSearchIndexes": coll_name,
                "indexes": [
                    {
                        "name": index_name,
                        "definition": definition,
                    }
                ],
            }
        )
        logger.info(f"Search vector index '{index_name}' creation requested.")
    except Exception as e:
        logger.warning(f"Vector index creation skipped/failure for '{index_name}': {e}")


def upsert_contract_chunks(docs: List[Dict[str, Any]]):
    """Upsert contract chunk docs into MongoDB.

    Retries transient topology errors (primary stepdown, reconnect) and fails fast
    after configured attempts. Operates directly against MongoDB (no local filesystem writes).

    Env overrides:
        MONGODB_MAX_RETRIES (int, default 3)
        MONGODB_RETRY_SLEEP_MS (int, default 500)
    """
    if not have_mongo():
        raise RuntimeError("MongoDB unavailable: MONGODB_ATLAS_URI not configured.")

    from pymongo import UpdateOne
    from pymongo.errors import (
        NotPrimaryError,
        AutoReconnect,
        PyMongoError,
        OperationFailure,
    )
    from pymongo.write_concern import WriteConcern

    db_name = os.getenv("MONGODB_DB_NAME", "pharmacovigilance")
    coll_name = os.getenv("MONGODB_CONTRACTS_COLLECTION", "contracts")
    max_retries = int(os.getenv("MONGODB_MAX_RETRIES", "3"))
    sleep_ms = int(os.getenv("MONGODB_RETRY_SLEEP_MS", "500"))

    if not docs:
        logger.info("No documents supplied (empty list).")
        return

    # Optional purge of existing chunks for these contract_ids before upserting new set.
    purge = os.getenv("INGEST_PURGE_BEFORE", "false").lower() == "true"
    contract_ids = sorted({d.get("contract_id") for d in docs if d.get("contract_id")})

    attempt = 0
    last_err: Exception | None = None

    # Reuse a single client so driver can perform topology refresh between retries.
    client = _mongo_client()
    coll = client[db_name].get_collection(
        coll_name,
        write_concern=WriteConcern(w=os.getenv("MONGODB_WRITE_CONCERN", "majority")),
    )

    # Purge existing documents if requested (remove stale chunks when grouping changes).
    if purge and contract_ids:
        try:
            deleted = 0
            for cid in contract_ids:
                res = coll.delete_many({"contract_id": cid})
                deleted += res.deleted_count
            logger.info(
                f"purge_before=true contracts={len(contract_ids)} deleted_chunks={deleted}"
            )
        except Exception as e:
            logger.warning(f"Purge before upsert failed: {e}")

    # Build bulk ops after optional purge.
    ops: List[UpdateOne] = []
    for d in docs:
        _id = f"{d['contract_id']}::{d['chunk_id']}"
        ops.append(UpdateOne({"_id": _id}, {"$set": {**d, "_id": _id}}, upsert=True))
    if not ops:
        logger.info("No documents to upsert (empty ops list after purge).")
        try:
            client.close()
        except Exception:
            pass
        return

    # Ensure indexes (scalar + vector) before writes.
    try:
        _ensure_collection_indexes(client, db_name, coll_name)
        index_name = os.getenv("MONGODB_VECTOR_INDEX_NAME", "contracts_vector_index")
        # Derive embedding dimension from first doc if available.
        emb_dim = 0
        for d in docs:
            emb = d.get("embedding")
            if isinstance(emb, list):
                emb_dim = len(emb)
                break
        _ensure_vector_search_index(client, db_name, coll_name, index_name, emb_dim)
    except Exception as e:
        logger.debug(f"Index ensure skipped: {e}")

    debug = os.getenv("MONGODB_DEBUG", "false").lower() == "true"
    if debug:
        try:
            hello = client.admin.command("hello")  # modern replacement for isMaster
            logger.debug(
                f"Mongo hello primary={hello.get('isWritablePrimary')} hosts={hello.get('hosts')} setName={hello.get('setName')}"
            )
        except Exception as e:
            logger.debug(f"Mongo hello command failed for debug: {e}")

    while attempt < max_retries:
        attempt += 1
        try:
            res = coll.bulk_write(ops, ordered=False)
            logger.info(
                f"Mongo upsert success (attempt {attempt}). matched={res.matched_count} upserted={len(res.upserted_ids)} modified={res.modified_count}"
            )
            client.close()
            return
        except (NotPrimaryError, AutoReconnect, OperationFailure) as e:
            # OperationFailure with code 10107 (NotWritablePrimary) should be treated similarly
            if isinstance(e, OperationFailure) and e.code != 10107:
                # Different operation failure; mark non-retryable
                last_err = e
                logger.error(
                    f"Mongo upsert OperationFailure (non-topology) code={e.code}: {e}"
                )
                break
            last_err = e
            if attempt < max_retries:
                logger.warning(
                    f"Mongo transient topology error '{e.__class__.__name__}: {e}'. Retrying in {sleep_ms} ms (attempt {attempt}/{max_retries})."
                )
                time.sleep(sleep_ms / 1000.0)
                # Trigger a topology check proactively
                try:
                    client.admin.command("ping")
                except Exception:
                    pass
                continue
            break
        except PyMongoError as e:
            # Non-retryable error; break immediately
            last_err = e
            logger.error(f"Mongo upsert failed with non-retryable error: {e}")
            break

    try:
        client.close()
    except Exception:
        pass

    # If we reach here, retries exhausted or non-retryable error occurred.
    raise RuntimeError(
        f"Mongo upsert failed after {attempt} attempts. Last error: {last_err}"
    )


class MongoDatabase:
    """Wrapper class implementing the DatabaseBackend protocol (ingestion only)."""

    def upsert_contract_chunks(self, docs: List[Dict[str, Any]]) -> None:  # type: ignore[override]
        upsert_contract_chunks(docs)


__all__ = [
    "MongoDatabase",
    "have_mongo",
    "upsert_contract_chunks",
]
