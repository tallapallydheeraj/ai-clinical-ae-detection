from typing import List, Dict, Any, Optional, TYPE_CHECKING
import os
import time
import asyncio
from threading import Lock
from anyio import to_thread
from ..models.embeddings import get_embedding_client
from ..util.logger import get_logger

logger = get_logger("mongo")


def _get_mongo_uri() -> str:
    """Build MongoDB connection URI based on authentication mechanism.

    Returns:
        MongoDB connection URI string

    Raises:
        RuntimeError: If required environment variables are not set
    """
    auth_mechanism = os.getenv("AE_MONGO_AUTH_MECHANISM", "").strip().upper()

    if auth_mechanism == "MONGODB-AWS":
        # AWS IAM Authentication - uses IRSA credentials automatically
        cluster_host = os.getenv("AE_MONGO_CLUSTER_HOST", "").strip()
        if not cluster_host:
            raise RuntimeError(
                "AE_MONGO_CLUSTER_HOST not configured for AWS IAM authentication"
            )

        uri = (
            f"mongodb+srv://{cluster_host}/"
            f"?authSource=%24external"
            f"&authMechanism=MONGODB-AWS"
            f"&retryWrites=true"
            f"&w=majority"
        )
        logger.info(f"Using AWS IAM authentication for MongoDB Atlas: {cluster_host}")
        return uri
    else:
        # Traditional connection string with username/password
        uri = os.getenv("AE_MONGO_URI", "").strip()
        if not uri:
            raise RuntimeError("AE_MONGO_URI not configured")
        logger.info("Using traditional connection string for MongoDB")
        return uri


def have_mongo() -> bool:
    """Check if MongoDB is configured.

    Returns:
        True if MongoDB is configured (either via IAM auth or connection string)
    """
    auth_mechanism = os.getenv("AE_MONGO_AUTH_MECHANISM", "").strip().upper()
    if auth_mechanism == "MONGODB-AWS":
        return bool(os.getenv("AE_MONGO_CLUSTER_HOST", "").strip())
    return bool(os.getenv("AE_MONGO_URI", "").strip())


_CLIENT = None
_CLIENT_LOCK = Lock()
_ASYNC_CLIENT: Optional[Any] = None
_ASYNC_LOCK = Lock()


def _create_mongo_client():
    """Instantiate a new MongoClient. Internal only; use get_mongo_client() for reuse."""
    from pymongo import MongoClient

    uri = _get_mongo_uri()
    server_sel_ms = int(os.getenv("MONGODB_SERVER_SELECTION_MS", "30000"))

    return MongoClient(
        uri,
        appname=os.getenv("AE_PROJECT_NAME", "clinical-ae"),
        serverSelectionTimeoutMS=server_sel_ms,
        retryWrites=True,
    )


def get_mongo_client():
    """Return a persistent MongoClient instance (lazy-init on first use).

    Guarded by a lock to prevent duplicate instantiation under concurrent first access.
    """
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = _create_mongo_client()
    return _CLIENT


def _create_async_mongo_client():
    """Create async MongoDB client for use with Motor."""
    try:
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "motor not installed; cannot use async Mongo pipeline"
        ) from e

    uri = _get_mongo_uri()
    server_sel_ms = int(os.getenv("MONGODB_SERVER_SELECTION_MS", "30000"))

    return AsyncIOMotorClient(
        uri, serverSelectionTimeoutMS=server_sel_ms, retryWrites=True
    )


def get_async_mongo_client():
    """Return a persistent async MongoClient instance."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is not None:
        return _ASYNC_CLIENT
    with _ASYNC_LOCK:
        if _ASYNC_CLIENT is None:
            _ASYNC_CLIENT = _create_async_mongo_client()
    return _ASYNC_CLIENT


def search(query: str, therapy: str, k: int = 6) -> List[Dict[str, Any]]:
    """Vector similarity search for contract snippets.

    Behavior:
    - Performs an explicit ping with configurable retries/backoff before running the vector search.
    - Raises a RuntimeError if connection cannot be established after retries (so upstream can abort before LLM).
    - Raises on PyMongo errors instead of returning an empty list silently.
    """
    if not have_mongo():
        raise RuntimeError("MongoDB not configured; vector search unavailable.")

    from pymongo.errors import PyMongoError

    db_name = os.getenv("MONGODB_DB_NAME", "pharmacovigilance")
    coll_name = os.getenv("MONGODB_CONTRACTS_COLLECTION", "contracts")
    index_name = os.getenv("AE_MONGO_VECTOR_INDEX", "contracts_vector_index")
    retries = int(os.getenv("AE_MONGO_CONNECT_RETRIES", "3"))
    backoff_ms = int(os.getenv("AE_MONGO_CONNECT_BACKOFF_MS", "500"))

    ec = get_embedding_client()
    qv = ec.embed_documents([query])[0]

    client = get_mongo_client()
    try:
        # Connection ping with retry (only performed if first use or previous ping failed).
        for attempt in range(retries):
            try:
                client.admin.command("ping")
                break
            except PyMongoError as e:
                if attempt == retries - 1:
                    logger.error(
                        f"Mongo connection failed after {retries} attempts: {e}"
                    )
                    # Drop cached client so a future attempt can recreate
                    try:
                        client.close()
                    except Exception:
                        pass
                    global _CLIENT
                    with _CLIENT_LOCK:
                        _CLIENT = None
                    raise RuntimeError("Mongo connection failure") from e
                logger.warning(
                    f"Mongo connection attempt {attempt+1} failed: {e}; retrying in {backoff_ms}ms"
                )
                time.sleep(backoff_ms / 1000.0)

        coll = client[db_name][coll_name]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": index_name,
                    "path": "embedding",
                    "queryVector": qv,
                    "numCandidates": max(100, k * 10),
                    "limit": k,
                    "filter": {"therapy": therapy},
                }
            },
            {
                "$project": {
                    "text": 1,
                    "therapy": 1,
                    "contract_id": 1,
                    "chunk_id": 1,
                    "_score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = list(coll.aggregate(pipeline))
        for r in results:
            r["_score"] = float(r.get("_score", 0.0))
        return results
    except PyMongoError as e:
        logger.error(f"Mongo search error: {e}")
        raise RuntimeError("Mongo search failure") from e


async def search_async(query: str, therapy: str, k: int = 6) -> List[Dict[str, Any]]:
    """Async version of vector similarity search."""
    if not have_mongo():
        raise RuntimeError("MongoDB not configured; vector search unavailable.")

    db_name = os.getenv("MONGODB_DB_NAME", "pharmacovigilance")
    coll_name = os.getenv("MONGODB_CONTRACTS_COLLECTION", "contracts")
    index_name = os.getenv("AE_MONGO_VECTOR_INDEX", "contracts_vector_index")
    retries = int(os.getenv("AE_MONGO_CONNECT_RETRIES", "3"))
    backoff_ms = int(os.getenv("AE_MONGO_CONNECT_BACKOFF_MS", "500"))

    ec = get_embedding_client()
    qv = await to_thread.run_sync(lambda: ec.embed_documents([query])[0])

    client = get_async_mongo_client()
    for attempt in range(retries):
        try:
            await client.admin.command("ping")
            break
        except Exception as e:
            if attempt == retries - 1:
                try:
                    client.close()
                except Exception:
                    pass
                global _ASYNC_CLIENT
                with _ASYNC_LOCK:
                    _ASYNC_CLIENT = None
                raise RuntimeError("Mongo connection failure") from e
            await asyncio.sleep(backoff_ms / 1000.0)

    coll = client[db_name][coll_name]
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": qv,
                "numCandidates": max(100, k * 10),
                "limit": k,
                "filter": {"therapy": therapy},
            }
        },
        {
            "$project": {
                "text": 1,
                "therapy": 1,
                "contract_id": 1,
                "chunk_id": 1,
                "_score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    cursor = coll.aggregate(pipeline)
    out: List[Dict[str, Any]] = []
    async for doc in cursor:
        doc["_score"] = float(doc.get("_score", 0.0))
        out.append(doc)
        if len(out) >= k:
            break
    return out
