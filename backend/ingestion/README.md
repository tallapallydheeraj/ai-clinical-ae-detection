## Ingestion Service

Contract ingestion pipeline:

1. Loads raw contract texts (local dev fixtures or S3 bucket in higher envs).
2. Normalizes and chunks text (section or semantic strategy).
3. Embeds each chunk using Azure OpenAI or HuggingFace embeddings.
4. Persists chunk + embedding to MongoDB (required; no local filesystem fallback).

### Quick Start (Local)

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ENVIRONMENT=dev
python -m src.app.run
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:ENVIRONMENT='dev'
python -m src.app.run
```

**Windows (Command Prompt):**

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set ENVIRONMENT=dev
python -m src.app.run
```

### Key Environment Variables

Core:

- `ENVIRONMENT` = dev | qa | prod
- `CHUNK_STRATEGY` = section | semantic (default section)
- `CHUNK_MAX_CHARS` override max chars per chunk
- `CHUNK_MAX_TOKENS` optional token-based limit (if set and tokenizer available)

Semantic strategy extras:

- `SEMANTIC_MIN_SIMILARITY` (float, default 0.40)
- `SEMANTIC_MAX_SENTENCES` (int, default 25)
- `SEMANTIC_SIMILARITY_WINDOW` (int, default 3)
- Adaptive (optional):
  - `SEMANTIC_ADAPTIVE_THRESHOLD` true|false enables variance-based dynamic threshold scaling
  - `SEMANTIC_TARGET_VARIANCE` target similarity variance (default 0.04)
  - `SEMANTIC_THRESHOLD_SCALE_MIN` / `SEMANTIC_THRESHOLD_SCALE_MAX` clamp scaling (defaults 0.7 / 1.3)

Database (MongoDB only, ingestion writes only):

- `MONGODB_ATLAS_URI` connection string (required)
- `MONGODB_DB_NAME` (default pharmacovigilance)
- `MONGODB_CONTRACTS_COLLECTION` (default contracts)
- `MONGODB_VECTOR_INDEX_NAME` (default contracts_vector_index; created if missing for downstream retrieval service)
- `INGEST_PURGE_BEFORE` true|false delete existing chunks for the contract(s) before upsert to avoid stale data
- Advanced (optional): `MONGODB_MAX_RETRIES` (default 3), `MONGODB_RETRY_SLEEP_MS` (default 500), `MONGODB_SERVER_SELECTION_MS` (default 30000), `MONGODB_WRITE_CONCERN` (default majority), `MONGODB_DEBUG` (true/false for topology debug logging)

Embedding:

- `EMBEDDINGS_PROVIDER` = azure | huggingface (default: azure)
- `EMBEDDINGS_DIM` (optional) hint for embedding dimension

Azure provider (`EMBEDDINGS_PROVIDER=azure`):

- `EMBEDDINGS_API_KEY` or OAuth via `AE_AICOE_OAUTH_*` vars
- `EMBEDDINGS_ENDPOINT` Azure OpenAI endpoint URL
- `EMBEDDINGS_API_VERSION` API version string
- `EMBEDDINGS_DEPLOYMENT` deployment name

HuggingFace provider (`EMBEDDINGS_PROVIDER=huggingface`):

- `HUGGINGFACE_MODEL` model name (default: all-MiniLM-L6-v2)
- `HUGGINGFACE_DEVICE` device to use: cpu | cuda | mps (default: auto-detect)

Popular HuggingFace models:

- `all-MiniLM-L6-v2` (fast, 384 dims) - good for development
- `all-mpnet-base-v2` (better quality, 768 dims)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

Logging:

- `LOG_LEVEL` INFO|DEBUG|...
- `LOG_JSON` true|false structured JSON output toggle

### Architecture Overview

Layered responsibilities:

- `util/text_normalize.py` lightweight cleanup only.
- `util/chunking.py` chunking algorithms (section & semantic).
- `models/embeddings.py` Embedding client (Azure or HuggingFace).
- `db/database.py` backend Protocol + factory (Mongo only currently).
- `db/mongo.py` production Mongo implementation with retry.

### Testing

Run tests (pytest):

```bash
pytest backend/ingestion/test -q
```

### Adding a New Database Backend

Implement `upsert_contract_chunks` & `search` in a new module then extend the factory in `db/database.py`. The service will raise if Mongo isn't configured—additional backends should also fail fast when misconfigured.

### Notes

Semantic chunking uses embeddings (Azure or HuggingFace) for sentence similarity; if embeddings fail at runtime, it falls back to section-based chunking. Persistence requires a MongoDB connection. Retrieval/search is implemented separately.

For local development without Azure access, use HuggingFace embeddings:

```bash
export EMBEDDINGS_PROVIDER=huggingface
export HUGGINGFACE_MODEL=all-MiniLM-L6-v2
```
