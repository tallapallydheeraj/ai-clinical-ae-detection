# Clinical Adverse Event Detection AI – Backend

This directory contains the backend services for the Clinical Adverse Event Detection system. The architecture is designed for modularity, scalability, and extensibility.

## Services Overview

### 1. **Agent Service** (`agent/`)

Core FastAPI service that orchestrates retrieval and LLM-based evaluation to classify clinical assessments as potential adverse events.

**Key Features:**

- Retrieval-Augmented Generation (RAG) using contract data
- Multiple LLM providers: Azure OpenAI, HuggingFace, Ollama
- Multiple embedding providers: Azure OpenAI, HuggingFace
- FHIR QuestionnaireResponse intake
- Contract-based validation and matching
- Explainable decision output with confidence scores
- LangSmith tracing for observability
- PHI redaction support

**API:** FastAPI with interactive docs at `/docs`

**Setup:** [See agent/README.md](agent/README.md)

### 2. **Ingestion Service** (`ingestion/`)

Data ingestion and preprocessing pipeline for clinical contracts and knowledge bases.

**Key Features:**

- Contract text normalization and chunking
- Flexible chunking strategies (section-based or semantic)
- Embedding generation and storage
- MongoDB integration with vector indexing
- Support for local and cloud storage sources
- Configurable embedding models

**Usage:** Batch processing script for populating knowledge base

**Setup:** [See ingestion/README.md](ingestion/README.md)

## Technology Stack

| Component         | Options                           | Default      |
| ----------------- | --------------------------------- | ------------ |
| **LLM Provider**  | Azure OpenAI, HuggingFace, Ollama | Azure OpenAI |
| **Embeddings**    | Azure OpenAI, HuggingFace         | Azure OpenAI |
| **Database**      | MongoDB                           | MongoDB      |
| **API Framework** | FastAPI                           | FastAPI      |
| **Async Runtime** | uvicorn / Python async            | uvicorn      |

## Directory Structure

```
backend/
├── agent/              # Detection and decision API
│   ├── src/
│   │   ├── app/        # FastAPI server and routes
│   │   ├── models/     # LLM and embedding clients
│   │   ├── rag/        # Retrieval and reasoning graphs
│   │   ├── db/         # Database clients
│   │   └── util/       # Utilities (logging, redaction, rules)
│   ├── test/           # Test data and outputs
│   └── requirements.txt
├── ingestion/          # Data preprocessing pipeline
│   ├── src/
│   │   ├── app/        # Entry point
│   │   ├── models/     # Embedding clients
│   │   ├── db/         # Database clients
│   │   └── util/       # Utilities (chunking, normalization)
│   ├── test/           # Test data and sample contracts
│   └── requirements.txt
└── README.md           # This file
```

## Getting Started

Each service has its own README with step-by-step setup instructions:

1. **Agent Service:** [agent/README.md](agent/README.md)
   - Quick start guides (macOS/Linux/Windows)
   - Environment configuration details
   - Docker deployment
   - LangSmith observability setup
   - Troubleshooting guide

2. **Ingestion Service:** [ingestion/README.md](ingestion/README.md)
   - Contract ingestion workflow
   - Environment variables reference
   - Chunking strategies
   - Database configuration

## Common Tasks

### Run Agent Service Locally

```bash
cd agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg
uvicorn src.app.server:app --host 0.0.0.0 --port 8080 --reload
```

### Ingest Contract Data

```bash
cd ingestion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ENVIRONMENT=dev
python -m src.app.run
```

### Run Both Services with Docker

```bash
# From backend/agent
docker-compose up -d
```

## Environment Variables

Both services use `.env` files for configuration. Key variables:

**Shared:**

- `ENVIRONMENT` – dev, qa, prod
- `LOG_LEVEL` – INFO, DEBUG, etc.
- `EMBEDDINGS_PROVIDER` – azure or huggingface

**Agent Service:**

- `AE_LLM_PROVIDER` – azure, huggingface, llama, ollama
- `AE_AZURE_ENDPOINT`, `AE_AZURE_API_KEY` – Azure OpenAI credentials
- `AE_HUGGINGFACE_LLM_MODEL` – HuggingFace model ID

**Ingestion Service:**

- `MONGODB_ATLAS_URI` – MongoDB connection string
- `CHUNK_STRATEGY` – section or semantic
- `HUGGINGFACE_MODEL` – HuggingFace embedding model

See service READMEs for comprehensive environment variable references.

## Database Setup

### MongoDB

Both services require a MongoDB instance:

1. **Local Development:**

   ```bash
   # Using Docker
   docker run -d -p 27017:27017 --name mongo mongo:latest

   # Or install MongoDB locally: https://docs.mongodb.com/manual/installation/
   ```

2. **MongoDB Atlas (Cloud):**
   - Create account at https://www.mongodb.com/cloud/atlas
   - Create cluster and get connection string
   - Set `MONGODB_ATLAS_URI` environment variable

3. **Vector Indexing:**
   - Created automatically by ingestion service
   - Required for RAG retrieval in agent service

## Testing

### Agent Service Tests

```bash
cd agent
pytest test/ -v
```

### Ingestion Service Tests

```bash
cd ingestion
pytest test/ -q
```

## Security & Best Practices

1. **PHI Redaction:** Enable `AE_REDACT_PHI=true` to mask sensitive data before processing
2. **API Keys:** Use `.env` files for secrets; never commit credentials
3. **OAuth:** Supported for Azure services via `AE_AICOE_OAUTH_*` variables
4. **Database Security:** Use MongoDB Atlas IP allowlisting and encrypted connections
5. **Observability:** Enable LangSmith tracing for production deployments

## Contributing

To add new features or services:

1. Follow the existing module structure
2. Update this README and service-specific READMEs
3. Add tests in the `test/` directory
4. Document environment variables and configuration
5. Update docker-compose.yml if needed

## Support

- Check service-specific READMEs for detailed troubleshooting
- Review `.env.example` files for configuration templates
- Enable `LOG_LEVEL=DEBUG` for verbose logging
- Use LangSmith UI for tracing and debugging LLM calls
