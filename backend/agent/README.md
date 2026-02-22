# Clinical Adverse Event Detection – Agent Service

FastAPI service that orchestrates retrieval + LLM evaluation to classify questionnaire assessments as potential adverse events. Supports Azure OpenAI and HuggingFace models for both embeddings and LLM inference.

## Quick Start

### macOS / Linux

```bash
# Clone and navigate to agent directory
cd backend/agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Run the server
uvicorn src.app.server:app --host 0.0.0.0 --port 8080 --reload
```

### Windows (PowerShell)

```powershell
# Clone and navigate to agent directory
cd backend\agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Copy and configure environment
copy .env.example .env
# Edit .env with your settings

# Run the server
uvicorn src.app.server:app --host 0.0.0.0 --port 8080 --reload
```

### Windows (Command Prompt)

```cmd
# Clone and navigate to agent directory
cd backend\agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg

# Copy and configure environment
copy .env.example .env
# Edit .env with your settings

# Run the server
uvicorn src.app.server:app --host 0.0.0.0 --port 8080 --reload
```

Visit: http://localhost:8080/docs

## Environment Configuration

### LLM Provider (`AE_LLM_PROVIDER`)

| Provider           | Description              | Required Env Vars                                                                                    |
| ------------------ | ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `azure` (default)  | Azure OpenAI             | `AE_AZURE_ENDPOINT`, `AE_AZURE_CHAT_DEPLOYMENT`, `AE_AZURE_API_VERSION`, `AE_AZURE_API_KEY` or OAuth |
| `huggingface`      | HuggingFace Transformers | `AE_HUGGINGFACE_LLM_MODEL`                                                                           |
| `llama` / `ollama` | Local Ollama server      | `AE_LLAMA_MODEL`, `AE_OLLAMA_BASE_URL`                                                               |

### Embedding Provider (`AE_EMBED_PROVIDER`)

| Provider          | Description                 | Required Env Vars                                                  |
| ----------------- | --------------------------- | ------------------------------------------------------------------ |
| `azure` (default) | Azure OpenAI Embeddings     | `AE_AZURE_ENDPOINT`, `AE_EMBED_DEPLOYMENT`, `AE_AZURE_API_VERSION` |
| `huggingface`     | Local sentence-transformers | `AE_HUGGINGFACE_EMBED_MODEL`                                       |

### HuggingFace Configuration

**For LLM:**

```bash
AE_LLM_PROVIDER=huggingface
AE_HUGGINGFACE_LLM_MODEL=meta-llama/Llama-3.2-1B-Instruct  # or any HF model
AE_HUGGINGFACE_DEVICE=auto                                  # auto, cpu, cuda, mps
AE_HUGGINGFACE_LOAD_4BIT=false                             # enable 4-bit quantization
AE_HUGGINGFACE_LOAD_8BIT=false                             # enable 8-bit quantization
```

**For Embeddings:**

```bash
AE_EMBED_PROVIDER=huggingface
AE_HUGGINGFACE_EMBED_MODEL=all-MiniLM-L6-v2   # fast, 384 dims
AE_HUGGINGFACE_DEVICE=cpu                      # cpu, cuda, mps
AE_EMBED_DIM=384                               # optional dimension hint
```

**Popular HuggingFace Models:**

| Model                                | Type      | Dimensions/Size | Notes              |
| ------------------------------------ | --------- | --------------- | ------------------ |
| `all-MiniLM-L6-v2`                   | Embedding | 384 dims        | Fast, good for dev |
| `all-mpnet-base-v2`                  | Embedding | 768 dims        | Better quality     |
| `meta-llama/Llama-3.2-1B-Instruct`   | LLM       | 1B params       | Small, fast        |
| `meta-llama/Llama-3.2-3B-Instruct`   | LLM       | 3B params       | Better quality     |
| `mistralai/Mistral-7B-Instruct-v0.3` | LLM       | 7B params       | High quality       |

### Ollama Configuration

```bash
AE_LLM_PROVIDER=llama
AE_LLAMA_MODEL=llama3.2
AE_OLLAMA_BASE_URL=http://localhost:11434
```

## Running with Docker

### Build the Image

```bash
docker build -t clinical-ae-agent:latest .
```

### Run the Container

```bash
docker run --rm -p 8080:8080 --env-file .env clinical-ae-agent:latest
```

### Using HuggingFace with Docker

For GPU support with HuggingFace models:

```bash
docker run --rm -p 8080:8080 --gpus all --env-file .env clinical-ae-agent:latest
```

## LangSmith Tracing

Observability is integrated in two layers:

1. Automatic tracing of LangChain components (prompt, model, parser) when `LANGSMITH_TRACING_V2=true` and `LANGSMITH_API_KEY` is set.
2. Custom decision runs emitted by the application using `safe_log_decision()` (see `src/util/langsmith.py`).

### Enable

Set the following in your `.env`:

```
LANGSMITH_TRACING_V2=true
LANGSMITH_API_KEY=sk_live_...
LANGSMITH_PROJECT=clinical-ae-agent   # optional
AE_LANGSMITH_LOG_DECISIONS=true       # custom run logging
```

### What Gets Logged

| Layer               | Content                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| LangChain Tracing   | Prompt text, model config (temperature, max_tokens), output JSON parsing span.                                              |
| Custom Decision Run | therapy, summary & contract lengths, filtering flags, final decision object (is_adverse_event, criteria, confidence, etc.). |

Response payloads now include `trace_run_id` when a custom run is recorded.

### Disabling

Turn off either layer independently:

```
LANGSMITH_TRACING_V2=false      # disables automatic spans
AE_LANGSMITH_LOG_DECISIONS=false  # keeps chain spans but no custom decision runs
```

## Code Pointers

| File                       | Purpose                                                  |
| -------------------------- | -------------------------------------------------------- |
| `src/app/server.py`        | FastAPI endpoints; instrumentation occurs post-decision. |
| `src/models/llm.py`        | LLM decision logic; supports Azure, HuggingFace, Ollama. |
| `src/models/embeddings.py` | Embedding client; supports Azure and HuggingFace.        |
| `src/util/langsmith.py`    | Safe helpers wrapping LangSmith client.                  |
| `src/rag/graph.py`         | Graph construction (retrieval + decision chain).         |

## Security & PHI

If `AE_REDACT_PHI=true`, text is masked prior to embeddings and LLM invocation; only redacted content is traced. Custom runs contain the structured decision object but not raw PHI text.

## Troubleshooting

| Symptom                         | Cause                                                          | Fix                                               |
| ------------------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| No traces in LangSmith UI       | Missing/invalid `LANGSMITH_API_KEY`                            | Set valid key & restart server                    |
| `trace_run_id` always null      | `AE_LANGSMITH_LOG_DECISIONS` not true or langsmith lib missing | Set flag, ensure dependency installed             |
| 401 errors from LangSmith       | Key revoked or wrong endpoint                                  | Verify key / endpoint                             |
| HuggingFace model download slow | Large model first download                                     | Use smaller model or pre-download                 |
| CUDA out of memory              | Model too large for GPU                                        | Enable `AE_HUGGINGFACE_LOAD_4BIT=true` or use CPU |
| Ollama connection refused       | Ollama not running                                             | Start Ollama: `ollama serve`                      |

## Extending Instrumentation

You can log additional domain events:

```python
from src.util.langsmith import safe_log_decision
safe_log_decision(inputs={"event":"contract_refresh"}, outputs={"count":42}, tags=["maintenance"])  # no-op if disabled
```

## Push to Docker Hub

To push the built agent image to your Docker Hub account, tag the image with your Docker Hub username and push it.

1. Build and tag (replace `your-docker-username` and `v1.0`):

```bash
docker build -t your-docker-username/clinical-ae-agent:v1.0 .
```

2. Log in to Docker Hub:

```bash
docker login
```

3. Push the image:

```bash
docker push your-docker-username/clinical-ae-agent:v1.0
```

Optional: tag `latest` and push

```bash
docker tag your-docker-username/clinical-ae-agent:v1.0 your-docker-username/clinical-ae-agent:latest
docker push your-docker-username/clinical-ae-agent:latest
```

Notes:

- If you prefer `docker-compose`, first set the image name in the `docker-compose.yml` and run `docker-compose build` then `docker push <image>`.
- For private repositories, ensure your account has appropriate permissions and you are logged in before pushing.
