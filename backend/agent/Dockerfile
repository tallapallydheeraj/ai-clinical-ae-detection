FROM python:3.12-bookworm

# Arguments for spaCy model selection
ARG SPACY_MODEL=en_core_web_lg

ENV SPACY_MODEL=${SPACY_MODEL}

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download ${SPACY_MODEL}

# Copy application source code
COPY src/ ./src/

# Copy test folder
COPY test/ ./test/

# Pre-download tiktoken encodings during build to avoid runtime network calls
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache
RUN mkdir -p /app/.tiktoken_cache && \
    python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base'); print('Tiktoken cache populated')" && \
    chmod -R 755 /app/.tiktoken_cache

# Set ownership for the default user
USER 0
RUN chown -R 1001:0 /app && \
    chmod -R g=u /app

# Switch to non-root user
# USER 1001

# Expose application port
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "src.app.server:app", "--host", "0.0.0.0", "--port", "8080"]