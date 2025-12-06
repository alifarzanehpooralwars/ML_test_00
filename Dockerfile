# ================================================================
# Stage 1 — Builder: install deps + download model + generate embeddings
# ================================================================
FROM python:3.12-slim AS builder

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project config first (cache layer)
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy entire source code
COPY . .

# Pre-download the model so embedding step is fast
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
EOF

# Disable MLflow during embedding generation
ENV DISABLE_MLFLOW=1

# Generate embeddings inside the Docker build
RUN python -m pipeline.embedder


# ================================================================
# Stage 2 — Final image: runtime only
# ================================================================
FROM python:3.12-slim

WORKDIR /app

# Install minimal runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy only the runtime source files + generated embeddings
COPY app/ app/
COPY pipeline/ pipeline/
COPY data/models/ data/models/
COPY data/arxiv_abstracts.jsonl data/arxiv_abstracts.jsonl

# Environment variables
ENV DISABLE_MLFLOW=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
