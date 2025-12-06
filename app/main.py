# ================================================================
# main.py — Complete Semantic Search API (Single File Version)
# ================================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

import numpy as np
import json
from pathlib import Path
from functools import wraps
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import mlflow

# ================================================================
# Configuration
# ================================================================

from pipeline.config import (
    MODEL_NAME,
    METADATA_FILE,
    ABSTRACTS_FILE,
    EMBEDDINGS_FILE,
    IDS_FILE,
)

# Toggle MLflow logging for docker / pipeline separation
DISABLE_MLFLOW = True


# ================================================================
# Pydantic Models
# ================================================================

class EmbedRequest(BaseModel):
    text: str = Field(..., example="quantum physics is amazing!")


class SearchRequest(BaseModel):
    query: str = Field(..., example="quantum computing algorithms")
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResult(BaseModel):
    id: int
    score: float
    abstract: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    ready: bool


class LivenessResponse(BaseModel):
    alive: bool


class InfoResponse(BaseModel):
    service: str
    model_name: str
    metadata: dict


# ================================================================
# Global Cached Model + Data Loaders
# ================================================================

_model: Optional[SentenceTransformer] = None
_embeddings: Optional[np.ndarray] = None
_ids: Optional[np.ndarray] = None
_abstracts: Optional[dict] = None
_app_metadata: Optional[dict] = None


def load_model():
    global _model
    if _model is None:
        print(f"📌 Loading model {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_embeddings():
    global _embeddings, _ids
    if _embeddings is None:
        print("📌 Loading embeddings...")
        _embeddings = np.load(EMBEDDINGS_FILE)
        _ids = np.load(IDS_FILE)
    return _embeddings, _ids


def load_abstracts():
    global _abstracts
    if _abstracts is None:
        print("📌 Loading abstracts...")
        import pandas as pd
        df = pd.read_json(ABSTRACTS_FILE, lines=True)
        _abstracts = dict(zip(df["id"], df["abstract"]))
    return _abstracts


def load_metadata():
    global _app_metadata
    if _app_metadata is None:
        print("📌 Loading metadata file...")
        _app_metadata = json.loads(Path(METADATA_FILE).read_text())
    return _app_metadata


# ================================================================
# Utility Functions
# ================================================================

def embed_text(text: str) -> np.ndarray:
    model = load_model()
    return model.encode([text], normalize_embeddings=True)


def search_similar(query: str, top_k: int):
    embeddings, ids = load_embeddings()
    abstracts = load_abstracts()

    query_vec = embed_text(query)
    sims = cosine_similarity(query_vec, embeddings)[0]
    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        paper_id = int(ids[i])
        results.append({
            "id": paper_id,
            "score": float(sims[i]),
            "abstract": abstracts[paper_id],
        })
    return results


def log_search_event(query: str, top_k: int, scores: List[float], model_name: str):
    if DISABLE_MLFLOW:
        return
    with mlflow.start_run(run_name="api-search-event"):
        mlflow.log_param("query", query)
        mlflow.log_param("top_k", top_k)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("avg_score", float(np.mean(scores)))
        mlflow.log_metric("max_score", float(np.max(scores)))


# ================================================================
# Prometheus Metrics
# ================================================================

REQUEST_COUNT = Counter(
    "request_count",
    "Total number of requests",
    ["endpoint"]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)


def track_metrics(endpoint: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(endpoint=endpoint).inc()
            with REQUEST_LATENCY.labels(endpoint=endpoint).time():
                return await func(*args, **kwargs)
        return wrapper
    return decorator


# ================================================================
# FastAPI App Setup
# ================================================================

app = FastAPI(title="Semantic Search API")


@app.on_event("startup")
def startup_event():
    print("🚀 Starting API...")
    load_metadata()
    load_embeddings()
    load_abstracts()
    load_model()
    print("✅ Startup complete.")


@app.on_event("shutdown")
def shutdown_event():
    print("🛑 Shutting down API.")


# ================================================================
# Endpoints
# ================================================================

@app.get("/health", response_model=HealthResponse)
@track_metrics("health")
async def health():
    return HealthResponse(status="ok")


@app.get("/liveness", response_model=LivenessResponse)
@track_metrics("liveness")
async def liveness():
    return LivenessResponse(alive=True)


@app.get("/readiness", response_model=ReadinessResponse)
@track_metrics("readiness")
async def readiness():
    ready = _model is not None and _embeddings is not None
    return ReadinessResponse(ready=ready)


@app.get("/info", response_model=InfoResponse)
@track_metrics("info")
async def info():
    return InfoResponse(
        service="Semantic Search API",
        model_name=MODEL_NAME,
        metadata=load_metadata(),
    )


@app.post("/embed")
@track_metrics("embed")
async def embed(req: EmbedRequest):
    vector = embed_text(req.text)
    return {"embedding": vector[0].tolist()}


@app.post("/search", response_model=SearchResponse)
@track_metrics("search")
async def search(req: SearchRequest):
    results = search_similar(req.query, req.top_k)

    scores = [r["score"] for r in results]
    log_search_event(req.query, req.top_k, scores, MODEL_NAME)

    return SearchResponse(results=[SearchResult(**r) for r in results])


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "message": "Semantic Search API",
        "endpoints": [
            "/health",
            "/liveness",
            "/readiness",
            "/info",
            "/embed",
            "/search",
            "/metrics",
        ]
    }
