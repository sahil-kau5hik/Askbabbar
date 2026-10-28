"""
FastAPI Application: Love Babbar DSA RAG

Endpoints:
  GET  /health              - Service health + collection stats
  POST /query               - Full RAG: embed query -> search -> LLM answer
  POST /search              - Retrieval only (no LLM), returns TopicSources
  GET  /collection/info     - Qdrant collection metadata

Startup:
  - Eagerly loads embedding model (warms the singleton)
  - Validates Qdrant connectivity
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.embeddings.embedder import get_embedding_model
from backend.embeddings.vector_store import get_collection_info, get_qdrant_client
from backend.llm.generator import generate_answer
from backend.models.schemas import QueryRequest, QueryResponse, TopicSource
from backend.retrieval.retriever import search
from backend.utils.logging import logger


# ---------------------------------------------------------------------------
# Lifespan: eager resource loading at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads embedding model and checks Qdrant on startup."""
    logger.info("=== RAG API Starting Up ===")
    try:
        get_embedding_model()          # Warm embedding model singleton
        get_qdrant_client()            # Validate Qdrant connection
        info = get_collection_info()
        if info:
            logger.info(f"Qdrant collection ready: {info}")
        else:
            logger.warning(
                f"Collection '{settings.QDRANT_COLLECTION}' not found yet. "
                "Run script 05_embed_and_upload.py first."
            )
    except Exception as exc:
        logger.error(f"Startup warning (non-fatal): {exc}")

    yield

    logger.info("=== RAG API Shutting Down ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Love Babbar DSA RAG API",
    description=(
        "Retrieval-Augmented Generation over Love Babbar's Complete C++ Placement "
        "DSA Course. Ask any DSA/C++ question and get answers grounded in lecture transcripts."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow frontend dev servers (Live Server 5500, Vite 5173, Next 3000, 8080, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> Dict[str, Any]:
    """
    Returns service health status and Qdrant collection stats.
    """
    collection_info = get_collection_info()
    return {
        "status": "ok",
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "collection_info": collection_info,
    }


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
      1. Embed the query
      2. Search Qdrant for top_k relevant chunks
      3. Generate a grounded LLM answer from retrieved context
      4. Return answer + cited sources with YouTube timestamps

    - `query`: Your DSA / C++ question
    - `top_k`: Number of sources to retrieve (default 5)
    - `similarity_threshold`: Minimum cosine score to include (default 0.35)
    - `expand_adjacent`: Whether to include adjacent chunk context (default true)
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    logger.info(f"[/query] Received: '{request.query}'")

    # Step 1: Retrieve
    sources: List[TopicSource] = search(request)

    # Step 2: Generate
    response: QueryResponse = generate_answer(
        query=request.query,
        sources=sources,
    )

    return response


@app.post("/search", response_model=List[TopicSource], tags=["RAG"])
async def search_only(request: QueryRequest) -> List[TopicSource]:
    """
    Retrieval-only endpoint. Returns ranked transcript chunks without LLM generation.
    Useful for debugging relevance or building custom UIs.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    logger.info(f"[/search] Received: '{request.query}'")
    return search(request)


@app.get("/collection/info", tags=["System"])
async def collection_info() -> Dict[str, Any]:
    """Returns Qdrant collection metadata (point count, vector dim, status)."""
    info = get_collection_info()
    if not info:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Collection '{settings.QDRANT_COLLECTION}' not found. "
                "Run 05_embed_and_upload.py first."
            ),
        )
    return info


# ---------------------------------------------------------------------------
# Static frontend mount
# ---------------------------------------------------------------------------
from pathlib import Path
from fastapi.staticfiles import StaticFiles

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
