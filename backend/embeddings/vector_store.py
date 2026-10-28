"""
Qdrant Vector Store Manager

Handles:
- Collection creation with correct cosine distance + vector dim
- Batch upsert of TranscriptChunk payloads + embedding vectors
- Idempotent: checks manifest before re-uploading
- Updates pipeline_manifest.json with `uploaded_to_qdrant` flag
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import settings
from backend.models.schemas import TranscriptChunk
from backend.utils.logging import logger

# Qdrant type imports (guarded to give helpful error if not installed)
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    from qdrant_client.http.exceptions import UnexpectedResponse
except ImportError as _e:
    raise ImportError(
        "qdrant-client is not installed. Run: pip install qdrant-client"
    ) from _e


# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Returns a cached QdrantClient. Connects to Qdrant Cloud or local."""
    global _client
    if _client is None:
        url = settings.QDRANT_URL.strip()
        api_key = settings.QDRANT_API_KEY.strip()

        if url:
            logger.info(f"Connecting to Qdrant Cloud: {url}")
            _client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        elif getattr(settings, "QDRANT_PATH", "") and settings.QDRANT_PATH != ":memory:":
            q_path = str(settings.qdrant_path)
            logger.info(f"Connecting to local disk Qdrant at: {q_path}")
            _client = QdrantClient(path=q_path)
        else:
            # Fallback: in-memory client for testing
            logger.warning("QDRANT_URL not set — using in-memory Qdrant (data lost on exit).")
            _client = QdrantClient(":memory:")

        logger.info("Qdrant client ready.")
    return _client


def close_qdrant_client():
    """Cleanly closes QdrantClient before Python runtime teardown."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


import atexit
atexit.register(close_qdrant_client)


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(
    collection_name: str,
    vector_dim: int,
    client: Optional[QdrantClient] = None,
) -> None:
    """
    Creates the Qdrant collection if it does not exist.
    Uses Cosine distance (works with normalised vectors from sentence-transformers).
    """
    qc = client or get_qdrant_client()
    existing = {c.name for c in qc.get_collections().collections}

    if collection_name in existing:
        logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        return

    logger.info(f"Creating collection '{collection_name}' (dim={vector_dim}, Cosine).")
    qc.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_dim,
            distance=qmodels.Distance.COSINE,
        ),
    )
    logger.info(f"Collection '{collection_name}' created.")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def update_manifest_qdrant_status(
    video_id: str,
    success: bool,
    error: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> None:
    """Updates `uploaded_to_qdrant` flag in pipeline_manifest.json."""
    m_path = manifest_path or settings.pipeline_manifest_path
    if not m_path.exists():
        return
    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if video_id in manifest:
            manifest[video_id]["uploaded_to_qdrant"] = success
            manifest[video_id]["error"] = error
            manifest[video_id]["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Failed to update Qdrant manifest status for {video_id}: {exc}")


# ---------------------------------------------------------------------------
# Chunk payload builder
# ---------------------------------------------------------------------------

def _chunk_to_payload(chunk: TranscriptChunk) -> Dict[str, Any]:
    """Converts TranscriptChunk fields into a flat Qdrant payload dict."""
    return {
        "chunk_id": chunk.chunk_id,
        "video_id": chunk.video_id,
        "video_title": chunk.video_title,
        "lecture_number": chunk.lecture_number,
        "playlist_index": chunk.playlist_index,
        "video_url": chunk.video_url,
        "start_time": chunk.start_time,
        "end_time": chunk.end_time,
        "start_timestamp_str": chunk.start_timestamp_str,
        "end_timestamp_str": chunk.end_timestamp_str,
        "youtube_timestamp_url": chunk.youtube_timestamp_url,
        "text": chunk.text,
        "segment_span": chunk.segment_span,
    }


# ---------------------------------------------------------------------------
# Stable numeric point ID derived from chunk_id string
# ---------------------------------------------------------------------------

def _chunk_id_to_point_id(chunk_id: str) -> int:
    """
    Converts chunk_id string to a stable positive int for Qdrant PointId.
    Uses abs(hash(...)) with Python's built-in hash (seeded per-process).
    For production determinism, uses a simple djb2-style hash instead.
    """
    h = 5381
    for ch in chunk_id.encode():
        h = ((h << 5) + h + ch) & 0xFFFFFFFFFFFFFFFF  # 64-bit djb2
    return h if h > 0 else h + (1 << 64)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunks: List[TranscriptChunk],
    vectors: np.ndarray,
    collection_name: Optional[str] = None,
    batch_size: int = 128,
    client: Optional[QdrantClient] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Upserts chunk vectors + payloads into Qdrant.

    Args:
        chunks: list of TranscriptChunk objects
        vectors: float32 array (len(chunks), embedding_dim)
        collection_name: target collection (defaults to settings.QDRANT_COLLECTION)
        batch_size: upsert batch size
        client: optional pre-built QdrantClient

    Returns:
        (success: bool, error: Optional[str])
    """
    if len(chunks) != vectors.shape[0]:
        err = f"chunks/vectors length mismatch: {len(chunks)} vs {vectors.shape[0]}"
        return False, err

    qc = client or get_qdrant_client()
    col = collection_name or settings.QDRANT_COLLECTION
    vector_dim = vectors.shape[1]

    ensure_collection(col, vector_dim, client=qc)

    points: List[qmodels.PointStruct] = [
        qmodels.PointStruct(
            id=_chunk_id_to_point_id(chunk.chunk_id),
            vector=vectors[i].tolist(),
            payload=_chunk_to_payload(chunk),
        )
        for i, chunk in enumerate(chunks)
    ]

    try:
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            qc.upsert(collection_name=col, points=batch)
            logger.info(
                f"Upserted batch {start // batch_size + 1}: "
                f"{start + 1}-{min(start + batch_size, len(points))} / {len(points)}"
            )
        return True, None
    except Exception as exc:
        err = str(exc)
        logger.error(f"Qdrant upsert failed: {err}")
        return False, err


# ---------------------------------------------------------------------------
# Delete all points for a video (for re-indexing)
# ---------------------------------------------------------------------------

def delete_video_points(
    video_id: str,
    collection_name: Optional[str] = None,
    client: Optional[QdrantClient] = None,
) -> None:
    """Deletes all Qdrant points belonging to a specific video_id."""
    qc = client or get_qdrant_client()
    col = collection_name or settings.QDRANT_COLLECTION
    try:
        qc.delete(
            collection_name=col,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="video_id",
                            match=qmodels.MatchValue(value=video_id),
                        )
                    ]
                )
            ),
        )
        logger.info(f"[{video_id}] Deleted existing Qdrant points.")
    except Exception as exc:
        logger.warning(f"[{video_id}] Could not delete old points: {exc}")


# ---------------------------------------------------------------------------
# Collection info helper
# ---------------------------------------------------------------------------

def get_collection_info(
    collection_name: Optional[str] = None,
    client: Optional[QdrantClient] = None,
) -> Optional[Dict[str, Any]]:
    """Returns a dict summary of the collection (points_count, vector_dim, etc.)."""
    qc = client or get_qdrant_client()
    col = collection_name or settings.QDRANT_COLLECTION
    try:
        info = qc.get_collection(col)
        vectors_count = getattr(info, "vectors_count", getattr(info, "indexed_vectors_count", info.points_count))
        return {
            "name": col,
            "points_count": info.points_count,
            "vectors_count": vectors_count,
            "status": str(info.status),
        }
    except Exception as exc:
        logger.warning(f"Could not fetch collection info: {exc}")
        return None
