"""
Embedder: Converts TranscriptChunks into dense float32 vectors.

Uses sentence-transformers (all-MiniLM-L6-v2 by default).
- Lazy model loading (singleton) to avoid repeated disk reads
- Batched encoding for efficiency
- Idempotent: skips already-embedded videos unless force=True
- Updates pipeline_manifest.json after each video
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import settings
from backend.ingestion.chunker import load_chunks_file
from backend.models.schemas import TranscriptChunk
from backend.utils.logging import logger

# ---------------------------------------------------------------------------
# Singleton model loader
# ---------------------------------------------------------------------------

_model = None


def get_embedding_model():
    """Returns the cached SentenceTransformer model (loads once per process)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL} on {settings.EMBEDDING_DEVICE}")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
        logger.info("Embedding model loaded successfully.")
    return _model


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[TranscriptChunk], batch_size: int = 64) -> np.ndarray:
    """
    Embeds a list of TranscriptChunks and returns a float32 numpy array
    of shape (len(chunks), embedding_dim).

    Text used: chunk.text (the merged transcript window).
    """
    if not chunks:
        return np.empty((0,), dtype=np.float32)

    model = get_embedding_model()
    texts = [c.text for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks in batches of {batch_size} ...")
    vectors: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity = dot product -> faster Qdrant search
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# Manifest helpers (mirrors chunker.py pattern exactly)
# ---------------------------------------------------------------------------

def update_manifest_embed_status(
    video_id: str,
    success: bool,
    error: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> None:
    """Updates `embedded` flag in pipeline_manifest.json."""
    m_path = manifest_path or settings.pipeline_manifest_path
    if not m_path.exists():
        return
    try:
        with open(m_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if video_id in manifest:
            manifest[video_id]["embedded"] = success
            manifest[video_id]["error"] = error
            manifest[video_id]["last_updated"] = datetime.now(timezone.utc).isoformat()
            with open(m_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning(f"Failed to update manifest embed status for {video_id}: {exc}")


# ---------------------------------------------------------------------------
# Single-video embed (returns chunks + vectors together)
# ---------------------------------------------------------------------------

def embed_single_video(
    video_id: str,
    chunks_dir: Optional[Path] = None,
    batch_size: int = 64,
) -> Tuple[bool, List[TranscriptChunk], Optional[np.ndarray], Optional[str]]:
    """
    Loads chunks for a video_id, embeds them, and returns:
        (success, chunks, vectors_array, error_message)

    vectors_array shape: (num_chunks, embedding_dim)  dtype=float32
    Does NOT write anything to disk -- caller decides what to do with vectors.
    """
    chunks = load_chunks_file(video_id, chunks_dir)
    if not chunks:
        err = f"No chunks found for video_id={video_id}"
        logger.warning(f"[{video_id}] {err}")
        update_manifest_embed_status(video_id, success=False, error=err)
        return False, [], None, err

    try:
        vectors = embed_chunks(chunks, batch_size=batch_size)
        logger.info(f"[{video_id}] Embedded {len(chunks)} chunks -> shape {vectors.shape}")
        update_manifest_embed_status(video_id, success=True)
        return True, chunks, vectors, None
    except Exception as exc:
        err = str(exc)
        logger.error(f"[{video_id}] Embedding failed: {err}")
        update_manifest_embed_status(video_id, success=False, error=err)
        return False, [], None, err


# ---------------------------------------------------------------------------
# Batch embed (returns dict of video_id -> (chunks, vectors))
# ---------------------------------------------------------------------------

def embed_batch(
    video_ids: List[str],
    chunks_dir: Optional[Path] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Embeds chunks for a list of video_ids.

    Returns summary dict:
        {
            "total": int,
            "embedded": int,
            "failed": int,
            "failed_ids": [...],
            "results": {video_id: (chunks, vectors)}
        }
    """
    summary: Dict[str, Any] = {
        "total": len(video_ids),
        "embedded": 0,
        "failed": 0,
        "failed_ids": [],
        "results": {},
    }

    for idx, vid in enumerate(video_ids, start=1):
        logger.info(f"Embedding ({idx}/{len(video_ids)}): {vid}")
        success, chunks, vectors, err = embed_single_video(vid, chunks_dir, batch_size)
        if success and vectors is not None:
            summary["embedded"] += 1
            summary["results"][vid] = (chunks, vectors)
        else:
            summary["failed"] += 1
            summary["failed_ids"].append({"video_id": vid, "error": err})

    return summary
