#!/usr/bin/env python3
"""
Master Pipeline Orchestrator: Love Babbar DSA RAG
Runs the complete ingestion pipeline end-to-end with optimal batching and progress tracking:
  Phase 1: Playlist metadata sync (cached)
  Phase 2: Audio extraction (yt-dlp)
  Phase 3: Timestamped transcription (faster-whisper)
  Phase 4: Sliding-window chunking
  Phase 5: Embedding & Vector DB upsert (sentence-transformers -> Qdrant)

Usage:
  python scripts/run_pipeline.py --lectures 12
  python scripts/run_pipeline.py --limit 3
  python scripts/run_pipeline.py --all
  python scripts/run_pipeline.py --status
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from backend.config import settings
from backend.embeddings.embedder import embed_single_video
from backend.embeddings.vector_store import get_collection_info, upsert_chunks, update_manifest_qdrant_status
from backend.ingestion.chunker import chunk_single_video
from backend.ingestion.downloader import download_audio_batch
from backend.ingestion.playlist import fetch_playlist_metadata, load_cached_videos
from backend.ingestion.transcriber import transcribe_audio_file
from backend.models.schemas import VideoMetadata
from backend.utils.logging import logger


def get_pipeline_status():
    """Prints a high-level summary of the entire ingestion pipeline."""
    import json
    m_path = settings.pipeline_manifest_path
    if not m_path.exists():
        print("Pipeline manifest does not exist yet. Run Phase 1 first.")
        return

    with open(m_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    total = len(manifest)
    downloaded = sum(1 for v in manifest.values() if v.get("downloaded"))
    transcribed = sum(1 for v in manifest.values() if v.get("transcribed"))
    chunked = sum(1 for v in manifest.values() if v.get("chunked"))
    uploaded = sum(1 for v in manifest.values() if v.get("uploaded_to_qdrant"))

    col_info = get_collection_info()
    points = col_info.get("points_count", 0) if col_info else 0

    print("\n" + "=" * 55)
    print(" LOVE BABBAR DSA RAG — PIPELINE STATUS")
    print("=" * 55)
    print(f" Total Playlist Lectures Tracked:  {total}")
    print(f" Phase 2 - Audio Downloaded:       {downloaded}/{total}")
    print(f" Phase 3 - Audio Transcribed:      {transcribed}/{total}")
    print(f" Phase 4 - Transcript Chunked:     {chunked}/{total}")
    print(f" Phase 5 - Indexed in Qdrant:      {uploaded}/{total}")
    print(f" Total Active Vectors in Qdrant:   {points}")
    print("=" * 55 + "\n")


def run_pipeline(
    target_videos: List[VideoMetadata],
    force: bool = False,
    whisper_model: Optional[str] = None,
    batch_size: int = 64,
):
    total = len(target_videos)
    logger.info(f"=== Starting End-to-End Ingestion Pipeline for {total} Lecture(s) ===")

    # -------------------------------------------------------------
    # Step 1: Download Audio
    # -------------------------------------------------------------
    logger.info(">>> STEP 1/4: Audio Download (yt-dlp)")
    dl_results = download_audio_batch(videos=target_videos, force_redownload=force)
    logger.info(f"Download complete: {dl_results['downloaded']} downloaded, {dl_results['skipped']} cached.")

    # -------------------------------------------------------------
    # Step 2: Transcribe
    # -------------------------------------------------------------
    logger.info(">>> STEP 2/4: Whisper Transcription")
    transcribed_count = 0
    for idx, vid in enumerate(target_videos, 1):
        v_id = vid.video_id
        audio_path = settings.audio_dir / f"{v_id}.mp3"
        if not audio_path.exists():
            logger.warning(f"[{v_id}] Skipping transcription: audio file missing.")
            continue

        ok, transcript, err = transcribe_audio_file(
            audio_path=audio_path,
            video_id=v_id,
            lecture_number=vid.lecture_number,
            video_title=vid.title,
            force_retranscribe=force,
            model_size=whisper_model,
        )
        if ok and transcript:
            transcribed_count += 1
            logger.info(f"[{idx}/{total}] [{v_id}] Transcribed ({len(transcript.segments)} segments).")
        else:
            logger.error(f"[{idx}/{total}] [{v_id}] Transcription failed: {err}")

    # -------------------------------------------------------------
    # Step 3: Chunking
    # -------------------------------------------------------------
    logger.info(">>> STEP 3/4: Transcript Chunking")
    chunked_count = 0
    for idx, vid in enumerate(target_videos, 1):
        v_id = vid.video_id
        ok, chunks, err = chunk_single_video(video=vid, force_rechunk=force)
        if ok and chunks:
            chunked_count += 1
            logger.info(f"[{idx}/{total}] [{v_id}] Chunked -> {len(chunks)} chunks.")
        else:
            logger.error(f"[{idx}/{total}] [{v_id}] Chunking failed: {err}")

    # -------------------------------------------------------------
    # Step 4: Embedding & Qdrant Upload
    # -------------------------------------------------------------
    logger.info(">>> STEP 4/4: Embedding & Upload to Qdrant")
    uploaded_count = 0
    for idx, vid in enumerate(target_videos, 1):
        v_id = vid.video_id
        ok, chunks, vectors, err = embed_single_video(video_id=v_id, batch_size=batch_size)
        if not ok or vectors is None:
            logger.error(f"[{idx}/{total}] [{v_id}] Embedding failed: {err}")
            continue

        upsert_ok, upsert_err = upsert_chunks(chunks=chunks, vectors=vectors)
        if upsert_ok:
            uploaded_count += 1
            update_manifest_qdrant_status(v_id, success=True)
            logger.info(f"[{idx}/{total}] [{v_id}] Uploaded {len(chunks)} chunks to Qdrant.")
        else:
            logger.error(f"[{idx}/{total}] [{v_id}] Qdrant upload failed: {upsert_err}")
            update_manifest_qdrant_status(v_id, success=False, error=upsert_err)

    logger.info("=== Pipeline Execution Complete ===")
    get_pipeline_status()


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline Runner: Ingest Love Babbar DSA lectures end-to-end"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current pipeline status and exit",
    )
    parser.add_argument(
        "--lectures",
        type=int,
        nargs="+",
        default=None,
        help="Lecture numbers to process (e.g. --lectures 1 2 12)",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific video IDs to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process first N videos",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all videos in the playlist",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if stages were previously completed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.WHISPER_MODEL,
        help=f"Whisper model size (default: '{settings.WHISPER_MODEL}')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Sentence-transformers encoding batch size (default: 64)",
    )
    args = parser.parse_args()

    if args.status:
        get_pipeline_status()
        return

    # Load video metadata
    all_videos = load_cached_videos()
    if not all_videos:
        logger.info("No cached videos found. Fetching playlist from YouTube...")
        all_videos = fetch_playlist_metadata()

    target_videos = all_videos
    if args.lectures:
        target_set = set(args.lectures)
        target_videos = [v for v in all_videos if v.lecture_number in target_set]
    elif args.video_ids:
        target_set = set(args.video_ids)
        target_videos = [v for v in all_videos if v.video_id in target_set]
    elif args.limit:
        target_videos = target_videos[:args.limit]
    elif not args.all:
        parser.print_help()
        print("\nPlease specify --lectures, --limit, --all, or --status.")
        sys.exit(1)

    if not target_videos:
        print("No matching lectures found.")
        sys.exit(0)

    run_pipeline(
        target_videos=target_videos,
        force=args.force,
        whisper_model=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
