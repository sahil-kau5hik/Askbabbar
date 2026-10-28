#!/usr/bin/env python3
"""
Phase 5 Ingestion Script: Embed Chunks & Upload to Qdrant
Reads chunk JSON files from data/chunks/, embeds with sentence-transformers,
and upserts into the configured Qdrant collection.
Updates data/manifests/pipeline_manifest.json with `embedded` and `uploaded_to_qdrant` flags.

Usage:
  python scripts/05_embed_and_upload.py
  python scripts/05_embed_and_upload.py --limit 5
  python scripts/05_embed_and_upload.py --lectures 1 2 3
  python scripts/05_embed_and_upload.py --video-ids ABC123 XYZ456
  python scripts/05_embed_and_upload.py --force
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.embeddings.embedder import embed_single_video
from backend.embeddings.vector_store import (
    delete_video_points,
    get_collection_info,
    upsert_chunks,
)
from backend.embeddings.vector_store import update_manifest_qdrant_status
from backend.ingestion.playlist import load_cached_videos
from backend.utils.logging import logger


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Embed transcript chunks and upload to Qdrant"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to process",
    )
    parser.add_argument(
        "--lectures",
        type=int,
        nargs="+",
        default=None,
        help="Specific lecture numbers to embed (e.g. --lectures 1 2 3)",
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific video IDs to embed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-embedding even if already uploaded",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size for sentence-transformers (default: 64)",
    )
    args = parser.parse_args()

    logger.info("=== Starting Phase 5: Embed & Upload to Qdrant ===")
    logger.info(f"Qdrant collection: {settings.QDRANT_COLLECTION}")
    logger.info(f"Embedding model:   {settings.EMBEDDING_MODEL}")

    all_videos = load_cached_videos()
    if not all_videos:
        logger.error("No videos found in cache! Run Phase 1 first.")
        sys.exit(1)

    target_videos = all_videos

    if args.lectures:
        target_set = set(args.lectures)
        target_videos = [v for v in all_videos if v.lecture_number in target_set]
        logger.info(f"Filtered by lectures {args.lectures}: {len(target_videos)} video(s).")

    elif args.video_ids:
        target_set = set(args.video_ids)
        target_videos = [v for v in all_videos if v.video_id in target_set]
        logger.info(f"Filtered by video IDs {args.video_ids}: {len(target_videos)} video(s).")

    if args.limit and args.limit > 0:
        target_videos = target_videos[: args.limit]
        logger.info(f"Applying limit: processing first {args.limit} video(s).")

    # Filter to only videos that have an existing chunks file
    available_videos = [v for v in target_videos if (settings.chunks_dir / f"{v.video_id}.json").exists()]
    logger.info(f"Videos with chunks available for embedding: {len(available_videos)}/{len(target_videos)}")
    target_videos = available_videos

    if not target_videos:
        logger.warning("No matching videos found with chunks to embed.")
        sys.exit(0)

    total = len(target_videos)
    embedded_count = 0
    uploaded_count = 0
    skipped_count = 0
    failed_embed = 0
    failed_upload = 0
    failed_ids = []

    manifest = {}
    if settings.pipeline_manifest_path.exists():
        try:
            with open(settings.pipeline_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    for idx, video in enumerate(target_videos, start=1):
        v_id = video.video_id

        if not args.force and manifest.get(v_id, {}).get("uploaded_to_qdrant"):
            skipped_count += 1
            continue

        logger.info(f"Processing ({idx}/{total}): [{v_id}] {video.title}")

        # --- Embed ---
        success, chunks, vectors, err = embed_single_video(
            video_id=v_id,
            batch_size=args.batch_size,
        )
        if not success or vectors is None:
            failed_embed += 1
            failed_ids.append({"video_id": v_id, "stage": "embed", "error": err})
            continue
        embedded_count += 1

        # --- Upload to Qdrant ---
        if args.force:
            delete_video_points(v_id)

        ok, upload_err = upsert_chunks(chunks=chunks, vectors=vectors)
        if ok:
            uploaded_count += 1
            update_manifest_qdrant_status(v_id, success=True)
            logger.info(f"[{v_id}] Successfully uploaded {len(chunks)} chunks to Qdrant.")
        else:
            failed_upload += 1
            failed_ids.append({"video_id": v_id, "stage": "upload", "error": upload_err})
            update_manifest_qdrant_status(v_id, success=False, error=upload_err)

    # --- Final summary ---
    logger.info("=== Phase 5 Summary ===")
    logger.info(f"Total target videos:  {total}")
    logger.info(f"Successfully embedded: {embedded_count}")
    logger.info(f"Uploaded to Qdrant:   {uploaded_count}")
    logger.info(f"Already Indexed:      {skipped_count}")
    logger.info(f"Failed (embed):       {failed_embed}")
    logger.info(f"Failed (upload):      {failed_upload}")

    if failed_ids:
        logger.error(f"Failed entries: {failed_ids}")

    # Print collection stats
    info = get_collection_info()
    if info:
        logger.info(f"Qdrant collection stats: {info}")

    logger.info("Phase 5 completed!")


if __name__ == "__main__":
    main()
