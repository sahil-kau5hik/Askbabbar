#!/usr/bin/env python3
"""
Phase 4 Ingestion Script: Timestamp-Preserving Chunker
Splits transcripts into ~60-second semantic windows while preserving exact Whisper timestamps.
Saves chunks to data/chunks/{video_id}.json and updates data/manifests/pipeline_manifest.json.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.ingestion.playlist import load_cached_videos
from backend.ingestion.chunker import chunk_batch
from backend.utils.logging import logger


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Create 60-second timestamp-preserving chunks from transcripts"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to chunk"
    )
    parser.add_argument(
        "--lectures",
        type=int,
        nargs="+",
        default=None,
        help="Specific lecture numbers to chunk (e.g. --lectures 12)"
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific video IDs to chunk"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=float(settings.CHUNK_SIZE_SECONDS),
        help=f"Target chunk duration in seconds (default: {settings.CHUNK_SIZE_SECONDS}s)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-chunking even if chunks already exist"
    )
    args = parser.parse_args()

    logger.info("=== Starting Phase 4: Transcript Chunking ===")
    all_videos = load_cached_videos()

    if not all_videos:
        logger.error("No videos found in cache! Run Phase 1 first.")
        sys.exit(1)

    target_videos = all_videos

    if args.lectures:
        target_set = set(args.lectures)
        target_videos = [v for v in all_videos if v.lecture_number in target_set]
        logger.info(f"Filtered by lecture numbers {args.lectures}: {len(target_videos)} video(s) found.")

    elif args.video_ids:
        target_set = set(args.video_ids)
        target_videos = [v for v in all_videos if v.video_id in target_set]
        logger.info(f"Filtered by video IDs {args.video_ids}: {len(target_videos)} video(s) found.")

    if args.limit and args.limit > 0:
        target_videos = target_videos[:args.limit]
        logger.info(f"Applying limit: chunking first {args.limit} video(s).")

    # Filter to only videos that have an existing transcript
    available_videos = [v for v in target_videos if (settings.transcripts_dir / f"{v.video_id}.json").exists()]
    logger.info(f"Videos with transcripts available for chunking: {len(available_videos)}/{len(target_videos)}")
    target_videos = available_videos

    if not target_videos:
        logger.warning("No matching videos found with transcripts to chunk.")
        sys.exit(0)

    results = chunk_batch(
        videos=target_videos,
        force_rechunk=args.force,
        target_duration=args.duration
    )

    logger.info("=== Phase 4 Summary ===")
    logger.info(f"Total Target Lectures: {results['total']}")
    logger.info(f"Newly Chunked:        {results['chunked']}")
    logger.info(f"Total Chunks Created: {results['total_chunks_created']}")
    logger.info(f"Already Cached:       {results['skipped']}")
    logger.info(f"Failed Chunking:      {results['failed']}")
    if results['failed_ids']:
        logger.error(f"Failed Video IDs: {results['failed_ids']}")

    logger.info("Phase 4 completed successfully!")


if __name__ == "__main__":
    main()
