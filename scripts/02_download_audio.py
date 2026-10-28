#!/usr/bin/env python3
"""
Phase 2 Ingestion Script: Audio Extraction
Downloads and converts YouTube lecture audio to 16kHz mono MP3 using yt-dlp + FFmpeg.
Saves audio to data/audio/{video_id}.mp3 and updates data/manifests/pipeline_manifest.json.
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
from backend.ingestion.downloader import download_audio_batch
from backend.utils.logging import logger


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract audio from Love Babbar DSA lectures"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to download"
    )
    parser.add_argument(
        "--lectures",
        type=int,
        nargs="+",
        default=None,
        help="Specific lecture numbers to download (e.g. --lectures 1 12 28)"
    )
    parser.add_argument(
        "--video-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific video IDs to download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if audio already exists"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Polite sleep delay in seconds between downloads (default 1.0s)"
    )
    args = parser.parse_args()

    logger.info("=== Starting Phase 2: Audio Extraction ===")
    all_videos = load_cached_videos()

    if not all_videos:
        logger.error("No videos found in cache! Run Phase 1 first: python scripts/01_collect_playlist.py")
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
        logger.info(f"Applying limit: downloading first {args.limit} video(s).")

    if not target_videos:
        logger.warning("No matching videos to download.")
        sys.exit(0)

    results = download_audio_batch(
        videos=target_videos,
        force_redownload=args.force,
        polite_delay=args.delay
    )

    logger.info("=== Phase 2 Summary ===")
    logger.info(f"Total Target Lectures: {results['total']}")
    logger.info(f"Newly Downloaded:     {results['downloaded']}")
    logger.info(f"Already Cached:       {results['skipped']}")
    logger.info(f"Failed Downloads:     {results['failed']}")
    if results['failed_ids']:
        logger.error(f"Failed Video IDs: {results['failed_ids']}")

    logger.info("Phase 2 finished successfully!")


if __name__ == "__main__":
    main()
