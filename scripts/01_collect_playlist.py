#!/usr/bin/env python3
"""
Phase 1 Ingestion Script: Playlist Metadata Extraction
Collects Love Babbar DSA C++ YouTube Playlist metadata and saves to data/manifests/videos.json.
Initializes data/manifests/pipeline_manifest.json for resilient resume and tracking.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.config import settings
from backend.ingestion.playlist import (
    fetch_playlist_metadata,
    save_videos_metadata,
    sync_pipeline_manifest,
)
from backend.utils.logging import logger


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Collect Love Babbar DSA Playlist Metadata"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=settings.PLAYLIST_URL,
        help="YouTube Playlist URL (default from .env)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch from YouTube even if cache exists"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos to store (useful for debugging/testing)"
    )
    args = parser.parse_args()

    logger.info("=== Starting Phase 1: Playlist Metadata Collection ===")
    logger.info(f"Target Playlist: {args.url}")

    videos = fetch_playlist_metadata(
        playlist_url=args.url,
        force_refresh=args.force
    )

    if args.limit and args.limit > 0:
        logger.info(f"Applying limit: taking first {args.limit} videos of {len(videos)}")
        videos = videos[:args.limit]

    # Save to data/manifests/videos.json
    save_path = save_videos_metadata(videos)

    # Synchronize pipeline manifest
    sync_pipeline_manifest(videos)

    logger.info("=== Phase 1 Summary ===")
    logger.info(f"Total Videos Extracted: {len(videos)}")
    if videos:
        logger.info(f"First Lecture: [{videos[0].video_id}] {videos[0].title} (Lec {videos[0].lecture_number})")
        logger.info(f"Last Lecture:  [{videos[-1].video_id}] {videos[-1].title} (Lec {videos[-1].lecture_number})")
    logger.info(f"Saved Metadata File: {save_path}")
    logger.info("Phase 1 completed successfully!")


if __name__ == "__main__":
    main()
