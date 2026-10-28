#!/usr/bin/env python3
"""
Server Entrypoint: Start the Love Babbar DSA RAG FastAPI server.

Usage:
  python run_server.py
  python run_server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from backend.config import settings


def main():
    parser = argparse.ArgumentParser(description="Start the Love Babbar DSA RAG API server")
    parser.add_argument("--host", default=settings.BACKEND_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.BACKEND_PORT, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    print(f"Starting RAG API at http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(
        "backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
