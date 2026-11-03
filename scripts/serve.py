"""Launch the doc-parser FastAPI server with uvicorn."""
from __future__ import annotations

import argparse
import faulthandler
import sys

# Print a Python traceback (all threads) on native crashes like segfaults.
faulthandler.enable(all_threads=True)

# ── Windows CPU-torch stability (layout model runs in a GLM-OCR worker thread) ──
# PP-DocLayoutV3 segfaulted (access violation in mask_to_box_coordinate) when
# torch's OpenMP/MKL thread pool ran inside a background Python thread.
# Must be set BEFORE torch is imported. Override via env if needed.
import os
import threading

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Bigger stacks for threads created from here on (GLM-OCR pipeline workers).
threading.stack_size(64 * 1024 * 1024)
from pathlib import Path

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn

from doc_parser.config import get_settings


def main() -> None:
    """Parse CLI args and start uvicorn."""
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Start the doc-parser RAG API server.")
    parser.add_argument("--host", default=settings.api_host, help="Bind host (default: %(default)s)")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Bind port (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=settings.api_workers, help="Number of worker processes (default: %(default)s)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    args = parser.parse_args()

    uvicorn.run(
        "doc_parser.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_config=None,  # Disable uvicorn's own logging; loguru handles it
    )


if __name__ == "__main__":
    main()
