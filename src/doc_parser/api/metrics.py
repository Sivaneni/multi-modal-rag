"""Prometheus metrics for the doc-parser API.

HTTP-level metrics (request rate / latency per route) come from
prometheus-fastapi-instrumentator, wired up in ``api/app.py``. This module adds
ingest-specific metrics that the generic HTTP histogram can't express:
per-stage timings, pages, chunks and success/failure per document.

Scraped from ``GET /metrics`` (API_WORKERS=1, so the default registry is fine).
"""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import Counter, Gauge, Histogram

# Ingest of a large PDF can take minutes, so buckets go well past HTTP defaults.
_INGEST_BUCKETS = (0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1200)

INGEST_STAGE_SECONDS = Histogram(
    "doc_parser_ingest_stage_seconds",
    "Time spent per ingest stage for one document.",
    ["stage"],  # parse | chunk | enrich | embed | upsert | total
    buckets=_INGEST_BUCKETS,
)
INGEST_DOCUMENTS = Counter(
    "doc_parser_ingest_documents_total",
    "Documents ingested, by outcome.",
    ["status"],  # success | failed
)
INGEST_PAGES = Counter("doc_parser_ingest_pages_total", "Pages parsed by successful ingests.")
INGEST_CHUNKS = Counter(
    "doc_parser_ingest_chunks_total",
    "Chunks upserted by successful ingests, by modality.",
    ["modality"],
)
INGEST_IN_PROGRESS = Gauge("doc_parser_ingest_in_progress", "Ingest requests currently running.")


@contextmanager
def stage_timer(stage: str) -> Iterator[None]:
    """Observe the duration of one ingest stage (also on failure)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        INGEST_STAGE_SECONDS.labels(stage=stage).observe(time.perf_counter() - start)


@contextmanager
def track_ingest() -> Iterator[None]:
    """Wrap a whole ingest: in-progress gauge, total time, success/failed count."""
    INGEST_IN_PROGRESS.inc()
    start = time.perf_counter()
    try:
        yield
    except BaseException:
        INGEST_DOCUMENTS.labels(status="failed").inc()
        raise
    else:
        INGEST_DOCUMENTS.labels(status="success").inc()
    finally:
        INGEST_STAGE_SECONDS.labels(stage="total").observe(time.perf_counter() - start)
        INGEST_IN_PROGRESS.dec()
