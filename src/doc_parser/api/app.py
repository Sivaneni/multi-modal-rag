"""FastAPI app factory with lifespan and middleware."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from doc_parser.api.middleware import LoggingMiddleware
from doc_parser.api.routes.generate import router as generate_router
from doc_parser.api.routes.health import router as health_router
from doc_parser.api.routes.ingest import router as ingest_router
from doc_parser.api.routes.search import router as search_router
from doc_parser.config import get_settings
from doc_parser.logging_config import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Configure logging and report startup/shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_json)
    logger.info(
        "Starting doc-parser API | parser={} | backend={} | collection={}",
        settings.parser_backend,
        settings.reranker_backend,
        settings.qdrant_collection_name,
    )
    yield
    logger.info("Shutting down doc-parser API")


def create_app() -> FastAPI:
    """Construct and return the FastAPI application."""
    app = FastAPI(
        title="doc-parser RAG API",
        description="Multimodal RAG pipeline: PDF ingestion, hybrid search, and reranking.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(LoggingMiddleware)
    app.include_router(health_router, tags=["health"])
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(search_router, prefix="/search", tags=["search"])
    app.include_router(generate_router, prefix="/generate", tags=["generate"])

    # Prometheus: per-route request rate/latency + ingest metrics (api/metrics.py)
    # on GET /metrics. Buckets extended to minutes because ingest is slow.
    Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(
        app,
        latency_lowr_buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
    ).expose(app, include_in_schema=False)
    return app


app = create_app()
