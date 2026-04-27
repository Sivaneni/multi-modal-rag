import asyncio
import logging
import dataclasses
import json
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
from pathlib import Path
from openai import AsyncOpenAI
from qdrant_client.models import FieldCondition, Filter, MatchValue
from doc_parser.pipeline import DocumentParser
from doc_parser.chunker import document_aware_chunking
from doc_parser.ingestion.image_captioner import enrich_chunks
from doc_parser.ingestion.embedder import get_embedder, embed_chunks
from doc_parser.ingestion.vector_store import QdrantDocumentStore
from doc_parser.config import get_settings
PDF_PATH   = Path('data/Kubernetes Crash Recovery Guide_V1.0.pdf')
SOURCE_FILE = 'Kubernetes Crash Recovery Guide_V1.0.pdf'

async def already_ingested(store: QdrantDocumentStore, source_file: str) -> bool:
    # Count points in Qdrant that have this source_file in their payload
    try:
        response = await store._client.count(
            collection_name=store._collection,
            count_filter=Filter(
                must=[FieldCondition(key='source_file', match=MatchValue(value=source_file))]
            ),
            exact=True,
        )
        return response.count > 0
    except Exception:
        return False  # collection does not exist yet

async def main():
    settings = get_settings()
    store = QdrantDocumentStore(settings)
    await store.create_collection(overwrite=False)   # no-op if already exists

    # ── Guard: skip full pipeline if already ingested ─────────────────────────
    if await already_ingested(store, SOURCE_FILE):
        print(f'[SKIP] {SOURCE_FILE} is already in Qdrant — skipping ingestion.')
        print('       Delete the collection or use overwrite=True to re-ingest.')
        print()
    else:
        # ── 1. Parse ──────────────────────────────────────────────────────────
        parser = DocumentParser()
        result = parser.parse_file(PDF_PATH)
        print(f'Parsed: {len(result.pages)} pages, {result.total_elements} elements')

        # ── 2. Chunk ──────────────────────────────────────────────────────────
        pages = [(p.page_num, p.elements) for p in result.pages]
        chunks = document_aware_chunking(pages, source_file=SOURCE_FILE)
        print(f'Chunks: {len(chunks)} total')

        # ── 3. Enrich (GPT-4o captions for tables/images/formulas) ───────────
        openai_client = AsyncOpenAI()
        chunks = await enrich_chunks(chunks, pdf_path=PDF_PATH, client=openai_client, model='gpt-4o')
        print('Enrichment done.')

        # ── 4. Embed (dense + sparse) ─────────────────────────────────────────
        embedder = get_embedder(settings)
        dense_embeddings, sparse_vectors = await embed_chunks(chunks, embedder, settings)
        print(f'Embeddings: {len(dense_embeddings)} dense ({len(dense_embeddings[0])}-dim)')

        # ── 5. Upsert into Qdrant ─────────────────────────────────────────────
        total = await store.upsert_chunks(chunks, dense_embeddings, sparse_vectors)
        print(f'Upserted {total} points to collection: {settings.qdrant_collection_name}')
        print()

    # ── 6. Sanity check: count points in collection ───────────────────────────
    count_resp = await store._client.count(
        collection_name=store._collection,
        count_filter=Filter(
            must=[FieldCondition(key='source_file', match=MatchValue(value=SOURCE_FILE))]
        ),
        exact=True,
    )
    print(f'Points in Qdrant for {SOURCE_FILE}: {count_resp.count}')

asyncio.run(main())