"""Debug script: trace each stage of the query pipeline end-to-end.

See docs/query_pipeline.md for a full explanation of every stage.

Prerequisites:
    docker start qdrant          (Qdrant must be running with data)
    .env: QDRANT_URL=http://localhost:6333, OPENAI_API_KEY=..., RERANKER_BACKEND=openai

Usage:
    uv run python docs/query_pipeline.py
"""
import asyncio
import logging

from openai import AsyncOpenAI

from doc_parser.config import get_settings
from doc_parser.ingestion.embedder import compute_sparse_vectors, get_embedder
from doc_parser.ingestion.vector_store import QdrantDocumentStore
from doc_parser.retrieval.reranker import get_reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── Config ────────────────────────────────────────────────────────────────────
QUERY  = "What is the concern in this blood test report? What are the key numbers and their values to really concern about?"  # example query to run through the pipeline
TOP_K  = 10     # candidates fetched from Qdrant before reranking
TOP_N  = 5      # final top-n after reranking passed to generation
RERANK = True   # set False to skip reranker


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sep(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _print_chunk(idx: int, chunk: dict, score_key: str | None = None) -> None:
    score = f"  score={chunk[score_key]:.4f}" if score_key and chunk.get(score_key) else ""
    print(
        f"\n  [{idx}] chunk_id={chunk.get('chunk_id', '?')}"
        f"  page={chunk.get('page', '?')}"
        f"  modality={chunk.get('modality', '?')}"
        f"  atomic={chunk.get('is_atomic', '?')}"
        f"{score}"
    )
    print(f"      text   : {(chunk.get('text') or '')[:300]!r}")
    caption = (chunk.get("caption") or "")[:200]
    if caption:
        print(f"      caption: {caption!r}")
    print(f"      has_image_base64: {bool(chunk.get('image_base64'))}")


# ── Pipeline stages ───────────────────────────────────────────────────────────
async def main() -> None:
    settings      = get_settings()
    openai_client = AsyncOpenAI()

    # Stage 1 — Embed query
    _sep("STAGE 1 — Embed Query")
    print(f"  Query: {QUERY!r}")
    embedder     = get_embedder(settings)
    query_dense  = (await embedder.embed([QUERY]))[0]
    query_sparse = compute_sparse_vectors([QUERY])[0]
    print(f"  Dense  : {len(query_dense)}-dim | first 5: {query_dense[:5]}")
    print(f"  Sparse : {len(query_sparse.indices)} non-zero buckets "
          f"| top 5 indices: {query_sparse.indices[:5]} "
          f"| top 5 values: {[round(v, 4) for v in query_sparse.values[:5]]}")

    # Stage 2 — Hybrid search (RRF)
    _sep("STAGE 2 — Hybrid Search (RRF)")
    print(f"  top_k={TOP_K}  |  dense top {TOP_K*2} + sparse top {TOP_K*2}  →  RRF merge  →  top {TOP_K}")
    store      = QdrantDocumentStore(settings)
    candidates = await store.search(
        query_text=QUERY,
        embedder=embedder,
        settings=settings,
        top_k=TOP_K,
    )
    print(f"\n  {len(candidates)} candidates returned (RRF order — best first):")
    for i, c in enumerate(candidates):
        _print_chunk(i, c)

    # Stage 3 — Rerank
    if RERANK and candidates:
        _sep(f"STAGE 3 — Rerank  (backend={settings.reranker_backend})")
        print(f"  Scoring {len(candidates)} candidates  →  keeping top {TOP_N}")
        reranker = get_reranker(settings)
        reranked = await reranker.rerank(QUERY, candidates, top_n=TOP_N)
        print(f"\n  Top {TOP_N} after reranking (rerank_score order — best first):")
        for i, c in enumerate(reranked):
            _print_chunk(i, c, score_key="rerank_score")
        final_candidates = reranked
    else:
        _sep("STAGE 3 — Rerank SKIPPED")
        final_candidates = candidates[:TOP_N]
        for c in final_candidates:
            c.setdefault("rerank_score", None)

    # Stage 4 — Build context
    _sep("STAGE 4 — Context Building")
    context_parts: list[str] = []
    for c in final_candidates:
        page     = c.get("page", "?")
        modality = c.get("modality", "text")
        if modality == "table":
            caption = c.get("caption") or ""
            summary = c.get("text") or ""
            text = f"{summary}\n\nFull table data:\n{caption}" if caption and summary else caption or summary
        else:
            text = c.get("text", "") or c.get("caption") or ""
        context_parts.append(f"[page {page}] {text}")

    context       = "\n\n".join(context_parts)
    visual_chunks = [c for c in final_candidates if c.get("image_base64")]
    print(f"  Chunks in context : {len(final_candidates)}")
    print(f"  Visual chunks     : {len(visual_chunks)}")
    print(f"  Message type      : {'MULTIMODAL (list)' if visual_chunks else 'TEXT-ONLY (string)'}")
    print(f"\n  Context preview (first 800 chars):\n\n  {context[:800]!r}")

    # Stage 5 — Generate
    _sep("STAGE 5 — Generation (GPT-4o)")
    system_prompt = (
        "You are a precise document assistant. Answer the question using ONLY the provided context. "
        "If the answer is not in the context, say \"I don't have enough information to answer this.\" "
        "Cite the source page numbers when possible."
    )

    if not visual_chunks:
        user_content: str | list = f"Context:\n{context}\n\nQuestion: {QUERY}"
        print("  Sending: text-only message")
    else:
        user_content = [
            {"type": "text", "text": f"Context:\n{context}\n\nQuestion: {QUERY}"},
        ]
        for c in visual_chunks:
            page     = c.get("page", "?")
            modality = c.get("modality", "visual")
            user_content.append({"type": "text", "text": f"[page {page}] {modality.capitalize()} visual:"})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{c['image_base64']}"}})
        print(f"  Sending: multimodal message with {len(visual_chunks)} inline image(s)")

    completion = await openai_client.chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},  # type: ignore[list-item]
        ],
        max_tokens=1024,
        temperature=0.0,
    )
    answer = completion.choices[0].message.content or ""

    # Final answer + summary
    _sep("FINAL ANSWER")
    print(f"\n  {answer}\n")

    _sep("PIPELINE SUMMARY")
    print(f"  Query          : {QUERY!r}")
    print(f"  RRF candidates : {len(candidates)}")
    print(f"  After rerank   : {len(final_candidates)}")
    print(f"  Visual chunks  : {len(visual_chunks)}")
    print(f"  Message type   : {'multimodal' if visual_chunks else 'text-only'}")
    print(f"  Model          : {settings.openai_llm_model}")
    print(f"  Reranker       : {settings.reranker_backend if RERANK else 'skipped'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())