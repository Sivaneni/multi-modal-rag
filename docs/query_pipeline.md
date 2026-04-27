# Query Pipeline — Full Trace & Explanation

---

## Pipeline Overview

```
User query
    │
    ▼
1. EMBED QUERY
   embed_texts([query])            → 3072-dim dense vector
   compute_sparse_vectors([query]) → TF feature-hash sparse vector
    │
    ▼
2. HYBRID SEARCH — vector_store.search()
   Two parallel Prefetch branches inside Qdrant:
     branch A: dense  top-20  (cosine similarity, text_dense space)
     branch B: sparse top-20  (TF dot product,   bm25_sparse space)
   FusionQuery(RRF) merges both → returns top-k payload dicts
    │
    ▼
3. RERANK — reranker.rerank(query, candidates)
   Scores every (query, chunk) pair for TRUE relevance.
   Returns top-n with a "rerank_score" key added to each dict.
    │
    ▼
4. CONTEXT BUILDING — generate.py context loop
   Tables → summary + full markdown table (caption)
   Images → structured TYPE/CAPTION/DETAIL text
   Text   → raw chunk.text
    │
    ▼
5. GENERATION — _build_user_content() + GPT-4o
   No visual chunks → plain string message
   Any visual chunk  → multimodal content list with inline base64 images
```

---

## Stage 2 — RRF Hybrid Search (vector_store.search)

```python
results = await self._client.query_points(
    collection_name=self._collection,
    prefetch=[
        Prefetch(query=query_dense,  using="text_dense",  limit=top_k * 2),
        Prefetch(query=query_sparse, using="bm25_sparse", limit=top_k * 2),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=top_k,
    with_payload=True,
    query_filter=query_filter,
)
```

### How Prefetch works

Qdrant executes **both branches independently and in parallel**:

- **Branch A** — cosine similarity in the 3072-dim `text_dense` space → top 20 candidates
- **Branch B** — sparse dot product in `bm25_sparse` space → top 20 candidates

The two lists may overlap (same chunk in both) or not. `FusionQuery(RRF)` merges them:

```
RRF_score = 1/(rank_A + 60) + 1/(rank_B + 60)
```

Final `top_k` results are sorted by RRF score. A chunk ranked highly in both lists wins decisively.

### filter_modality

If `filter_modality="table"` is passed, Qdrant applies a payload filter **before** the Prefetch branches — only table-modality points are searched. Useful for modality-specific queries.

---

## Stage 3 — Reranker Backends

### Why rerank at all?

RRF retrieves broadly — it finds good candidates based on rank position but doesn't deeply read the content. Rerankers are **cross-encoders**: they see both the query and the document together and score their joint relevance. More expensive but significantly more accurate.

---

### OpenAI Backend (`RERANKER_BACKEND=openai`) — default

- **Model:** `gpt-4o-mini`
- Fires **one completion per chunk, all in parallel** via `asyncio.gather`

**For text chunks:**
```
prompt = "Rate relevance 1–10. Query: {q}  Document: {text[:2000]}"
→ parses integer from response, sorts descending
```

**For image chunks** (`image_base64` present):
```python
messages = [{"role": "user", "content": [
    {"type": "text",      "text": "Rate 1–10. Query: {q}\nCaption: {text}"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}]
```
GPT-4o-mini sees the **actual cropped image AND the caption** — truly multimodal scoring.

| Property | Value |
|---|---|
| Cost | ~$0.03–0.10 per call (20 candidates) |
| Latency | ~800ms–2s (parallel async) |
| Image support | Yes — inline base64 vision message |

---

### Jina Backend (`RERANKER_BACKEND=jina`)

- **Model:** `jina-reranker-m0` (native multimodal reranker)
- Sends **ALL candidates in a SINGLE API call** (unlike OpenAI which fires N calls)

```python
documents = [
    {"text": chunk.text}                          # text chunks
    {"text": chunk.text, "images": [image_b64]}   # image chunks
]
payload = {"model": "jina-reranker-m0", "query": query, "documents": documents, "top_n": top_n}
```

Returns ranked `indices` + `relevance_score` per item.

| Property | Value |
|---|---|
| Cost | ~$0.01–0.02 per call |
| Latency | ~500ms–2s |
| Image support | Yes — `images` field alongside text |

---

### BGE Backend (`RERANKER_BACKEND=bge`) — local, text-only

- **Model:** `BAAI/bge-reranker-v2-minicpm-layerwise`
- `cutoff_layers=[28]` — stops inference at layer 28 instead of all 40 → 30% faster, minimal accuracy drop
- Runs **synchronously** (blocking), offloaded to thread pool so the async event loop is not blocked:

```python
loop.run_in_executor(None, compute_score, pairs)
```

- For **image chunks**: `chunk.text` is the GPT-4o structured description (TYPE/CAPTION/DETAIL) — used as a text proxy since BGE has no vision input

| Property | Value |
|---|---|
| Cost | Free (local) |
| Latency | ~50–100ms on CPU/Apple Silicon |
| Image support | No — uses caption text as proxy |

---

### Qwen VL Backend (`RERANKER_BACKEND=qwen`) — local, multimodal

- **Model:** `Qwen3-VL-Reranker-2B` (ranked #1 on MMEB-V2)
- For image chunks: decodes base64 → PIL Image → passed to `AutoProcessor`
- Each chunk scored individually, offloaded to thread pool
- Requires ~8–12 GB RAM

| Property | Value |
|---|---|
| Cost | Free (local) |
| Latency | ~400–800ms MPS; ~1–2s CPU |
| Image support | Yes — decodes base64 to PIL Image |

---

### Backend comparison

| | OpenAI | Jina | BGE | Qwen |
|---|---|---|---|---|
| Multimodal | Yes | Yes | No (text proxy) | Yes |
| API calls | N parallel | 1 batch call | 0 (local) | 0 (local) |
| Cost | $0.03–0.10 | $0.01–0.02 | Free | Free |
| Latency | 800ms–2s | 500ms–2s | 50–100ms | 400ms–2s |
| Extra deps | None | JINA_API_KEY | FlagEmbedding | transformers + torch |

---

## Stage 4 — Context Building

### Table special treatment (`generate.py` lines 109–115)

Tables have **two representations** stored in Qdrant:

| Field | Content | Used for |
|---|---|---|
| `chunk.text` | Semantic summary | Retrieval / embedding |
| `chunk.caption` | Full markdown table (every row, every column) | Generation |

The context builder uses **both**:
```python
text = f"{summary}\n\nFull table data:\n{caption}"
```

Why? The embedding was over the summary (semantically rich for retrieval). But GPT-4o needs the **full table data** to answer factual questions — sending just the summary would lose individual cell values like `"Haemoglobin: 13.5 g/dL"`.

---

## Stage 5 — `_build_user_content()` Decision Tree

```python
visual_chunks = [c for c in candidates if c.get("image_base64")]

if not visual_chunks:
    return f"Context:\n{context}\n\nQuestion: {query}"   # plain string

else:
    return [
        {"type": "text",      "text": "Context:...\n\nQuestion: {query}"},
        {"type": "text",      "text": "[page 1] Table visual:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text",      "text": "[page 2] Image visual:"},
        {"type": "image_url", "image_url": {...}},
    ]
```

- If **ALL** top-n chunks are text-only → plain string call (cheaper, faster)
- As soon as **ONE** table/image with a crop is in top-n → full multimodal call
- Images are **interleaved** with text labels so GPT-4o knows which page each visual belongs to

---

## Prerequisites to run the debug script

1. **Qdrant running with data already ingested:**
   ```bash
   docker start qdrant
   ```

2. **`.env` settings:**
   ```
   QDRANT_URL=http://localhost:6333
   OPENAI_API_KEY=sk-...
   RERANKER_BACKEND=openai   # or jina / bge / qwen
   ```

3. **Run:**
   ```bash
   uv run python docs/query_pipeline.py
   ```

---

## What to look for when debugging

| Observation | What it means |
|---|---|
| RRF order vs rerank order differ significantly | Reranker is finding relevance RRF missed — working correctly |
| Table chunk shows `Full table data:` in Stage 4 context | Caption (full markdown) is being passed to generation |
| `Message type: MULTIMODAL` in Stage 4 | At least one chunk has an image crop → GPT-4o sees images |
| A relevant chunk missing from Stage 2 | Try lowering `TOP_K` threshold or check if it was ingested |
| Rerank score all equal (e.g. all 5.0) | Check RERANKER_BACKEND config or API key |