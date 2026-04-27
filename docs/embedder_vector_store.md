# Embedder & Vector Store — Deep Dive

---

## 1. `embedder.py` — Dense + Sparse Encoding

### Why two vector types?

| Type | What it captures | Used for |
|---|---|---|
| **Dense** (`text_dense`) | Semantic meaning — synonyms, paraphrases, context | "What does this table show?" |
| **Sparse** (`bm25_sparse`) | Exact keywords — acronyms, lab test names, numbers | "WBC 7200", "Haemoglobin" |

Running both and fusing results with RRF (Reciprocal Rank Fusion) gives better retrieval than either alone.

---

### `compute_sparse_vectors()` (lines 76–122)

Converts raw text into a BM25-style sparse vector using the **feature hashing trick** — no pre-built vocabulary needed.

```python
def compute_sparse_vectors(texts: list[str], n_features: int = 2**17) -> list[SparseVector]:
```

**Step-by-step for one text:**

```
Input: "Haemoglobin 13.5 g/dL 12-17 Normal"

Step 1 — Tokenise (lowercase alphanumeric only):
  ["haemoglobin", "13", "5", "g", "dl", "12", "17", "normal"]

Step 2 — Count term frequencies:
  Counter({"haemoglobin": 1, "13": 1, "normal": 1, ...})

Step 3 — Hash each term to a bucket (0 to 131071):
  hash("haemoglobin") % 131072 → bucket 84321
  hash("normal")      % 131072 → bucket 12044
  ...

Step 4 — Normalise by total token count (TF normalisation):
  bucket 84321 → weight = 1 / 8 = 0.125
  bucket 12044 → weight = 1 / 8 = 0.125

Step 5 — Sort by bucket index (Qdrant requires sorted sparse vectors):
  SparseVector(indices=[12044, 84321, ...], values=[0.125, 0.125, ...])
```

**Why `2^17 = 131072` buckets?**
Large enough that hash collisions (two different words landing in the same bucket) are rare, small enough that Qdrant handles it efficiently.

**Collision handling:** last-write wins — rare and acceptable for a TF-proxy.

---

### `embed_texts()` (lines 37–73)

Calls the OpenAI embeddings API in batches of 100 (well within the 2048-input limit):

```python
async def embed_texts(texts, client, model="text-embedding-3-large", dimensions=3072, batch_size=100):
```

- Empty strings are replaced with `"[empty]"` because the API rejects blank inputs
- Returns vectors in the **same order** as the input list (API guarantees this)
- `dimensions=3072` is the full size of `text-embedding-3-large`; the SDK supports truncation via the `dimensions` parameter

---

### `embed_chunks()` (lines 196–218)

The main entry point that wires everything together:

```python
async def embed_chunks(chunks, embedder, settings) -> tuple[list[list[float]], list[SparseVector]]:
    texts = [c.text for c in chunks]      # uses chunk.text (enriched by image_captioner)
    dense  = await embedder.embed(texts)   # async OpenAI/Gemini API calls
    sparse = compute_sparse_vectors(texts) # synchronous, CPU-only
    return dense, sparse
```

Key point: it reads `chunk.text` — which after `enrich_chunks()` is already the **enriched semantic description**, not raw OCR. So the embedding reflects the GPT-4o enriched content.

---

### Embedder class hierarchy

```
BaseEmbedder (ABC)
  ├─ OpenAIEmbedder   → AsyncOpenAI client → text-embedding-3-large (3072-dim)
  └─ GeminiEmbedder   → google-genai client → gemini-embedding-2-preview
                        (runs sync SDK in executor to avoid blocking event loop)

get_embedder(settings) → reads EMBEDDING_PROVIDER from .env → returns correct instance
```

---

## 2. `vector_store.py` — How Chunks Land in Qdrant

### Collection structure

Qdrant collection `"documents"` has **two named vector spaces**:

```
"text_dense"  → VectorParams(size=3072, distance=COSINE, hnsw_config={m=16, ef_construct=100})
"bm25_sparse" → SparseVectorParams(index=SparseIndexParams(on_disk=False))
```

- `COSINE` distance for dense (semantic similarity)
- HNSW index parameters: `m=16` (graph connectivity), `ef_construct=100` (index build quality)
- Sparse stored in RAM (`on_disk=False`) for fast keyword lookup

---

### `upsert_chunks()` (lines 108–165)

```python
async def upsert_chunks(chunks, dense_embeddings, sparse_vectors, batch_size=64) -> int:
```

**What one Qdrant `PointStruct` looks like** for a table chunk from the blood test PDF:

```python
PointStruct(
    id = "550e8400-e29b-41d4-a716-446655440000",   # UUID5 from chunk_id — deterministic, stable
    vector = {
        "text_dense":  [0.021, -0.043, 0.018, ...],   # 3072-dim float vector
        "bm25_sparse": SparseVector(
            indices=[12044, 84321, 99201, ...],         # hash buckets
            values= [0.125,  0.125,  0.083, ...]        # normalised TF weights
        )
    },
    payload = {
        "text":          "This table shows a complete blood count...",  # enriched summary (for embedding)
        "chunk_id":      "sample_blood_urine_test.pdf_1_1",
        "source_file":   "sample_blood_urine_test.pdf",
        "page":          1,
        "element_types": ["table"],
        "bbox":          [46.0, 205.0, 952.0, 380.0],   # None for text chunks
        "is_atomic":     True,
        "modality":      "table",
        "image_base64":  "iVBORw0KGgoAAAANSUhEUgAA...",  # PNG crop of the table region
        "caption":       "| Test | Result | Unit |...",  # full markdown table (for generation LLM)
    }
)
```

**Why UUID5 for the ID?**
`uuid.uuid5(NAMESPACE_DNS, chunk_id)` is deterministic — re-ingesting the same file with the same chunk produces the same UUID, so `upsert` **overwrites** rather than duplicates.

**Batching (lines 154–162):**
Points are upserted 64 at a time to stay within Qdrant's recommended request size.

---

### `search()` — Hybrid retrieval (lines 167–212)

```python
results = await client.query_points(
    prefetch=[
        Prefetch(query=query_dense,  using="text_dense",  limit=top_k * 2),  # semantic candidates
        Prefetch(query=query_sparse, using="bm25_sparse", limit=top_k * 2),  # keyword candidates
    ],
    query=FusionQuery(fusion=Fusion.RRF),   # merge both ranked lists
    limit=top_k,
)
```

Two separate ranked lists (dense top-20, sparse top-20) are merged using **RRF** (Reciprocal Rank Fusion):
`score = 1/(rank_dense + 60) + 1/(rank_sparse + 60)`

Results returned as a list of payload dicts — the raw `chunk.text`, `caption`, `image_base64`, etc.

---

## Complete ingestion pipeline position

```
DocumentParser.parse_file()           ← OCR (Ollama / GLM-OCR)
    └─ ParseResult
document_aware_chunking(pages)        ← structure-aware splitting
    └─ list[Chunk]  (raw OCR text)
enrich_chunks(chunks, pdf_path)       ← GPT-4o multimodal enrichment
    └─ list[Chunk]  (enriched text + captions + base64)
embed_chunks(chunks, embedder)        ← THIS: dense (OpenAI) + sparse (feature hash)
    └─ (dense_embeddings, sparse_vectors)
upsert_chunks(chunks, dense, sparse)  ← THIS: write to Qdrant
    └─ N points in collection "documents"
```

---

## Step-by-step: Run Qdrant locally with Docker (persistent volume)

### Step 1 — Start Qdrant with a named volume

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

| Flag | Purpose |
|---|---|
| `-d` | Run in background |
| `-p 6333:6333` | REST API (used by the Python client) |
| `-p 6334:6334` | gRPC (optional) |
| `-v qdrant_data:/qdrant/storage` | **Named Docker volume** — all collection data, vectors, and payloads survive container restarts |

The named volume `qdrant_data` is managed by Docker and persists on disk at:
`C:\Users\<you>\AppData\Local\Docker\volumes\qdrant_data\_data` (Windows)

**To stop and restart without losing data:**
```bash
docker stop qdrant
docker start qdrant      # data is still there
```

**To wipe all data and start fresh:**
```bash
docker stop qdrant && docker rm qdrant
docker volume rm qdrant_data
# then re-run the docker run command above
```

### Step 2 — Verify it's running

```bash
curl http://localhost:6333/collections
# Expected: {"result":{"collections":[]},"status":"ok","time":...}
```

Or open `http://localhost:6333/dashboard` in your browser for the Qdrant Web UI.

### Step 3 — Update `.env` for local testing

The `.env` currently has `QDRANT_URL=http://qdrant:6333` (the Docker Compose service name).
For local testing change it to:

```
QDRANT_URL=http://localhost:6333
```

### Step 4 — Run the script below (it skips ingestion if already done)

---

## Full end-to-end test script — with ingestion skip on re-run

The script checks Qdrant first. If the PDF has already been ingested (points exist for that `source_file`), it skips the entire Parse → Chunk → Enrich → Embed → Upsert pipeline and goes straight to printing results. This means you only pay the OCR + GPT-4o cost once.

```bash
uv run python -c "
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

PDF_PATH   = Path('data/sample_blood_urine_test.pdf')
SOURCE_FILE = 'sample_blood_urine_test.pdf'

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
"
```

### What each step does

| Step | Input | Output | Skipped on re-run? |
|---|---|---|---|
| Ingestion check | `source_file` filter in Qdrant | `count > 0` → skip | — |
| Parse | PDF file | `ParseResult` | Yes |
| Chunk | `ParseResult.pages` | `list[Chunk]` raw OCR | Yes |
| Enrich | `list[Chunk]` + PDF | Enriched text + captions | Yes |
| Embed | `list[Chunk]` | Dense + sparse vectors | Yes |
| Upsert | Vectors + payloads | Points in Qdrant | Yes |
| Count check | Qdrant filter | Points count printed | **Always runs** |

### Quick sanity checks

```bash
# How many points are in the collection?
curl http://localhost:6333/collections/documents

# Browse points visually
open http://localhost:6333/dashboard   # Collections → documents → Points
```

### Force re-ingestion (e.g. after changing enrichment prompts)

Change `overwrite=False` → `overwrite=True` in `create_collection()` — this drops and recreates the collection, then the ingestion check finds 0 points and runs the full pipeline again.

---

## RRF (Reciprocal Rank Fusion) — Deep Dive

### What problem does RRF solve?

After a hybrid search, you have **two independent ranked lists**:

- `text_dense` results — ordered by cosine similarity of embeddings (semantic)
- `bm25_sparse` results — ordered by keyword overlap (lexical)

These lists use different scoring scales that cannot be directly compared or averaged:
- Dense score: `0.87` (cosine similarity, range 0–1)
- Sparse score: `0.043` (TF-weighted dot product, range 0–∞)

You can't just average them — they mean different things. RRF solves this by **ignoring the scores entirely** and only using **rank positions**.

---

### The RRF formula

```
RRF_score(chunk) = 1 / (rank_dense + k) + 1 / (rank_sparse + k)
```

Where `k = 60` (Qdrant's default constant — prevents top-ranked items from dominating too heavily).

The final list is sorted by `RRF_score` descending.

---

### Concrete example — blood test PDF query

**Query:** `"What is the patient's haemoglobin level?"`

After embedding and searching, Qdrant returns two ranked lists of top-5 candidates:

**Dense results** (semantic similarity — understands "haemoglobin level" ≈ "blood count values"):

| Rank | Chunk | Content preview |
|---|---|---|
| 1 | chunk_1_1 | Table: CBC report with all test values |
| 2 | chunk_1_2 | "Complete blood count results for patient John..." |
| 3 | chunk_1_0 | "CITY DIAGNOSTICS LAB — Blood & Urine Report" |
| 4 | chunk_2_0 | "Urine analysis: colour, clarity, pH..." |
| 5 | chunk_1_3 | "Reference ranges are based on adult norms..." |

**Sparse results** (keyword match — finds exact word "haemoglobin"):

| Rank | Chunk | Content preview |
|---|---|---|
| 1 | chunk_1_1 | Table: CBC report with all test values |
| 2 | chunk_1_3 | "Reference ranges are based on adult norms..." |
| 3 | chunk_2_0 | "Urine analysis: colour, clarity, pH..." |
| 4 | chunk_1_2 | "Complete blood count results for patient John..." |
| 5 | chunk_1_0 | "CITY DIAGNOSTICS LAB — Blood & Urine Report" |

**RRF score calculation** (k=60):

| Chunk | Dense rank | Sparse rank | RRF score |
|---|---|---|---|
| chunk_1_1 | 1 | 1 | `1/(1+60) + 1/(1+60)` = **0.03279** |
| chunk_1_2 | 2 | 4 | `1/(2+60) + 1/(4+60)` = 0.01613 + 0.01563 = **0.03176** |
| chunk_1_3 | 5 | 2 | `1/(5+60) + 1/(2+60)` = 0.01538 + 0.01613 = **0.03151** |
| chunk_1_0 | 3 | 5 | `1/(3+60) + 1/(5+60)` = 0.01587 + 0.01538 = **0.03125** |
| chunk_2_0 | 4 | 3 | `1/(4+60) + 1/(3+60)` = 0.01563 + 0.01587 = **0.03150** |

**Final merged ranking after RRF:**

```
1. chunk_1_1  (0.03279) ← table chunk, ranked #1 in BOTH lists → wins decisively
2. chunk_1_2  (0.03176) ← text chunk, top in dense but low in sparse
3. chunk_1_3  (0.03151) ← reference ranges chunk, top in sparse but low in dense
4. chunk_2_0  (0.03150) ← urine chunk, middle in both
5. chunk_1_0  (0.03125) ← lab header, low in both
```

---

### Why k=60 matters

`k` controls how much the top positions dominate. Imagine two extreme cases:

**k=0 (no smoothing):**
```
rank 1 → 1/1 = 1.000
rank 2 → 1/2 = 0.500   ← rank 1 is 2× better than rank 2
rank 10 → 1/10 = 0.100
```
A single #1 ranking dominates everything. A chunk ranked #1 in dense but #10 in sparse gets a very high score even if the sparse signal is weak.

**k=60 (Qdrant default):**
```
rank 1  → 1/61  = 0.01639
rank 2  → 1/62  = 0.01613   ← rank 1 is barely better than rank 2
rank 10 → 1/70  = 0.01429
```
Ranks are compressed — the difference between rank 1 and rank 10 is small. This means **consistent mid-ranking in both lists beats a single #1 with no presence in the other list**.

This is important for hybrid search: a chunk that ranks #3 in dense AND #3 in sparse is more trustworthy than one ranked #1 in dense but absent from sparse entirely.

---

### When dense wins vs sparse wins

| Query type | Winner | Why |
|---|---|---|
| `"what are the patient's kidney function results"` | Dense | "kidney function" → semantically matches "Creatinine, BUN, eGFR" even without exact words |
| `"Haemoglobin 13.5 g/dL"` | Sparse | Exact lab values + units are rare tokens, dense embeddings may not distinguish them |
| `"blood test normal ranges"` | Both tied | Common medical phrase, both vectors handle it well |
| `"WBC 7200"` | Sparse | Specific number + acronym — dense may retrieve other numeric results |

RRF naturally handles all these cases without you needing to tune a weight between the two — it lets each list contribute proportionally based on where candidates appear in both rankings.
