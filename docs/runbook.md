# Hybrid GraphRAG — Operational Runbook

> Branch: `hybrid-rag-deployment`  
> Last updated: 2026-05-03

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Starting External Services](#3-starting-external-services)
4. [Starting the API Server](#4-starting-the-api-server)
5. [Starting the Streamlit UI](#5-starting-the-streamlit-ui)
6. [Ingesting Documents](#6-ingesting-documents)
7. [Verifying Ingestion](#7-verifying-ingestion)
8. [Asking Questions](#8-asking-questions)
9. [Switching LLM Provider](#9-switching-llm-provider)
10. [Troubleshooting](#10-troubleshooting)
11. [Service Summary](#11-service-summary)

---

## 1. Prerequisites

| Requirement | Version / Notes |
|---|---|
| Python | 3.12 (managed by `uv`) |
| uv | `pip install uv` |
| Qdrant | Running locally on port `6333` |
| Neo4j | Running locally on port `7687` (optional, enables graph features) |
| Ollama | Running locally on port `11434` (if `PARSER_BACKEND=ollama`) |
| OpenAI API key | Required for embeddings always |
| MeshAPI key | Required when `LLM_PROVIDER=meshapi` |

---

## 2. Environment Setup

Copy `.env` and fill in your keys. All settings are read from `.env`; values there override system environment variables.

```env
# ── Parser ─────────────────────────────────────────────────────────
PARSER_BACKEND=ollama          # "cloud" (Z.AI) | "ollama" (local)
Z_AI_API_KEY=                  # only needed when PARSER_BACKEND=cloud

# ── LLM provider ───────────────────────────────────────────────────
LLM_PROVIDER=meshapi           # "meshapi" (default) | "openai"
MESH_API_KEY=rsk_...           # MeshAPI key

# ── OpenAI (embeddings only when LLM_PROVIDER=meshapi) ─────────────
OPENAI_API_KEY=sk-proj-...
OPENAI_LLM_MODEL=openai/gpt-4o-mini

# ── Embedding ──────────────────────────────────────────────────────
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# ── Qdrant ─────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=documentss

# ── Neo4j (comment out to disable graph features) ──────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# ── Reranker ───────────────────────────────────────────────────────
RERANKER_BACKEND=jina
JINA_API_KEY=jina_...

# ── API server ─────────────────────────────────────────────────────
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
```

> **Important:** `EMBEDDING_DIMENSIONS` cannot be changed after a collection is created. To change it, re-ingest with `--overwrite`.

---

## 3. Starting External Services

### Qdrant (vector store)

```bash
# Docker (recommended)
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Verify
curl http://localhost:6333/healthz
```

### Neo4j (knowledge graph) — optional

```bash
# Docker
docker run \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5

# Web UI: http://localhost:7474
# Bolt URI: bolt://localhost:7687
```

### Ollama (local parser)

```bash
ollama serve          # starts on port 11434
ollama pull llama3    # or whichever model your config.yaml specifies
```

---

## 4. Starting the API Server

```bash
cd D:/paul_lectures/multi-modal-rag
uv run python scripts/serve.py
```

The server starts at `http://localhost:8000`.

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /health` | GET | Health check |
| `POST /ingest/file` | POST | Upload and ingest a PDF |
| `POST /ingest` | POST | Ingest by local file path |
| `POST /generate` | POST | RAG query → answer |
| `GET /docs` | GET | Swagger UI |

---

## 5. Starting the Streamlit UI

Open a **second terminal**:

```bash
cd D:/paul_lectures/multi-modal-rag
uv run streamlit run rag_app.py --server.port 8501
```

Open `http://localhost:8501` in a browser.

---

## 6. Ingesting Documents

### Via Streamlit UI (recommended for one-off uploads)

1. In the sidebar, click **Choose a PDF** and select your file.
2. Click **Ingest**.
3. Wait for the spinner — it runs: parse → chunk → caption → embed → Qdrant upsert → Neo4j graph extraction.
4. On success you see: `Done — N chunks (X text, Y table, Z image)`.

### Via CLI (recommended for batch ingestion)

```bash
# Single file
uv run python scripts/ingest.py path/to/document.pdf

# Directory of files
uv run python scripts/ingest.py path/to/folder/

# Skip image captioning (faster)
uv run python scripts/ingest.py path/to/document.pdf --no-captions

# Re-ingest with a fresh collection (drops and recreates)
uv run python scripts/ingest.py path/to/document.pdf --overwrite
```

---

## 7. Verifying Ingestion

```bash
# Check that a file is present in both Qdrant and Neo4j
uv run python scripts/verify_graph.py airtel_bills.pdf

# Check all ingested data
uv run python scripts/verify_graph.py
```

Expected output when both stores have data:

```
============================================================
  QDRANT  —  collection: documentss
============================================================
  Points for 'airtel_bills.pdf': 46
  • chunk_id=airtel_bills.pdf_2_14  page=2  modality=table
  ...

============================================================
  NEO4J
============================================================
  Entity nodes for 'airtel_bills.pdf': 12
  Relationships                       : 8
  • SIVANENI Srinivasa PRASANNA  (Person)
  ...

============================================================
  ✓  'airtel_bills.pdf' is present in BOTH Qdrant (46 chunks) and Neo4j (12 entities)
============================================================
```

If Neo4j shows 0 entities after ingestion, the graph extraction LLM call may have failed — check the API server logs.

---

## 8. Asking Questions

In the Streamlit UI:

1. Type your question in the chat box at the bottom.
2. The answer shows a **grounding badge**:
   - **Green (≥70%)** — answer is well grounded in retrieved chunks.
   - **Yellow (40–69%)** — partially grounded, verify key facts.
   - **Red (<40%)** — low grounding, likely hallucination.
3. If chunks came from **multiple documents**, a warning banner lists all source files.
4. Expand **Knowledge graph context** to see what Neo4j contributed.
5. Expand **Sources (N chunks)** to verify every fact against the original chunk.

**Search settings (sidebar):**

| Setting | Default | Effect |
|---|---|---|
| Candidates (top_k) | 20 | How many chunks Qdrant retrieves before reranking |
| Final results (top_n) | 5 | How many chunks reach the LLM after reranking |
| Rerank toggle | On | Uses Jina reranker to reorder by relevance |

---

## 9. Switching LLM Provider

Edit `.env`:

```env
# Use MeshAPI for LLM (embeddings still use OpenAI)
LLM_PROVIDER=meshapi
MESH_API_KEY=rsk_...

# Use OpenAI for everything
LLM_PROVIDER=openai
```

Restart the API server after changing `.env`. No other code changes needed.

**What each provider handles:**

| Task | LLM_PROVIDER=meshapi | LLM_PROVIDER=openai |
|---|---|---|
| Embeddings | OpenAI always | OpenAI |
| Answer generation | MeshAPI | OpenAI |
| Image/table captioning | MeshAPI | OpenAI |
| Graph entity extraction | MeshAPI | OpenAI |

---

## 10. Troubleshooting

### API server won't connect

```
Cannot reach the API server. Run: `uv run python scripts/serve.py`
```

Check that `scripts/serve.py` is running and `API_PORT=8000` in `.env` matches.

---

### 401 Authentication error (LLM calls)

The system environment variable `OPENAI_API_KEY` is overriding the one in `.env`.

**Fix already applied:** `settings_customise_sources()` in `config.py` ensures `.env` always wins over system env vars.

Verify with:
```bash
uv run python -c "from doc_parser.config import get_settings; s=get_settings(); print(s.openai_api_key.get_secret_value()[:10])"
```

---

### Neo4j shows 0 entities after ingestion

Two possible causes:

**A. File was ingested before the Neo4j step was added**  
Re-ingest the file through the UI or CLI — both now include Step 6 (graph extraction).

**B. LLM returned markdown-fenced JSON**  
Fixed in `graph_extractor.py` — the extractor now strips ` ```json ``` ` fences before parsing.

---

### `UnboundLocalError: cannot access local variable 'asyncio'`

Caused by `import asyncio` inside an `if` block in a function that also uses `asyncio` at module level. Python treats it as a local variable for the whole function.

**Fix:** Remove inner `import asyncio` — use the top-level import.

---

### Qdrant `AttributeError: 'QdrantClient' object has no attribute 'search'`

qdrant-client ≥1.10 removed `.search()`. Use `.query_points()` instead:

```python
# Old (broken)
results = qdrant.search(collection_name=..., query_vector=..., limit=...)

# New
results = qdrant.query_points(collection_name=..., query=..., limit=...)
for pt in results.points:
    ...
```

---

### Hallucinated or wrong answers

1. **Check grounding score** — if yellow/red, the answer is not well-supported.
2. **Check multi-document warning** — if multiple files were retrieved, the LLM may be mixing facts.
3. **Increase top_k** — more candidates means better recall.
4. **Use specific questions** — broad questions retrieve chunks from many documents.
5. **Check source chunks** — expand "Sources" to see what the LLM actually received.

---

### `json.decoder.JSONDecodeError` in graph extraction

The LLM returned ` ```json\n{...}``` ` instead of plain JSON. Fixed in `graph_extractor.py` — markdown fences are stripped before parsing.

---

## 11. Service Summary

```
┌─────────────────────────────────────────────────────────┐
│  Terminal 1   uv run python scripts/serve.py            │
│               → FastAPI on http://localhost:8000         │
│                                                         │
│  Terminal 2   uv run streamlit run rag_app.py           │
│               → Streamlit on http://localhost:8501       │
│                                                         │
│  Docker       Qdrant on port 6333                       │
│               Neo4j  on port 7687 / 7474                │
│               Ollama on port 11434 (if parser=ollama)   │
└─────────────────────────────────────────────────────────┘
```
