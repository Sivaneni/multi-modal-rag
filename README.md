# Hybrid GraphRAG — Multi-Modal Document Intelligence

> A production-grade RAG system that combines **semantic vector search** (Qdrant) with a **knowledge graph** (Neo4j) to answer questions over multi-modal PDF documents — text, tables, images, formulas, and algorithms.

---

## Problem Statement

Standard RAG systems embed document text and retrieve by semantic similarity. This breaks down for:

- **Structured data** (tables, invoices, lab reports) where exact values matter more than meaning
- **Entity-relationship queries** ("who referred Prasanna?", "what is connected to X?") that require graph traversal, not similarity search
- **Multi-document queries** where facts need to be kept per-source to avoid hallucinated blending
- **Multi-modal documents** where critical information lives in images, charts, and formulas, not just prose

This system solves all four by combining hybrid vector search, a knowledge graph, multi-modal LLM captioning, and a hallucination grounding detector.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Streamlit UI  (:8501)                    │
│         Upload PDF ──► POST /ingest     Ask ──► POST /generate   │
└──────────────────────────┬───────────────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼───────────────────────────────────────┐
│                   FastAPI Server  (:8000)                         │
│                                                                  │
│  INGEST PIPELINE                    QUERY PIPELINE               │
│  1. Parse PDF (Ollama / Cloud)      1. Embed query (OpenAI)      │
│  2. Chunk (structure-aware)         2. Hybrid search → Qdrant    │
│  3. Caption chunks (LLM)            3. Rerank (Jina/OpenAI)      │
│  4. Embed dense+sparse (OpenAI)     4. Fetch graph → Neo4j       │
│  5. Upsert → Qdrant                 5. Build labelled context     │
│  6. Extract entities → Neo4j        6. Generate answer (LLM)     │
└───────────┬──────────────────────────────────┬───────────────────┘
            │                                  │
  ┌─────────▼──────┐                 ┌─────────▼──────┐
  │     Qdrant     │                 │     Neo4j      │
  │  Dense + Sparse│                 │ Entity Graph   │
  │  port 6333     │                 │  port 7687     │
  └────────────────┘                 └────────────────┘
                        LLM calls
               ┌────────────────────────┐
               │  MeshAPI  or  OpenAI   │
               │  (set via LLM_PROVIDER)│
               └────────────────────────┘
```

### RAG Pipeline Flow

```
PDF
 │
 ├─ Parse ──► text blocks · tables · images · formulas · algorithms
 │
 ├─ Chunk ──► structure-aware grouping (cross-page headings, captions)
 │
 ├─ Enrich ──► LLM captions each image/table/formula/algorithm
 │
 ├─ Embed ──► OpenAI dense (3072-dim) + BM25 sparse per chunk
 │
 ├─ Qdrant ──► hybrid collection upserted
 │
 └─ Neo4j ──► LLM extracts entities + relationships → MERGE graph
                    ↓
             At query time:
             Qdrant hybrid search → rerank → top-N chunks
             Neo4j entity context for retrieved source files
             LLM generates answer with strict source attribution
             Streamlit shows grounding score + source verification
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Document Parsing** | Ollama (local) / Z.AI Cloud MaaS |
| **Chunking** | Custom structure-aware chunker (multi-modal) |
| **LLM — Captioning & Generation** | MeshAPI (OpenAI-compatible) / OpenAI GPT-4o |
| **Embeddings** | OpenAI `text-embedding-3-large` (3072-dim) |
| **Vector Store** | Qdrant — hybrid dense + sparse (BM25) |
| **Knowledge Graph** | Neo4j 5 — entity nodes + typed relationships |
| **Reranker** | Jina Reranker / OpenAI cross-encoder |
| **API Framework** | FastAPI + Uvicorn |
| **UI** | Streamlit |
| **Config Management** | pydantic-settings (`.env` always wins over system env vars) |
| **Package Manager** | uv (Python 3.12) |

---

## Setup & Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | Managed by `uv` |
| uv | `pip install uv` |
| Qdrant | Docker on port `6333` |
| Neo4j | Docker on port `7687` *(optional — graph features disabled without it)* |
| Ollama | Port `11434` *(if `PARSER_BACKEND=ollama`)* |
| OpenAI API key | Always required for embeddings |
| MeshAPI key | Required when `LLM_PROVIDER=meshapi` |

### 1. Clone and install

```bash
git clone https://github.com/Sivaneni/multi-modal-rag.git
cd multi-modal-rag
git checkout hybrid-rag-deployment
uv sync
```

### 2. Configure `.env`

Create a `.env` file in the project root (use `.env.example` as template):

```env
# Parser
PARSER_BACKEND=ollama          # "cloud" | "ollama"

# LLM provider
LLM_PROVIDER=meshapi           # "meshapi" | "openai"
MESH_API_KEY=rsk_...

# OpenAI — embeddings always use OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_LLM_MODEL=openai/gpt-4o-mini

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=documents

# Neo4j (comment out to disable graph features)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Reranker
RERANKER_BACKEND=jina
JINA_API_KEY=jina_...

# API
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start external services

```bash
# Qdrant
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Neo4j (optional)
docker run --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password neo4j:5

# Ollama (if PARSER_BACKEND=ollama)
ollama serve
```

---

## How to Run Locally

### Terminal 1 — API Server

```bash
uv run python scripts/serve.py
# → http://localhost:8000
# → Swagger UI: http://localhost:8000/docs
```

### Terminal 2 — Streamlit UI

```bash
uv run streamlit run rag_app.py --server.port 8501
# → http://localhost:8501
```

### Ingest a document (CLI)

```bash
uv run python scripts/ingest.py path/to/document.pdf
```

### Verify ingestion in both stores

```bash
uv run python scripts/verify_graph.py document.pdf
```

### Ask a question via API

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total amount due?", "top_k": 20, "top_n": 5, "rerank": true}'
```

---

## Key Features

| Feature | Description |
|---|---|
| **Multi-modal ingestion** | Tables → full markdown; Images → structured CAPTION/DETAIL; Formulas → plain-English description |
| **Hybrid search** | Dense (semantic) + Sparse (BM25) fused with RRF |
| **Knowledge graph** | LLM extracts entities/relationships → Neo4j; injected as context at query time |
| **Hallucination grounding score** | % of answer words found in retrieved source chunks — green/yellow/red badge |
| **Multi-document warning** | Alert when answer spans multiple source files |
| **Strict attribution** | Every context chunk labelled `[filename \| page N]`; LLM instructed to cite and not blend |
| **Pluggable LLM** | Switch between MeshAPI and OpenAI via single `.env` change |
| **Idempotent ingest** | Neo4j uses `MERGE` — safe to re-upload the same document |

---

## Demo

> 🎥 **Demo Video:** _[Link to be added]_

Screenshots:

- Upload PDF → Ingest (parse + caption + embed + graph extraction)
- Ask question → Grounding badge + Knowledge graph context expander + Source chunk verification
- Multi-document warning when answer crosses files

  <img width="1896" height="955" alt="image" src="https://github.com/user-attachments/assets/803e4dbe-fcb2-47bb-b340-ca621108d2e9" />
  <img width="1433" height="821" alt="image" src="https://github.com/user-attachments/assets/643a8724-3e86-4b4a-b720-5298a41a1b8a" />
  <img width="1906" height="792" alt="image" src="https://github.com/user-attachments/assets/4c109b28-8f67-4f86-9461-e08def570a9c" />
  <img width="1157" height="427" alt="image" src="https://github.com/user-attachments/assets/7c42c5be-abfd-43fc-bab7-8b53aa18fb23" />
  




---

## Team

| Name | Role |
|---|---|
| **SIVANENI Srinivasa Prasanna** | Solo — Architecture, Implementation, Testing |

---

## Limitations & Future Scope

### Current Limitations

| Limitation | Detail |
|---|---|
| **Grounding score is word-overlap only** | High score doesn't guarantee factual accuracy — it measures word presence, not truth |
| **Graph extraction is one-shot** | Entire document text sent in one LLM call; very large documents get truncated at 12,000 chars |
| **Embedding model is fixed post-collection** | Changing `EMBEDDING_DIMENSIONS` requires full re-ingestion with `--overwrite` |
| **Neo4j queried by source file** | Graph context includes all entities from retrieved documents, not just the most relevant ones |
| **Reranker adds latency** | Cross-encoder reranking adds ~1–2s per query; togglable but on by default |
| **Single collection** | All documents share one Qdrant collection; no per-user or per-project isolation |
| **No streaming** | LLM response is returned all at once; no token streaming to UI |

### Future Scope

- **Streaming answers** — Stream LLM tokens to Streamlit for better UX
- **Graph-aware retrieval** — Use entity names from the question to query Neo4j first, then use matched entities to guide vector search
- **Chunk-level graph linking** — Store chunk ID alongside Neo4j nodes to link graph entities directly to their source chunks
- **Multi-tenant collections** — Per-user/session Qdrant collections for data isolation
- **Evaluation pipeline** — RAGAS / TruLens integration to measure retrieval precision and answer faithfulness automatically
- **Incremental graph updates** — Update only changed entities on re-ingest instead of full MERGE sweep
- **Table-aware querying** — Special query path for numeric/structured queries that bypasses embedding similarity and uses SQL-like filtering
- **Chat memory** — Maintain conversation context so follow-up questions work correctly

---

## Documentation

| Document | Description |
|---|---|
| [`docs/runbook.md`](docs/runbook.md) | Full operational runbook — services, ingestion, troubleshooting
---

## Project Structure

```
multi-modal-rag/
├── rag_app.py                          ← Streamlit UI
├── .env                                ← Configuration (gitignored)
├── .env.example                        ← Template
├── docs/
│   ├── runbook.md                      ← Operational runbook
│   └── complete_learning_guide.md      ← Learning guide + issue log
├── scripts/
│   ├── ingest.py                       ← CLI batch ingestion
│   ├── serve.py                        ← Start FastAPI server
│   └── verify_graph.py                 ← Verify Qdrant + Neo4j alignment
└── src/doc_parser/
    ├── config.py                       ← Settings + LLM client factories
    ├── api/
    │   ├── dependencies.py             ← FastAPI dependency providers
    │   └── routes/
    │       ├── ingest.py               ← POST /ingest/file
    │       └── generate.py             ← POST /generate (full RAG pipeline)
    ├── ingestion/
    │   ├── embedder.py                 ← Dense + sparse embedding
    │   ├── vector_store.py             ← Qdrant wrapper
    │   ├── image_captioner.py          ← Multi-modal LLM enrichment
    │   ├── graph_models.py             ← Pydantic models for graph data
    │   ├── graph_extractor.py          ← LLM → entity/relationship JSON
    │   └── neo4j_ingestor.py           ← MERGE into Neo4j
    └── retrieval/
        ├── reranker.py                 ← Cross-encoder reranker
        ├── graph_context.py            ← Cypher: entities by source file
        └── graph_formatter.py          ← Graph records → readable facts
```
