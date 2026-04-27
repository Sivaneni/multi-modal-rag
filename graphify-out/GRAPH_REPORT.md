# Graph Report - .  (2026-04-25)

## Corpus Check
- Corpus is ~20,307 words - fits in a single context window. You may not need a graph.

## Summary
- 379 nodes · 701 edges · 21 communities detected
- Extraction: 60% EXTRACTED · 40% INFERRED · 0% AMBIGUOUS · INFERRED: 282 edges (avg confidence: 0.59)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_API Config & Dependencies|API Config & Dependencies]]
- [[_COMMUNITY_Embedding & Sparse Vectors|Embedding & Sparse Vectors]]
- [[_COMMUNITY_Config & Settings Management|Config & Settings Management]]
- [[_COMMUNITY_RAG Generation API|RAG Generation API]]
- [[_COMMUNITY_Local Ollama Parse Stack|Local Ollama Parse Stack]]
- [[_COMMUNITY_PDF Visualization|PDF Visualization]]
- [[_COMMUNITY_Document Parsing Pipeline|Document Parsing Pipeline]]
- [[_COMMUNITY_Retrieval & Reranking Models|Retrieval & Reranking Models]]
- [[_COMMUNITY_Streamlit Visualizer|Streamlit Visualizer]]
- [[_COMMUNITY_FastAPI App Lifecycle|FastAPI App Lifecycle]]
- [[_COMMUNITY_Document-Aware Chunking|Document-Aware Chunking]]
- [[_COMMUNITY_PDF Utilities|PDF Utilities]]
- [[_COMMUNITY_CLI Parser Tests|CLI Parser Tests]]
- [[_COMMUNITY_Debug Scripts|Debug Scripts]]
- [[_COMMUNITY_Backend Constraint Validation|Backend Constraint Validation]]
- [[_COMMUNITY_Parse Result Builder|Parse Result Builder]]
- [[_COMMUNITY_UV Package Manager|UV Package Manager]]
- [[_COMMUNITY_Ruff Linter|Ruff Linter]]
- [[_COMMUNITY_Loguru Logging|Loguru Logging]]
- [[_COMMUNITY_PyMuPDF Library|PyMuPDF Library]]
- [[_COMMUNITY_Tiktoken Library|Tiktoken Library]]

## God Nodes (most connected - your core abstractions)
1. `Chunk` - 52 edges
2. `Settings` - 45 edges
3. `DocumentParser` - 25 edges
4. `QdrantDocumentStore` - 25 edges
5. `BaseEmbedder` - 20 edges
6. `get_settings()` - 19 edges
7. `_run_ingest()` - 16 edges
8. `BaseReranker` - 14 edges
9. `ingest_file()` - 13 edges
10. `ParseResult` - 12 edges

## Surprising Connections (you probably didn't know these)
- `Return one float vector per text, in input order.` --uses--> `Chunk`  [INFERRED]
  D:\paul_lectures\multi-modal-rag\src\doc_parser\ingestion\embedder.py → D:\paul_lectures\multi-modal-rag\src\doc_parser\chunker.py
- `A RAG-ready document chunk.      Attributes:         text: The chunk text con` --uses--> `ElementLike`  [INFERRED]
  D:\paul_lectures\multi-modal-rag\src\doc_parser\chunker.py → D:\paul_lectures\multi-modal-rag\src\doc_parser\post_processor.py
- `Score candidates via Qwen VL (offloaded to thread pool) and return top-n.` --uses--> `Settings`  [INFERRED]
  D:\paul_lectures\multi-modal-rag\src\doc_parser\retrieval\reranker.py → D:\paul_lectures\multi-modal-rag\src\doc_parser\config.py
- `FastAPI app factory with lifespan and middleware.` --uses--> `DocumentParser`  [INFERRED]
  D:\paul_lectures\multi-modal-rag\src\doc_parser\api\app.py → D:\paul_lectures\multi-modal-rag\src\doc_parser\pipeline.py
- `FastAPI app factory with lifespan and middleware.` --uses--> `ParseResult`  [INFERRED]
  D:\paul_lectures\multi-modal-rag\src\doc_parser\api\app.py → D:\paul_lectures\multi-modal-rag\src\doc_parser\pipeline.py

## Hyperedges (group relationships)
- **Four-Phase RAG Pipeline: Parse â†’ Ingest â†’ Search/Re-rank â†’ REST API** — readme_phase1_parse, readme_phase2_ingest, readme_phase3_search_rerank, readme_phase4_rest_api [EXTRACTED 1.00]
- **Ollama Local Parsing Stack: GLM-OCR + PP-DocLayout-V3 + glmocr SDK + config** — ollama_readme_glm_ocr_ollama, ollama_readme_pp_doclayout_v3_local, ollama_readme_glmocr_sdk, ollama_readme_config_yaml [EXTRACTED 0.95]
- **Hybrid Vector Ingestion: Dense Embeddings + BM25 Sparse + RRF in Qdrant** — readme_pluggable_embeddings, readme_bm25_sparse_vectors, readme_qdrant_hybrid_store, readme_hybrid_search_rrf [EXTRACTED 1.00]

## Communities

### Community 0 - "API Config & Dependencies"
Cohesion: 0.05
Nodes (49): ABC, BaseSettings, Application settings loaded from environment variables / .env file., Settings, get_openai_client(), get_reranker_dep(), get_store(), Shared FastAPI dependency providers. (+41 more)

### Community 1 - "Embedding & Sparse Vectors"
Cohesion: 0.07
Nodes (43): Chunk, A RAG-ready document chunk.      Attributes:         text: The chunk text con, compute_sparse_vectors(), embed_chunks(), embed_texts(), GeminiEmbedder, OpenAIEmbedder, Embedder: dense text embeddings (OpenAI/Gemini) + BM25 sparse vectors (feature h (+35 more)

### Community 2 - "Config & Settings Management"
Cohesion: 0.1
Nodes (31): configure_logging(), get_settings(), Configuration management using pydantic-settings., Return the singleton Settings instance., Configure root logger with the given level., get_embedder_dep(), get_embedder(), _collect_files() (+23 more)

### Community 3 - "RAG Generation API"
Cohesion: 0.11
Nodes (34): BaseModel, _build_user_content(), generate(), POST /generate endpoint — full RAG in one call., Build the user message content for GPT-4o.      Returns a plain string when no, Retrieve relevant chunks and generate an answer with GPT-4o.      1. Embed que, delete_collection(), health() (+26 more)

### Community 4 - "Local Ollama Parse Stack"
Cohesion: 0.07
Nodes (31): api_mode: ollama_generate (Ollama endpoint config), ollama/config.yaml, GLM-OCR via Ollama (Local Testing), glmocr SDK, OpenCV (image preprocessing), ollama/output/ (saved parse results), PP-DocLayout-V3 (Local via HuggingFace), Rationale: api_mode=ollama_generate needed to avoid 502 errors on Ollama endpoint (+23 more)

### Community 5 - "PDF Visualization"
Cohesion: 0.12
Nodes (22): build_legend(), draw_bboxes(), get_color(), Render a color legend for the element types found on this page., Render a PDF page as a PIL Image at RENDER_DPI., Draw colored bounding boxes onto the page image.      bbox_2d values from the, render_page(), collect_input_files() (+14 more)

### Community 6 - "Document Parsing Pipeline"
Cohesion: 0.1
Nodes (20): from_sdk_result(), PageResult, ParsedElement, Main document parsing pipeline wrapping the GLM-OCR SDK., Save this result as Markdown and JSON files.          Args:             outpu, A single detected and recognized document element.      Attributes:         l, Parsing result for a single document page.      Attributes:         page_num:, assemble_markdown() (+12 more)

### Community 7 - "Retrieval & Reranking Models"
Cohesion: 0.1
Nodes (23): BGE Reranker (BAAI/bge-reranker-v2-minicpm), BM25 Sparse Vectors, Google Gemini Embeddings, GPT-4o Image Captioner, Hybrid Search with RRF Fusion, Jina Reranker M0, OpenAI Embeddings (text-embedding-3-large/small), OpenAI GPT-4o-mini Re-ranker (+15 more)

### Community 8 - "Streamlit Visualizer"
Cohesion: 0.12
Nodes (18): build_legend(), draw_bboxes(), draw_polygons(), find_pdf(), get_color(), load_result(), Streamlit app: visualize Ollama/PP-DocLayoutV3 parsed results.  Supports two w, Draw translucent polygon overlays for elements that have polygon data. (+10 more)

### Community 9 - "FastAPI App Lifecycle"
Cohesion: 0.12
Nodes (14): create_app(), lifespan(), FastAPI app factory with lifespan and middleware., Configure logging and report startup/shutdown., Construct and return the FastAPI application., BaseHTTPMiddleware, _InterceptHandler, Loguru logging setup with stdlib interception. (+6 more)

### Community 10 - "Document-Aware Chunking"
Cohesion: 0.22
Nodes (14): document_aware_chunking(), _estimate_tokens(), _infer_modality(), Structure-aware and document-aware chunkers for RAG-ready document chunks., Chunk a whole document across ALL pages in a single pass.      Unlike ``struct, Chunk a single page's elements respecting structure boundaries.      For multi, Derive chunk modality from element label(s).      Args:         element_types, Estimate token count using word count heuristic.      Args:         text: Inp (+6 more)

### Community 11 - "PDF Utilities"
Cohesion: 0.25
Nodes (7): count_pdf_pages(), pdf_page_to_image(), PyMuPDF helpers for PDF → image extraction and validation., Extract a single PDF page as a PIL Image.      Args:         pdf_path: Path t, Return the number of pages in a PDF file.      Args:         pdf_path: Path t, Validate that a file exists and has a supported extension.      Args:, validate_input_file()

### Community 12 - "CLI Parser Tests"
Cohesion: 1.0
Nodes (2): main(), parse_args()

### Community 13 - "Debug Scripts"
Cohesion: 1.0
Nodes (1): Debug script — prints the raw SDK response so we can fix the parser.

### Community 14 - "Backend Constraint Validation"
Cohesion: 1.0
Nodes (1): Enforce backend-specific constraints and auto-set config path.

### Community 15 - "Parse Result Builder"
Cohesion: 1.0
Nodes (1): Build a ParseResult from a raw GLM-OCR SDK PipelineResult.          The SDK re

### Community 19 - "UV Package Manager"
Cohesion: 1.0
Nodes (1): uv Package Manager

### Community 20 - "Ruff Linter"
Cohesion: 1.0
Nodes (1): Ruff Linter

### Community 21 - "Loguru Logging"
Cohesion: 1.0
Nodes (1): Loguru Structured Logging

### Community 22 - "PyMuPDF Library"
Cohesion: 1.0
Nodes (1): PyMuPDF (PDF to image extraction)

### Community 23 - "Tiktoken Library"
Cohesion: 1.0
Nodes (1): tiktoken (token counting)

## Knowledge Gaps
- **86 isolated node(s):** `Streamlit app: visualize Ollama/PP-DocLayoutV3 parsed results.  Supports two w`, `Render a PDF page as a PIL Image at RENDER_DPI.`, `Draw colored bounding boxes onto the page image.`, `Draw translucent polygon overlays for elements that have polygon data.`, `Render a color legend for the element types found on this page.` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `CLI Parser Tests`** (3 nodes): `test_parse.py`, `main()`, `parse_args()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Debug Scripts`** (2 nodes): `Debug script — prints the raw SDK response so we can fix the parser.`, `debug_raw.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Backend Constraint Validation`** (1 nodes): `Enforce backend-specific constraints and auto-set config path.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Parse Result Builder`** (1 nodes): `Build a ParseResult from a raw GLM-OCR SDK PipelineResult.          The SDK re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `UV Package Manager`** (1 nodes): `uv Package Manager`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ruff Linter`** (1 nodes): `Ruff Linter`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Loguru Logging`** (1 nodes): `Loguru Structured Logging`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `PyMuPDF Library`** (1 nodes): `PyMuPDF (PDF to image extraction)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tiktoken Library`** (1 nodes): `tiktoken (token counting)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Chunk` connect `Embedding & Sparse Vectors` to `API Config & Dependencies`, `Document-Aware Chunking`, `Config & Settings Management`, `PDF Visualization`?**
  _High betweenness centrality (0.171) - this node is a cross-community bridge._
- **Why does `get_settings()` connect `Config & Settings Management` to `API Config & Dependencies`, `RAG Generation API`, `PDF Visualization`, `Document Parsing Pipeline`, `FastAPI App Lifecycle`?**
  _High betweenness centrality (0.168) - this node is a cross-community bridge._
- **Why does `Settings` connect `API Config & Dependencies` to `Embedding & Sparse Vectors`, `Config & Settings Management`, `RAG Generation API`?**
  _High betweenness centrality (0.157) - this node is a cross-community bridge._
- **Are the 49 inferred relationships involving `Chunk` (e.g. with `POST /ingest endpoint — file upload and JSON-path variants.` and `Return a list of supported document paths from a file or directory.`) actually correct?**
  _`Chunk` has 49 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `Settings` (e.g. with `BaseEmbedder` and `OpenAIEmbedder`) actually correct?**
  _`Settings` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `DocumentParser` (e.g. with `FastAPI app factory with lifespan and middleware.` and `Render a PDF page as a PIL Image at RENDER_DPI.`) actually correct?**
  _`DocumentParser` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `QdrantDocumentStore` (e.g. with `POST /ingest endpoint — file upload and JSON-path variants.` and `Return a list of supported document paths from a file or directory.`) actually correct?**
  _`QdrantDocumentStore` has 18 INFERRED edges - model-reasoned connections that need verification._