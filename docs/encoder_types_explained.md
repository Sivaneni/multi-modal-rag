# Encoder Types in This Project — Deep Dive

---

## The Big Picture — Why Different Encoders?

No single model does everything well. The RAG pipeline uses different encoder types at different stages because each type makes a different **speed vs accuracy vs capability trade-off**.

```
INGESTION PIPELINE
─────────────────────────────────────────────────────────
Document
  │
  ├─ GLM-OCR (Vision-Language Model)     ← reads pixels → text
  ├─ GPT-4o  (Generative LLM)           ← text/image → rich description
  ├─ text-embedding-3-large (Bi-encoder) ← text → 3072-dim vector
  └─ Feature Hashing (Sparse encoder)   ← text → TF sparse vector

QUERY PIPELINE
─────────────────────────────────────────────────────────
Query
  │
  ├─ text-embedding-3-large (Bi-encoder) ← query → 3072-dim vector
  ├─ Feature Hashing (Sparse encoder)   ← query → TF sparse vector
  ├─ RRF Fusion                         ← rank merge (no model)
  └─ GPT-4o-mini / BGE / Jina / Qwen   ← cross-encoders → rerank score
                                           (reads query + doc together)
```

---

## 1. Bi-Encoders (Dual Encoders)

### What are they?

A bi-encoder encodes the **query** and the **document** completely independently through the same (or similar) neural network. Both are converted into fixed-size vectors in the same embedding space. Similarity is then measured with cosine similarity or dot product.

```
Query  ─→ [Encoder] ─→ vector_q  ─┐
                                    ├─ cosine_similarity(vector_q, vector_d) → score
Doc    ─→ [Encoder] ─→ vector_d  ─┘
```

The key word is **independently** — the encoder never sees query and document together.

### Why use bi-encoders?

Because you can **pre-compute document vectors once** at ingestion time and store them in the vector database. At query time you only need to encode the query (fast!), then do a nearest-neighbour search over millions of pre-computed vectors in milliseconds.

### How they work internally

They are typically transformer models (BERT-family) trained with **contrastive learning**:
- Positive pairs (query, relevant doc) are pushed close together in vector space
- Negative pairs (query, irrelevant doc) are pushed apart
- Training loss: InfoNCE / in-batch negatives

### Where used in this project

**`embedder.py` — `OpenAIEmbedder` and `GeminiEmbedder`**

| Model | Dimensions | Used for |
|---|---|---|
| `text-embedding-3-large` | 3072 | Dense vector for each chunk at ingestion; query vector at search time |
| `gemini-embedding-2-preview` | varies | Alternative provider, same role |

```python
# Ingestion: encode all chunks once
dense_embeddings = await embedder.embed([chunk.text for chunk in chunks])

# Query time: encode only the query
query_dense = (await embedder.embed([query]))[0]
```

### Trade-offs

| Pros | Cons |
|---|---|
| Very fast at search time (pre-computed docs) | Can't model query-document interaction |
| Scales to millions of documents | Misses subtle relevance nuances |
| One query encoding → search entire index | |

---

## 2. Sparse Encoders (Feature Hashing / BM25 proxy)

### What are they?

Sparse encoders convert text into **high-dimensional vectors where most values are zero**. Only the dimensions corresponding to words that actually appear in the text have non-zero values. This is the vectorised form of keyword matching.

### Why use sparse encoders?

Dense bi-encoders are great for semantic similarity but can **miss exact keyword matches** — especially for:
- Medical lab abbreviations: `WBC`, `eGFR`, `HbA1c`
- Specific numbers: `7200 /µL`, `13.5 g/dL`
- Rare proper nouns that weren't well represented in embedding training data

Sparse vectors catch these exactly because they literally hash the term and store its frequency.

### How they work in this project

**`embedder.py` — `compute_sparse_vectors()`**

This uses the **feature hashing trick** (no vocabulary needed):

```
Step 1 — Tokenise:
  "Haemoglobin 13.5 g/dL Normal"
  → ["haemoglobin", "13", "5", "g", "dl", "normal"]

Step 2 — Count (TF):
  Counter({"haemoglobin": 1, "13": 1, ...})

Step 3 — Hash to bucket (0 to 131071):
  hash("haemoglobin") % 131072 → bucket 84321, weight = 1/6 = 0.167
  hash("normal")      % 131072 → bucket 12044, weight = 1/6 = 0.167

Step 4 — SparseVector:
  indices=[12044, 84321], values=[0.167, 0.167]
```

The vector has 131,072 dimensions but only 6 non-zero values — hence "sparse".

### Where used in this project

Both at ingestion (`embed_chunks`) and at query time (`search`):
```python
sparse_vectors = compute_sparse_vectors([chunk.text for chunk in chunks])  # ingestion
query_sparse   = compute_sparse_vectors([query])[0]                         # search
```

### Trade-offs

| Pros | Cons |
|---|---|
| Exact keyword matching | No semantic understanding |
| No training required | "kidney function" ≠ "renal results" |
| Zero API cost (CPU-only) | Sensitive to spelling variations |
| Deterministic | |

---

## 3. Cross-Encoders

### What are they?

A cross-encoder takes the **query and the document concatenated together** as a single input and outputs a relevance score. The model sees both at the same time, allowing it to model complex interactions between them.

```
[CLS] query [SEP] document [SEP]
            │
        [Transformer]
            │
        [Score head]
            │
         score (0.0 – 1.0)
```

### Why use cross-encoders?

Because they are **much more accurate than bi-encoders** at judging relevance. The attention mechanism can directly compare words in the query against words in the document — e.g. it can notice that "haemoglobin" in the query matches "Haemoglobin: 13.5" in a table, and that "13.5 is within normal range 12–17" answers the question.

Bi-encoders can't do this — they encode query and document separately so there's no direct word-to-word interaction.

### Why not use cross-encoders for retrieval?

Because they're **slow at scale**. You can't pre-compute document representations — every (query, document) pair must be re-scored at query time. With 1 million documents that would take minutes.

**The standard pattern:**
```
Bi-encoder retrieval → top-100 candidates (fast, approximate)
Cross-encoder reranking → top-5 (slow, accurate, but only over 100 docs)
```

### Cross-encoders in this project

**`reranker.py` — 4 backends, all cross-encoders conceptually:**

#### BGE — `BAAI/bge-reranker-v2-minicpm-layerwise` (true cross-encoder)

The most traditional cross-encoder in the project. It's a BERT-style model with a classification head:

```
Input:  [CLS] query [SEP] document [SEP]
Model:  transformer (40 layers, but cutoff at layer 28 for speed)
Output: single logit → relevance score
```

- **LayerWise** means it can produce outputs at intermediate layers (not just the final layer)
- `cutoff_layers=[28]` stops at layer 28 — 30% faster, ~1% accuracy drop
- **Text-only** — for image chunks it uses `chunk.text` (the GPT-4o structured description) as a text proxy

#### Jina — `jina-reranker-m0` (multimodal cross-encoder, cloud API)

A native multimodal cross-encoder trained to score (query, document+image) triples:
```
Input:  query + document_text + document_image (optional)
Output: relevance_score per candidate
```

Can score image chunks using both the caption text AND the pixel content.

#### Qwen — `Qwen3-VL-Reranker-2B` (multimodal cross-encoder, local)

A vision-language model used as a sequence classifier:
```
Input:  [query, document_text] or [query, document_text, PIL_Image]
Model:  Qwen3 transformer + classification head
Output: logit → relevance score
```

Ranked #1 on MMEB-V2 (multimodal embedding benchmark). 2B parameters — small enough to run locally.

---

## 4. GPT-4o-mini as a Reranker — What Kind of Model Is It?

### What it IS: a Generative LLM (Decoder-only transformer)

GPT-4o-mini is **not** a traditional cross-encoder. It is a **generative, decoder-only large language model** (like GPT-4, Claude, Llama). Its architecture:

```
Input tokens → [Causal Self-Attention (decoder-only)] → output token probabilities
```

It generates text autoregressively — one token at a time — rather than producing a single classification score.

### What kind of encoder role it plays: Pointwise LLM Reranker

When used in `reranker.py`, GPT-4o-mini is being used as a **prompted relevance scorer** — sometimes called a "listwise" or "pointwise LLM reranker":

```python
prompt = "Rate the relevance of the following document to the query
          on a scale of 1 to 10. Reply with ONLY the integer score.

          Query: {query}
          Document: {text}"
```

This is NOT the same as a cross-encoder. It works differently:

| | True Cross-Encoder (BGE) | GPT-4o-mini as Reranker |
|---|---|---|
| Architecture | BERT encoder + classification head | GPT decoder-only |
| How it scores | Learns relevance from labelled data | Prompted to reason about relevance |
| Output | Raw logit → sigmoid → [0,1] | Generated text "7" parsed to float |
| Can reason | No — pattern match only | Yes — can explain why relevant |
| Multimodal | BGE: No, Jina/Qwen: Yes | Yes — vision input supported |
| Speed | Fast (single forward pass) | Slower (generates tokens) |

### Why use GPT-4o-mini instead of a true cross-encoder?

1. **Multimodal** — it can rate image chunks by looking at the actual image, not just text
2. **Zero configuration** — no model download, no GPU, uses your existing OpenAI key
3. **Reasoning** — it can implicitly reason about relevance ("this table contains the exact lab value asked about")
4. **Default backend** — works out of the box without extra dependencies

### Why it's not always better than BGE

- Costs money per call (~$0.03–0.10 per rerank)
- Slower than a local cross-encoder (~800ms vs ~50ms for BGE)
- Score scale (1–10) is less calibrated than a true probability score
- Non-deterministic (though `temperature=0.0` helps)

---

## 5. GLiNER2 — What Is It and Where Would It Fit?

### What is GLiNER?

**GLiNER** (Generalist and Lightweight Named Entity Recognition) is a model for **Named Entity Recognition (NER)** that can detect arbitrary entity types defined at inference time — not just the fixed types (PERSON, ORG, LOC) of traditional NER.

**GLiNER2** is the second generation with improved architecture and accuracy.

### What kind of encoder is it?

GLiNER uses a **bi-encoder architecture adapted for span classification**:

```
Text spans:   [Encoder] ─→ span_vectors
Entity types: [Encoder] ─→ label_vectors
                              │
                    cosine_similarity(span, label) → is this span a "DIAGNOSIS"?
```

It encodes the text spans and entity type labels **separately** (like a bi-encoder) but then computes similarity between them to decide whether a span matches a label. This is why it can handle arbitrary entity types at runtime — you just add new label strings.

### Why is this architecture clever?

Traditional NER (e.g. BERT-NER) trains a classifier head for fixed entity types:
```
"Patient" → [O]
"John"    → [B-PERSON]
"has"     → [O]
"WBC"     → [B-MEDICAL_TEST]   ← only works if trained on MEDICAL_TEST
```

GLiNER can detect `"DIAGNOSIS"`, `"LAB_TEST"`, `"MEDICATION"` at inference time because it computes similarity between the span and the label string — no retraining needed.

### How it works step by step

```
Input text: "Patient John's Haemoglobin is 13.5 g/dL (Normal)"
Labels:     ["patient_name", "lab_test", "lab_value", "clinical_status"]

Step 1 — Encode the full text with a transformer (token embeddings)
Step 2 — Extract span representations for all possible spans
          (e.g. "Haemoglobin", "13.5", "g/dL", "Normal")
Step 3 — Encode each label string: "lab_test" → label_vector
Step 4 — cosine_sim(span_"Haemoglobin", label_"lab_test") → 0.91 → MATCH
          cosine_sim(span_"13.5", label_"lab_value")       → 0.88 → MATCH
          cosine_sim(span_"Normal", label_"clinical_status")→ 0.85 → MATCH

Output: {
  "lab_test":        "Haemoglobin",
  "lab_value":       "13.5 g/dL",
  "clinical_status": "Normal"
}
```

### Where would GLiNER2 fit in this project?

It is **not currently used** in the project, but it would be a natural addition in two places:

#### Option A — Post-chunking metadata enrichment
After `document_aware_chunking`, run GLiNER2 on each chunk to extract structured entities and store them in the Qdrant payload:

```python
# After chunking, before embedding
entities = gliner.predict_entities(
    chunk.text,
    labels=["lab_test", "lab_value", "reference_range", "clinical_status",
            "patient_name", "diagnosis", "medication"]
)
chunk.metadata["entities"] = entities
# → stored in Qdrant payload
```

This enables **entity-level filtering** at retrieval time: "show me all chunks that mention a diagnosis".

#### Option B — Query understanding
Before searching, run GLiNER2 on the user query to extract what entities are being asked about:

```python
query = "What is the patient's WBC count?"
entities = gliner.predict_entities(query, labels=["lab_test"])
# → {"lab_test": "WBC"}
# → use this to filter_modality="table" in Qdrant search
```

### GLiNER2 vs other NER approaches

| Model | Entity types | Architecture | Requires retraining? |
|---|---|---|---|
| spaCy NER | Fixed (PER, ORG, LOC) | CNN/LSTM + CRF | Yes, for new types |
| BERT-NER | Fixed (whatever trained on) | Bi-encoder + classification head | Yes |
| **GLiNER2** | **Arbitrary at runtime** | **Bi-encoder + span-label similarity** | **No** |
| GPT-4o prompting | Arbitrary at runtime | Generative LLM | No |

GLiNER2 is much faster and cheaper than GPT-4o prompting for NER while being more flexible than traditional NER.

---

## Summary — All Encoder Types in This Project

| Encoder Type | Model | Where Used | Speed | Accuracy | Cost |
|---|---|---|---|---|---|
| **Bi-encoder** | text-embedding-3-large | Ingestion + search (dense) | Fast | Good semantic | API cost |
| **Bi-encoder** | gemini-embedding-2-preview | Ingestion + search (dense, alt) | Fast | Good semantic | API cost |
| **Sparse encoder** | Feature hashing (TF) | Ingestion + search (keyword) | Very fast | Exact match | Free |
| **Vision-Language** | GLM-OCR (Ollama) | OCR: image → text | Slow | High OCR quality | Free (local) |
| **Generative LLM** | GPT-4o | Enrichment: image/table → description | Slow | Very high | API cost |
| **Generative LLM (as cross-encoder)** | GPT-4o-mini | Reranking (pointwise scorer) | Medium | High + multimodal | API cost |
| **True cross-encoder** | BGE reranker | Reranking (local, text-only) | Fast | High | Free |
| **Multimodal cross-encoder** | Jina M0 | Reranking (cloud, multimodal) | Medium | High + multimodal | API cost |
| **Multimodal cross-encoder** | Qwen3-VL-2B | Reranking (local, multimodal) | Medium | Highest | Free |
| **Bi-encoder for NER** | GLiNER2 | Not used yet — entity extraction | Fast | High | Free |

---

## The Core Intuition

```
                SPEED
                  ↑
  Sparse ─────────┤
  (feature hash)  │
                  │  Bi-encoder
                  │  (text-embedding-3-large)
                  │
                  │         Cross-encoder
                  │         (BGE, Jina, Qwen)
                  │
                  │                    Generative LLM
                  │                    (GPT-4o, GPT-4o-mini)
                  └────────────────────────────────────────→ ACCURACY / CAPABILITY
```

The pipeline uses them in order from left to right:
1. Sparse + Bi-encoder for fast broad retrieval (top-20 from millions)
2. Cross-encoder / LLM for slow accurate reranking (top-5 from 20)
3. Generative LLM for final answer synthesis (1 answer from top-5)

Each step narrows the candidate set while spending more compute per candidate.