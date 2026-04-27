# `document_aware_chunking` — Deep Dive

This function takes the entire parsed document (all pages, all elements) and breaks it into **RAG-ready chunks** — pieces small enough to embed and retrieve, but smart enough to keep related content together.

---

## Inputs

```python
pages: list[tuple[int, list[ElementLike]]]   # [(page_num, [elements]), ...]
source_file: str                              # used to build chunk IDs
max_chunk_tokens: int = 512                  # max size of a text chunk
```

For a blood test PDF, `pages` would look like:
```python
[
  (1, [
    ParsedElement(label="paragraph_title", text="## CITY DIAGNOSTICS LAB", reading_order=0),
    ParsedElement(label="text",            text="123 Health Street...",    reading_order=1),
    ParsedElement(label="table",           text="...",                     reading_order=3),
    ...
  ])
]
```

---

## Step 1: Flatten and sort all elements (lines 151–159)

```python
all_pairs = [(page_num, el) for page_num, elements in pages for el in elements]
all_pairs.sort(key=lambda x: (x[0], x[1].reading_order))
```

Multi-page documents get flattened into a **single stream** sorted by `(page, reading_order)`. This is the core advantage over `structure_aware_chunking` — a heading at the bottom of page 1 can attach to content at the top of page 2.

---

## Step 2: Accumulator state (lines 165–173)

```python
current_texts: list[str] = []    # text pieces building up the current chunk
current_labels: list[str] = []   # their labels
current_tokens: int = 0          # running token estimate
current_page: int = ...          # page of the first element in this chunk

pending_title: str | None = None          # a heading waiting to attach forward
pending_title_label: str | None = None
pending_title_page: int = ...
```

Think of it as a **shopping cart** (`current_texts`) plus a **sticky note** (`pending_title`). The sticky note holds a heading and waits to be glued to the next real content.

---

## Step 3: Processing each element (lines 215–315)

Every element falls into one of four cases:

### Case A — Atomic element (`table`, `formula`, `image`, `figure`, `algorithm`)

These **must never be split or merged** with other content. Each gets its own chunk.

```
[pending_title = "Figure 3"]  +  [image element arrives]
         ↓
figure_caption = "Figure 3"   ← intercepted BEFORE flush
flush_current()                ← emit whatever text was accumulating
emit atomic chunk:
  text = "Figure 3\n\nimage_text"
  is_atomic = True
  bbox = element.bbox          ← only atomic chunks keep the bounding box
```

Special case: if `pending_title` is a **`figure_title`** (a caption label from PP-DocLayoutV3), it gets **prepended into the atomic chunk** so the caption and the figure are co-located in the same chunk. This matters for retrieval — you want "Figure 3: Blood glucose trend" and the table data in the same chunk.

---

### Case B — Title element (`paragraph_title`, `document_title`, `figure_title`)

Titles are **never emitted immediately** — they wait to attach to the next content element.

```
[current_texts has content]  →  flush_current()  →  set pending_title
[current_texts is empty]     →  flush orphan pending_title  →  set new pending_title
[current_texts is empty, no pending_title]  →  just set pending_title
```

Example across pages:
```
Page 1: "## Results"              → pending_title = "## Results"
Page 2: "Patient showed..."       → pending_title absorbed into this chunk
```
Result: one chunk = `"## Results\n\nPatient showed..."` instead of `"## Results"` being a useless orphan chunk.

---

### Case C — Large text element (bigger than `max_chunk_tokens` on its own)

```python
if token_estimate > max_chunk_tokens:
    flush_current()
    sub_chunks = _split_text_into_sub_chunks(text, max_chunk_tokens)
    # emits N chunks, each within the limit
```

Splits on **whitespace boundaries** (never mid-word). Token count is estimated as `word_count × 1.3` to approximate BPE tokenizers like tiktoken.

---

### Case D — Regular text element (the most common case)

```python
# Would this element overflow the current chunk?
if current_texts and (current_tokens + token_estimate + pending_tokens > max_chunk_tokens):
    flush_current()

# Absorb the pending title heading into the accumulator
if pending_title is not None:
    current_texts.append(pending_title)
    ...

# Add this element to the accumulator
current_texts.append(text)
current_tokens += token_estimate

# Auto-flush if we hit the limit exactly
if current_tokens >= max_chunk_tokens:
    flush_current()
```

Note `pending_tokens` is included in the overflow check — the heading's size is factored in before deciding to flush.

---

## `flush_current()` — the emit function (lines 175–213)

Called whenever accumulated content needs to become a chunk:

```python
chunk = Chunk(
    text="\n\n".join(texts_to_flush),   # heading + content joined
    chunk_id=f"{source_file}_{page}_{chunk_idx}",
    page=page_to_use,                   # page where the chunk started
    is_atomic=False,
    modality=_infer_modality(labels),   # "text" | "table" | "image" | "formula"
    bbox=None,                          # multi-element chunks don't have a bbox
)
```

After flushing, `current_texts`, `current_labels`, `current_tokens` are reset to empty.

---

## Full example with blood test PDF

```
Elements in order:
  (1, paragraph_title) "## CITY DIAGNOSTICS LAB"
  (1, text)            "123 Health Street..."
  (1, table)           "| Test | Result | Range |..."

Processing:
  "## CITY DIAGNOSTICS LAB"  → pending_title = "## CITY DIAGNOSTICS..."
  "123 Health Street..."      → absorb pending_title + this text → accumulator
                                current_texts = ["## CITY DIAGNOSTICS...", "123 Health Street..."]
  "| Test | Result |..."      → ATOMIC → flush accumulator first → emit text chunk
                                → then emit table as its own atomic chunk (with bbox)

Output chunks:
  Chunk 0: text="## CITY DIAGNOSTICS LAB\n\n123 Health Street...", is_atomic=False, modality="text"
  Chunk 1: text="| Test | Result | Range |...",                    is_atomic=True,  modality="table", bbox=[46,205,952,...]
```

---

## Why this matters for RAG

| Decision | Reason |
|---|---|
| Tables are atomic | A split table row is meaningless for retrieval |
| Titles attach forward | Retrieving "Patient Name: John" without "PATIENT DETAILS" header loses context |
| Cross-page flattening | Section headings at page bottom don't become orphan chunks |
| Figure caption + image in one chunk | LLM gets caption and visual data together |
| Token budget includes heading | Heading + content always fits in one embedding window |
