"""Streamlit RAG UI — upload PDFs and ask questions using the doc-parser API."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

API_BASE = "http://localhost:8000"

# Stricter system prompt — forces "not found" over guessing, no cross-document blending
_SYSTEM_PROMPT = (
    "You are a precise document assistant. "
    "Each context chunk is labelled [filename | page N]. "
    "Answer the question using ONLY information from those labelled chunks. "
    "Rules you MUST follow:\n"
    "1. Cite the exact filename and page for every fact: (source: filename, page N).\n"
    "2. If facts come from DIFFERENT files, present them separately — do NOT blend or merge them into one narrative.\n"
    "3. If a specific piece of information is NOT in the context, say exactly: "
    "'This information was not found in the retrieved documents.'\n"
    "4. Do NOT infer, guess, or use any prior knowledge.\n"
    "5. Do NOT assume two people or entities are the same just because they share a name across documents."
)

st.set_page_config(
    page_title="Hybrid GraphRAG",
    page_icon="🕸️",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files: list[dict] = []
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

MODALITY_COLORS = {
    "text":      "#2563eb",
    "table":     "#d97706",
    "image":     "#7c3aed",
    "formula":   "#059669",
    "algorithm": "#dc2626",
}


def _modality_badge(modality: str) -> str:
    color = MODALITY_COLORS.get(modality, "#6b7280")
    return (
        f'<span style="background:{color};color:white;padding:1px 7px;'
        f'border-radius:4px;font-size:0.72rem;">{modality}</span>'
    )


def _grounding_score(answer: str, sources: list[dict], graph_context: str | None = None) -> int:
    """Return 0-100 grounding score: % of answer words (>4 chars) found in sources.

    A rough proxy for hallucination — low score means the answer contains
    many words/names not present in any retrieved chunk.
    Includes both text and caption (table data) from sources, plus graph context.
    """
    parts = []
    for s in sources:
        parts.append(s.get("text", "") or "")
        parts.append(s.get("caption", "") or "")  # tables store full data in caption
    if graph_context:
        parts.append(graph_context)
    source_text = " ".join(parts).lower()
    words = [w for w in re.findall(r"[a-zA-Z0-9]+", answer.lower()) if len(w) > 4]
    if not words:
        return 100
    matched = sum(1 for w in words if w in source_text)
    return int((matched / len(words)) * 100)


def _grounding_badge(score: int) -> str:
    if score >= 70:
        color, label, icon = "#16a34a", "Well grounded", "✓"
    elif score >= 40:
        color, label, icon = "#d97706", "Partially grounded", "⚠"
    else:
        color, label, icon = "#dc2626", "Low grounding — verify manually", "✗"
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:5px;font-size:0.8rem;">{icon} {label} ({score}%)</span>'
    )


def _ingest_file(uploaded_file) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/ingest/file",
            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
            data={"caption": "true", "max_chunk_tokens": "512"},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API server. Run: `uv run python scripts/serve.py`")
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
    return None


def _generate(query: str, top_k: int, top_n: int, rerank: bool) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}/generate",
            json={
                "query": query,
                "top_k": top_k,
                "top_n": top_n,
                "rerank": rerank,
                "filter_modality": None,
                "system_prompt": _SYSTEM_PROMPT,
                "max_tokens": 1024,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the API server. Run: `uv run python scripts/serve.py`")
    except Exception as e:
        st.error(f"Generation failed: {e}")
    return None


def _render_answer(answer: str, sources: list[dict], graph_context: str | None) -> None:
    """Render answer + grounding badge + expandable sources + graph context."""
    score = _grounding_score(answer, sources, graph_context)
    st.markdown(_grounding_badge(score), unsafe_allow_html=True)

    if score < 40:
        st.warning(
            "The answer may contain information not found in the retrieved chunks. "
            "Check the Sources below and consider rephrasing your question or increasing top_k.",
            icon="⚠️",
        )

    source_files = list({s.get("source_file", "") for s in sources if s.get("source_file")})
    if len(source_files) > 1:
        st.warning(
            f"This answer draws from **{len(source_files)} different documents**: "
            + ", ".join(f"`{f}`" for f in source_files)
            + ". Facts are kept separate per document in the answer, but verify carefully.",
            icon="📂",
        )

    st.markdown(answer)

    if graph_context:
        with st.expander("Knowledge graph context"):
            st.code(graph_context, language=None)

    if sources:
        with st.expander(f"Sources ({len(sources)} chunks) — click to verify"):
            for src in sources:
                badge = _modality_badge(src.get("modality", "text"))
                score_val = src.get("rerank_score")
                score_str = f"score: {score_val:.3f}" if score_val else ""
                st.markdown(
                    f"{badge} &nbsp; **{src.get('source_file', '')}** "
                    f"p.{src.get('page', '?')} &nbsp; {score_str}",
                    unsafe_allow_html=True,
                )
                if src.get("text"):
                    st.caption(src["text"][:300])


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🕸️ Hybrid GraphRAG")
    st.caption("Multi-modal RAG + Knowledge Graph")
    st.divider()

    st.subheader("Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        if st.button("Ingest", type="primary", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}…"):
                result = _ingest_file(uploaded)
            if result:
                already = [f["source_file"] for f in st.session_state.ingested_files]
                if result["source_file"] not in already:
                    st.session_state.ingested_files.append(result)
                st.success(
                    f"Done — {result['chunks_upserted']} chunks  "
                    f"({', '.join(f'{v} {k}' for k, v in result['modality_counts'].items())})"
                )

    if st.session_state.ingested_files:
        st.divider()
        st.subheader("Ingested files")
        for f in st.session_state.ingested_files:
            counts = ", ".join(f"{v} {k}" for k, v in f["modality_counts"].items())
            st.markdown(f"📄 **{f['source_file']}**  \n`{counts}`")

    st.divider()
    st.subheader("Search settings")
    top_k = st.slider("Candidates (top_k)", 5, 50, 20)
    top_n = st.slider("Final results (top_n)", 1, 10, 5)
    rerank = st.toggle("Rerank", value=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("Ask your documents")

if not st.session_state.ingested_files:
    st.info(
        "No files uploaded in this session. "
        "You can still ask questions if documents were ingested previously.",
        icon="ℹ️",
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            meta = msg.get("meta", {})
            _render_answer(msg["content"], meta.get("sources", []), meta.get("graph_context"))

# Chat input
query = st.chat_input("Ask a question about your documents…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = _generate(query, top_k, top_n, rerank)

        if result:
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            graph_context = result.get("graph_context")

            _render_answer(answer, sources, graph_context)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "meta": {"graph_context": graph_context, "sources": sources},
            })
