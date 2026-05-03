"""Extract named entities and relationships from document chunks using an LLM."""
from __future__ import annotations

import json
import logging
import uuid

from doc_parser.chunker import Chunk
from doc_parser.config import get_settings, make_sync_llm_client
from doc_parser.ingestion.graph_models import GraphComponents, Node, Relationship

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """
You are a knowledge graph extraction engine.
Given the following document text, extract ALL named entities and relationships between them.

Return ONLY valid JSON matching this exact schema:
{{
  "nodes": [
    {{"id": "<uuid>", "name": "<entity name>", "type": "<Person|Location|Project|Company|Department|Skill|Concept>", "properties": {{}}}}
  ],
  "relationships": [
    {{"source_id": "<node id>", "target_id": "<node id>", "type": "<RELATIONSHIP_TYPE>", "properties": {{}}}}
  ]
}}

Rules:
- Generate a unique UUID for each node
- Use UPPER_SNAKE_CASE for relationship types (e.g. WORKS_AT, RELATED_TO, PART_OF)
- Only extract relationships explicitly stated in the text
- Return ONLY the JSON object, no explanation

Document text (source: {source_file}):
{text}
"""


def _to_valid_uuid(id_str: str) -> str:
    """Normalize LLM-generated IDs that may not be valid UUIDs."""
    try:
        uuid.UUID(id_str)
        return id_str
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(id_str)))


def extract_graph_components(
    chunks: list[Chunk],
    model: str | None = None,
) -> GraphComponents:
    """Extract entities and relationships from a list of enriched chunks.

    Joins all chunk texts into one document and calls the LLM once per
    source file. Returns a GraphComponents with normalized UUIDs.
    Uses the provider configured in LLM_PROVIDER (.env).
    """
    if not chunks:
        return GraphComponents()

    source_file = chunks[0].source_file
    full_text = "\n\n".join(
        f"[page {c.page}] {c.text}" for c in chunks if c.text
    )

    if not full_text.strip():
        return GraphComponents()

    settings = get_settings()
    if model is None:
        model = settings.openai_llm_model
    client = make_sync_llm_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": _EXTRACTION_PROMPT.format(
                    source_file=source_file,
                    text=full_text[:12_000],  # guard against token limits
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences that some LLMs wrap around JSON output
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip().rstrip("`").strip()
        raw = json.loads(content)
    except Exception:
        logger.exception("Graph extraction failed for %s", source_file)
        return GraphComponents()

    components = GraphComponents(**raw)

    # Normalize IDs — LLM sometimes generates placeholder UUIDs with invalid hex
    id_map = {node.id: _to_valid_uuid(node.id) for node in components.nodes}
    for node in components.nodes:
        node.id = id_map[node.id]
        node.properties["source_file"] = source_file  # tag every node with its origin
    for rel in components.relationships:
        rel.source_id = id_map.get(rel.source_id, _to_valid_uuid(rel.source_id))
        rel.target_id = id_map.get(rel.target_id, _to_valid_uuid(rel.target_id))

    logger.info(
        "Extracted %d nodes, %d relationships from %s",
        len(components.nodes), len(components.relationships), source_file,
    )
    return components
