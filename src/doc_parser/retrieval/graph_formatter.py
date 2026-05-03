"""Format Neo4j graph records into a readable context string for the LLM."""
from __future__ import annotations


def format_graph_context(graph_data: list[dict]) -> str:
    """Convert raw Neo4j records into a de-duplicated, human-readable fact list."""
    if not graph_data:
        return ""

    lines = ["=== Knowledge Graph Context ==="]
    seen: set[str] = set()

    for row in graph_data:
        if not row.get("relationship_type"):
            continue
        fact = (
            f"{row['entity_name']} ({row['entity_type']}) "
            f"--[{row['relationship_type']}]--> "
            f"{row['related_name']} ({row['related_type']})"
        )
        if fact not in seen:
            seen.add(fact)
            src = row.get("source_file", "")
            lines.append(f"- {fact}  [source: {src}]")

    return "\n".join(lines) if len(lines) > 1 else ""
