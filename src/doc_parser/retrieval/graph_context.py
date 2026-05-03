"""Fetch entity graph context from Neo4j for a set of retrieved source files."""
from __future__ import annotations

import logging

from neo4j import Driver

logger = logging.getLogger(__name__)

_CYPHER = """
MATCH (n:Entity)
WHERE n.source_file IN $source_files
OPTIONAL MATCH (n)-[r]-(m:Entity)
RETURN
    n.name        AS entity_name,
    n.type        AS entity_type,
    n.source_file AS source_file,
    type(r)       AS relationship_type,
    m.name        AS related_name,
    m.type        AS related_type
LIMIT 100
"""


def fetch_graph_context(source_files: list[str], driver: Driver) -> list[dict]:
    """Return all entities and their relationships for the given source files."""
    if not source_files:
        return []
    try:
        with driver.session() as session:
            result = session.run(_CYPHER, source_files=source_files)
            return [dict(record) for record in result]
    except Exception:
        logger.exception("Neo4j graph context fetch failed")
        return []
