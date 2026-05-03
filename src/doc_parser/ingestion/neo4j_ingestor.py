"""Write extracted graph components to Neo4j."""
from __future__ import annotations

import logging

from neo4j import Driver, GraphDatabase

from doc_parser.ingestion.graph_models import GraphComponents

logger = logging.getLogger(__name__)


def get_neo4j_driver(uri: str, username: str, password: str) -> Driver:
    return GraphDatabase.driver(uri, auth=(username, password))


def ingest_to_neo4j(components: GraphComponents, driver: Driver) -> None:
    """MERGE nodes and relationships into Neo4j.

    Uses MERGE (not CREATE) so re-ingestion is idempotent.
    """
    with driver.session() as session:
        for node in components.nodes:
            session.run(
                """
                MERGE (n:Entity {id: $id})
                SET n.name = $name, n.type = $type
                SET n += $properties
                """,
                id=node.id,
                name=node.name,
                type=node.type,
                properties=node.properties,
            )

        for rel in components.relationships:
            session.run(
                """
                MATCH (a:Entity {id: $source_id})
                MATCH (b:Entity {id: $target_id})
                MERGE (a)-[r:`"""
                + rel.type
                + """`]->(b)
                SET r += $properties
                """,
                source_id=rel.source_id,
                target_id=rel.target_id,
                properties=rel.properties,
            )

    logger.info(
        "Neo4j: merged %d nodes, %d relationships",
        len(components.nodes), len(components.relationships),
    )
