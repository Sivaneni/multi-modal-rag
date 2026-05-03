"""Verify that a source file ingested into Qdrant also has entities in Neo4j.

Usage:
    uv run python scripts/verify_graph.py sample_blood_urine_test.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_parser.config import get_settings
from qdrant_client import QdrantClient
from neo4j import GraphDatabase


def main() -> None:
    source_file = sys.argv[1] if len(sys.argv) > 1 else None
    settings = get_settings()

    # ── Qdrant ────────────────────────────────────────────────────────────────
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    qdrant = QdrantClient(url=settings.qdrant_url, api_key=api_key)

    from qdrant_client.models import FieldCondition, Filter, MatchValue

    if source_file:
        q_filter = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))])
        q_count = qdrant.count(collection_name=settings.qdrant_collection_name, count_filter=q_filter, exact=True).count
        sample = qdrant.scroll(
            collection_name=settings.qdrant_collection_name,
            scroll_filter=q_filter,
            limit=3,
            with_payload=True,
        )[0]
    else:
        q_count = qdrant.count(collection_name=settings.qdrant_collection_name, exact=True).count
        sample = qdrant.scroll(collection_name=settings.qdrant_collection_name, limit=3, with_payload=True)[0]

    print(f"\n{'='*60}")
    print(f"  QDRANT  —  collection: {settings.qdrant_collection_name}")
    print(f"{'='*60}")
    print(f"  Points for '{source_file or 'ALL'}': {q_count}")
    for pt in sample:
        p = pt.payload or {}
        print(f"  • chunk_id={p.get('chunk_id')}  page={p.get('page')}  modality={p.get('modality')}")

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    if not settings.neo4j_uri or not settings.neo4j_password:
        print("\n[SKIP] NEO4J_URI or NEO4J_PASSWORD not configured — skipping Neo4j check.")
        return

    password = settings.neo4j_password.get_secret_value()
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_username, password))

    with driver.session() as session:
        if source_file:
            n4j_count = session.run(
                "MATCH (n:Entity {source_file: $sf}) RETURN count(n) AS c", sf=source_file
            ).single()["c"]
            rel_count = session.run(
                "MATCH (a:Entity {source_file: $sf})-[r]->(b) RETURN count(r) AS c", sf=source_file
            ).single()["c"]
            entities = session.run(
                "MATCH (n:Entity {source_file: $sf}) RETURN n.name AS name, n.type AS type LIMIT 5",
                sf=source_file,
            ).data()
        else:
            n4j_count = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            entities = session.run("MATCH (n:Entity) RETURN n.name AS name, n.type AS type LIMIT 5").data()

    driver.close()

    print(f"\n{'='*60}")
    print(f"  NEO4J")
    print(f"{'='*60}")
    print(f"  Entity nodes for '{source_file or 'ALL'}': {n4j_count}")
    print(f"  Relationships                             : {rel_count}")
    for e in entities:
        print(f"  • {e['name']}  ({e['type']})")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if q_count > 0 and n4j_count > 0:
        print(f"  ✓  '{source_file}' is present in BOTH Qdrant ({q_count} chunks) and Neo4j ({n4j_count} entities)")
    elif q_count > 0 and n4j_count == 0:
        print(f"  ✗  '{source_file}' is in Qdrant but MISSING from Neo4j — re-run ingest with NEO4J_URI set")
    elif q_count == 0:
        print(f"  ✗  '{source_file}' not found in Qdrant — run ingest first")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
