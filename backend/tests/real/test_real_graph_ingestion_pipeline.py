"""
Real Neo4j graph ingestion pipeline test.

This test verifies:

FilesystemConnector
        ↓
CSVLoader
        ↓
ParserPipeline
        ↓
GraphIngestionPipeline
        ↓
GraphProjector
        ↓
Neo4jStore
        ↓
Neo4j

Uses the real cleaned SupportMind CSV dataset.

IMPORTANT
---------
This is a real integration test.

It:
- connects to real Neo4j
- reads the real CSV
- writes the graph
- verifies actual graph statistics
- verifies actual relationships

It does NOT automatically clear Neo4j.

If a clean graph is required, explicitly clear the database
before running this test.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.retrieval.graph_rag.graph_ingestion_pipeline import (
    GraphIngestionPipeline,
)

from app.retrieval.graph_rag.neo4j_store import (
    Neo4jConfig,
    Neo4jStore,
)


# ============================================================================
# Environment
# ============================================================================

load_dotenv()


CSV_PATH = Path(
    r"C:\Users\danie\supportmind-ai\rawsupportminddata"
    r"\final clean Customer Support Ticket Dataset"
    r"\supportmind_final_clean_dataset.csv"
)


# ============================================================================
# Neo4j fixture
# ============================================================================


@pytest.fixture(scope="module")
def neo4j_store() -> Neo4jStore:

    config = Neo4jConfig(
        uri=os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        ),
        username=os.getenv(
            "NEO4J_USERNAME",
            "neo4j",
        ),
        password=os.getenv(
            "NEO4J_PASSWORD",
            "",
        ),
        database=os.getenv(
            "NEO4J_DATABASE",
            "neo4j",
        ),
    )

    try:

        store = Neo4jStore(
            config
        )

    except Exception as exc:

        pytest.fail(
            f"Unable to initialize Neo4jStore: {exc}"
        )

    if not store.verify_connection():

        store.close()

        pytest.fail(
            "Neo4j is not reachable."
        )

    yield store

    store.close()


# ============================================================================
# Test
# ============================================================================


@pytest.mark.real
def test_real_graph_ingestion_pipeline(
    neo4j_store: Neo4jStore,
) -> None:

    # ------------------------------------------------------------
    # Dataset validation
    # ------------------------------------------------------------

    assert CSV_PATH.exists(), (
        f"Dataset does not exist: "
        f"{CSV_PATH}"
    )

    assert CSV_PATH.is_file()

    # ------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------

    pipeline = GraphIngestionPipeline(
        neo4j_store=neo4j_store,
        batch_size=500,
    )

    result = pipeline.run(
        str(CSV_PATH.parent)
    )

    # ------------------------------------------------------------
    # Pipeline success
    # ------------------------------------------------------------

    assert result.success, (
        "Graph ingestion pipeline failed.\n"
        f"Errors: {result.errors}"
    )

    assert (
        result.csv_files_processed >= 1
    )

    assert (
        result.csv_files_failed == 0
    )

    assert (
        result.rows_read > 0
    )

    assert (
        result.rows_projected > 0
    )

    # ------------------------------------------------------------
    # Dataset-specific expectation
    #
    # The cleaned dataset contains 8469 rows.
    # ------------------------------------------------------------

    assert (
        result.rows_read == 8469
    )

    assert (
        result.rows_projected
        == 8469
    )

    assert (
        result.rows_skipped == 0
    )

    # ------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------

    graph_stats = (
        neo4j_store.get_stats()
    )

    assert (
        graph_stats.node_count > 0
    )

    assert (
        graph_stats.relationship_count > 0
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    label_counts = (
        neo4j_store.get_label_counts()
    )

    expected_labels = {
        "Customer",
        "Ticket",
        "Product",
        "TicketType",
        "Priority",
        "Status",
        "Channel",
    }

    assert expected_labels.issubset(
        label_counts.keys()
    )

    # ------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------

    relationship_counts = (
        neo4j_store
        .get_relationship_counts()
    )

    expected_relationships = {
        "CREATED",
        "ABOUT",
        "HAS_TYPE",
        "HAS_PRIORITY",
        "HAS_STATUS",
        "VIA_CHANNEL",
    }

    assert expected_relationships.issubset(
        relationship_counts.keys()
    )

    # ------------------------------------------------------------
    # Actual connected graph
    # ------------------------------------------------------------

    subgraph = (
        neo4j_store
        .get_visualization_subgraph(
            limit=20
        )
    )

    assert subgraph

    assert all(
        item["source_id"]
        for item in subgraph
    )

    assert all(
        item["target_id"]
        for item in subgraph
    )

    assert all(
        item["relationship_type"]
        for item in subgraph
    )

    # ------------------------------------------------------------
    # Meaningful multi-hop relationship
    # ------------------------------------------------------------

    rows = neo4j_store.run_read(
        """
        MATCH path =
            (c:Customer)
            -[:CREATED]->
            (t:Ticket)
            -[:ABOUT]->
            (p:Product)

        RETURN
            c.node_id AS customer_id,
            t.node_id AS ticket_id,
            p.node_id AS product_id,
            p.name AS product

        LIMIT 10
        """
    )

    assert rows

    for row in rows:

        assert row["customer_id"]
        assert row["ticket_id"]
        assert row["product_id"]
        assert row["product"]

    # ------------------------------------------------------------
    # Ticket -> Priority relationship
    # ------------------------------------------------------------

    priority_rows = neo4j_store.run_read(
        """
        MATCH
            (t:Ticket)
            -[:HAS_PRIORITY]->
            (p:Priority)

        RETURN
            t.ticket_id AS ticket_id,
            p.name AS priority

        LIMIT 10
        """
    )

    assert priority_rows

    # ------------------------------------------------------------
    # Ticket -> Status relationship
    # ------------------------------------------------------------

    status_rows = neo4j_store.run_read(
        """
        MATCH
            (t:Ticket)
            -[:HAS_STATUS]->
            (s:Status)

        RETURN
            t.ticket_id AS ticket_id,
            s.name AS status

        LIMIT 10
        """
    )

    assert status_rows

    # ------------------------------------------------------------
    # Useful diagnostic output
    # ------------------------------------------------------------

    print("\n========== GRAPH INGESTION ==========")

    print(
        f"Rows read: "
        f"{result.rows_read}"
    )

    print(
        f"Rows projected: "
        f"{result.rows_projected}"
    )

    print(
        f"Rows skipped: "
        f"{result.rows_skipped}"
    )

    print(
        f"Nodes created: "
        f"{result.nodes_created}"
    )

    print(
        f"Nodes matched: "
        f"{result.nodes_matched}"
    )

    print(
        f"Relationships created: "
        f"{result.relationships_created}"
    )

    print(
        f"Relationships matched: "
        f"{result.relationships_matched}"
    )

    print(
        "\nNeo4j labels:"
    )

    for label, count in label_counts.items():

        print(
            f"  {label}: {count}"
        )

    print(
        "\nNeo4j relationships:"
    )

    for relationship, count in (
        relationship_counts.items()
    ):

        print(
            f"  {relationship}: {count}"
        )

    print(
        "\nExample multi-hop paths:"
    )

    for row in rows[:5]:

        print(
            f"  {row['customer_id']} "
            f"-> {row['ticket_id']} "
            f"-> {row['product']}"
        )

    print(
        "\n====================================="
    )