"""
Real Neo4j Graph-RAG retrieval test.

Uses:
    - Real Neo4j database
    - Real GraphRetriever
    - Real graph data
    - Real RetrievalRequest

Read-only test.
No graph mutation.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv



from app.retrieval.graph_rag.graph_retriever import (
    GraphRetriever,
    Neo4jConfig,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)
load_dotenv()

def _neo4j_config() -> Neo4jConfig:
    """
    Build Neo4j configuration from environment.
    """

    return Neo4jConfig(
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


@pytest.fixture(scope="module")
def graph_retriever() -> GraphRetriever:
    """
    Create a real Neo4j GraphRetriever.
    """

    config = _neo4j_config()

    try:
        retriever = GraphRetriever(
            config=config,
            max_hops=2,
        )

    except Exception as exc:
        pytest.fail(
            f"Unable to initialize Neo4j GraphRetriever: "
            f"{exc}"
        )

    if not retriever.verify_connection():
        retriever.close()

        pytest.fail(
            "Neo4j is not reachable."
        )

    yield retriever

    retriever.close()


def test_real_neo4j_refund_policy_retrieval(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify graph retrieval against the real Neo4j graph.
    """

    request = RetrievalRequest.model_validate(
        {
            "query": "refund policy",
            "top_k": 5,
             "collection_name": os.getenv(
              "QDRANT_COLLECTION_NAME",
              "supportmind_retrieval_pdf_test",
        ),
        }
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results, (
        "Neo4j returned no graph retrieval results "
        "for 'refund policy'."
    )

    assert len(results) <= 5

    # ------------------------------------------------------
    # Ranking.
    # ------------------------------------------------------

    assert [
        result.rank
        for result in results
    ] == list(
        range(
            1,
            len(results) + 1,
        )
    )

    # ------------------------------------------------------
    # Graph retrieval source.
    # ------------------------------------------------------

    assert all(
        result.source.name.lower() == "graph"
        for result in results
    )

    # ------------------------------------------------------
    # Graph scoring metadata.
    # ------------------------------------------------------

    assert all(
        "graph_score"
        in result.retrieval_scores
        for result in results
    )

    assert all(
        "matched_tokens"
        in result.retrieval_scores
        for result in results
    )

    assert all(
        "relationship_count"
        in result.retrieval_scores
        for result in results
    )

    # ------------------------------------------------------
    # Actual graph evidence.
    # ------------------------------------------------------

    assert any(
        result.retrieval_scores[
            "matched_tokens"
        ] > 0
        for result in results
    )

    # ------------------------------------------------------
    # Every returned graph result should have usable
    # identity information.
    # ------------------------------------------------------

    assert all(
        result.id
        for result in results
    )

    assert all(
        result.node_id
        for result in results
    )

    assert all(
        result.document_id
        for result in results
    )

    # ------------------------------------------------------
    # Graph traversal evidence should exist for at least
    # one result if your populated graph contains
    # relationships.
    # ------------------------------------------------------

    assert any(
        result.retrieval_scores[
            "relationship_count"
        ] > 0
        for result in results
    ), (
        "Neo4j returned lexical matches but no relationship "
        "evidence. This may indicate that the real graph "
        "does not contain connected policy/entity nodes."
    )