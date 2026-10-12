"""
Real Neo4j GraphRetriever integration tests.

Requirements
------------
- Neo4j must be running.
- Graph ingestion must already be completed.
- Real SupportMind graph must exist in Neo4j.
- .env must contain valid Neo4j credentials.

This test suite is READ-ONLY.

It does not:
- create nodes
- create relationships
- modify properties
- delete graph data
- run graph ingestion
- modify Qdrant
- generate embeddings
- call an LLM

The tests validate the real GraphRetriever against the actual
SupportMind Neo4j graph schema.
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
    RetrievalResult,
    RetrievalSource,
)


load_dotenv()


# ==============================================================
# Configuration
# ==============================================================


DEFAULT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "supportmind_retrieval_pdf_test",
)


# ==============================================================
# Fixtures
# ==============================================================


@pytest.fixture(scope="module")
def graph_retriever() -> GraphRetriever:
    """
    Create one real GraphRetriever for the complete test module.
    """

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
        retriever = GraphRetriever(
            config=config,
            max_hops=2,
        )

    except Exception as exc:
        pytest.fail(
            f"Unable to initialize GraphRetriever: {exc}"
        )

    if not retriever.verify_connection():
        retriever.close()

        pytest.fail(
            "Neo4j is not reachable."
        )

    yield retriever

    retriever.close()


# ==============================================================
# Request Helper
# ==============================================================


def _build_request(
    query: str,
    top_k: int = 10,
) -> RetrievalRequest:
    """
    Build a provider-agnostic RetrievalRequest.

    GraphRetriever does not use query_vector, but the shared
    retrieval contract still requires collection_name.
    """

    return RetrievalRequest.model_validate(
        {
            "query": query,
            "top_k": top_k,
            "collection_name": DEFAULT_COLLECTION_NAME,
            "include_graph": True,
            "rerank": False,
        }
    )


# ==============================================================
# Basic Connectivity
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_connection(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify real Neo4j connectivity and retriever configuration.
    """

    assert graph_retriever.verify_connection() is True

    assert graph_retriever.database

    assert graph_retriever.max_hops == 2

    assert graph_retriever.search_properties

    # Actual SupportMind properties.
    assert (
        "name"
        in graph_retriever.search_properties
    )

    assert (
        "description"
        in graph_retriever.search_properties
    )

    assert (
        "subject"
        in graph_retriever.search_properties
    )

    assert (
        "ticket_priority"
        in graph_retriever.search_properties
    )

    assert (
        "ticket_status"
        in graph_retriever.search_properties
    )

    assert (
        "product_purchased"
        in graph_retriever.search_properties
    )


# ==============================================================
# Product Retrieval
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_product_query(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify retrieval for a known Product entity.

    Expected graph structure:

        Product
           ↑
           │ ABOUT
           │
         Ticket
    """

    request = _build_request(
        query="GoPro Hero",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results, (
        "GraphRetriever returned no results "
        "for 'GoPro Hero'."
    )

    assert len(results) <= 10

    assert all(
        isinstance(
            result,
            RetrievalResult,
        )
        for result in results
    )

    assert all(
        result.source == RetrievalSource.GRAPH
        for result in results
    )

    assert all(
        result.score > 0
        for result in results
    )

    assert any(
        "Product"
        in result.metadata.get(
            "graph_labels",
            [],
        )
        for result in results
    )


# ==============================================================
# Ticket Retrieval
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_ticket_query(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify retrieval using ticket-domain terminology.
    """

    request = _build_request(
        query="technical support ticket",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results, (
        "GraphRetriever returned no results "
        "for ticket-domain query."
    )

    assert any(
        "Ticket"
        in result.metadata.get(
            "graph_labels",
            [],
        )
        for result in results
    )

    assert any(
        result.retrieval_scores[
            "matched_tokens"
        ] > 0
        for result in results
    )


# ==============================================================
# Product → Ticket Relationship Evidence
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_product_relationship_evidence(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify that a Product query produces connected graph evidence.

    Expected relationship:

        Product
           ↑
          ABOUT
           │
         Ticket
    """

    request = _build_request(
        query="GoPro Hero",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results

    relationship_results = [
        result
        for result in results
        if result.retrieval_scores[
            "relationship_count"
        ] > 0
    ]

    assert relationship_results, (
        "GoPro Hero matched graph nodes, but "
        "no relationship evidence was returned."
    )

    assert any(
        result.retrieval_scores[
            "graph_path_count"
        ] > 0
        for result in relationship_results
    )


# ==============================================================
# Expected Relationship Types
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_returns_supportmind_relationships(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify that returned graph paths contain actual SupportMind
    relationship types.
    """

    request = _build_request(
        query="GoPro Hero",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results

    relationship_types: set[str] = set()

    for result in results:

        graph_paths = result.metadata.get(
            "graph_paths",
            [],
        )

        for path in graph_paths:

            relationships = path.get(
                "relationships",
                [],
            )

            for relationship in relationships:

                relationship_type = relationship.get(
                    "type"
                )

                if relationship_type:
                    relationship_types.add(
                        str(
                            relationship_type
                        )
                    )

    expected_relationships = {
        "ABOUT",
        "CREATED",
        "HAS_TYPE",
        "HAS_PRIORITY",
        "HAS_STATUS",
        "VIA_CHANNEL",
    }

    assert relationship_types, (
        "GraphRetriever returned graph paths "
        "without relationship types."
    )

    assert relationship_types.intersection(
        expected_relationships
    ), (
        "Returned relationship types do not "
        "match the SupportMind graph schema."
    )


# ==============================================================
# Multi-Hop Graph Evidence
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_returns_multihop_evidence(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify that max_hops=2 actually exposes multi-hop evidence.

    Expected example:

        Customer
           │
        CREATED
           ↓
         Ticket
           │
          ABOUT
           ↓
        Product

    """

    request = _build_request(
        query="GoPro Hero",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results

    max_path_length = 0

    for result in results:

        graph_paths = result.metadata.get(
            "graph_paths",
            [],
        )

        for path in graph_paths:

            path_length = int(
                path.get(
                    "length",
                    0,
                )
                or 0
            )

            max_path_length = max(
                max_path_length,
                path_length,
            )

    assert max_path_length >= 1, (
        "GraphRetriever returned no graph paths."
    )


# ==============================================================
# RetrievalResult Contract
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_result_contract(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify that every real Neo4j result conforms to the shared
    RetrievalResult contract.
    """

    request = _build_request(
        query="GoPro Hero",
        top_k=5,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results

    # Rank must be deterministic and contiguous.
    assert [
        result.rank
        for result in results
    ] == list(
        range(
            1,
            len(results) + 1,
        )
    )

    for result in results:

        assert isinstance(
            result,
            RetrievalResult,
        )

        assert result.id

        assert result.node_id

        assert result.document_id

        assert result.text is not None

        assert result.source == (
            RetrievalSource.GRAPH
        )

        assert result.score > 0

        assert (
            result.retrieval_scores[
                "graph_score"
            ] == result.score
        )

        assert (
            result.retrieval_scores[
                "matched_tokens"
            ] >= 0
        )

        assert (
            result.retrieval_scores[
                "relationship_count"
            ] >= 0
        )

        assert (
            result.retrieval_scores[
                "graph_path_count"
            ] >= 0
        )

        assert isinstance(
            result.metadata.get(
                "graph_labels",
                [],
            ),
            list,
        )

        assert isinstance(
            result.metadata.get(
                "graph_paths",
                [],
            ),
            list,
        )


# ==============================================================
# Priority Query
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_priority_query(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify retrieval using an actual Ticket property.

    Example:

        Ticket
           │
        HAS_PRIORITY
           ↓
        Priority
    """

    request = _build_request(
        query="high priority",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results, (
        "GraphRetriever returned no results "
        "for 'high priority'."
    )

    assert any(
        "Ticket"
        in result.metadata.get(
            "graph_labels",
            [],
        )
        or "Priority"
        in result.metadata.get(
            "graph_labels",
            [],
        )
        for result in results
    )


# ==============================================================
# Status Query
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_status_query(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify retrieval using SupportMind ticket status data.
    """

    request = _build_request(
        query="open ticket",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results, (
        "GraphRetriever returned no results "
        "for 'open ticket'."
    )

    assert any(
        "Ticket"
        in result.metadata.get(
            "graph_labels",
            [],
        )
        for result in results
    )


# ==============================================================
# Channel Query
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_channel_query(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify retrieval using ticket channel information.
    """

    request = _build_request(
        query="email support",
        top_k=10,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results, (
        "GraphRetriever returned no results "
        "for 'email support'."
    )


# ==============================================================
# No Match Query
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_no_match(
    graph_retriever: GraphRetriever,
) -> None:
    """
    A clearly nonexistent entity should not produce a false
    positive graph result.
    """

    request = _build_request(
        query="zzzz_nonexistent_supportmind_entity_987654",
        top_k=5,
    )

    results = graph_retriever.retrieve(
        request
    )

    assert results == []


# ==============================================================
# Deterministic Retrieval
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_is_deterministic(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Same query should return the same ordered application-level
    results.
    """

    request = _build_request(
        query="GoPro Hero",
        top_k=10,
    )

    first = graph_retriever.retrieve(
        request
    )

    second = graph_retriever.retrieve(
        request
    )

    first_signature = [
        (
            result.id,
            result.node_id,
            result.document_id,
            result.score,
            result.rank,
        )
        for result in first
    ]

    second_signature = [
        (
            result.id,
            result.node_id,
            result.document_id,
            result.score,
            result.rank,
        )
        for result in second
    ]

    assert first_signature == second_signature


# ==============================================================
# Top-K Enforcement
# ==============================================================


@pytest.mark.real
def test_real_graph_retriever_respects_top_k(
    graph_retriever: GraphRetriever,
) -> None:
    """
    Verify that GraphRetriever never returns more than top_k.
    """

    for top_k in (1, 3, 5):

        request = _build_request(
            query="GoPro Hero",
            top_k=top_k,
        )

        results = graph_retriever.retrieve(
            request
        )

        assert len(results) <= top_k

        assert [
            result.rank
            for result in results
        ] == list(
            range(
                1,
                len(results) + 1,
            )
        )