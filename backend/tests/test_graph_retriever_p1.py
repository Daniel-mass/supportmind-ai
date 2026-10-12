
"""
P1 tests for SupportMind AI GraphRetriever.

These tests verify the GraphRetriever boundary without requiring
a live Neo4j database.

Coverage
--------
- Neo4jConfig validation
- dependency injection
- max_hops validation
- search property validation
- request validation
- tokenization
- parameterized Neo4j retrieval
- query token/limit forwarding
- Neo4j record conversion
- graph scoring
- metadata preservation
- deterministic ranking
- top_k handling
- empty query handling
- connection verification
- driver closing
- application factory
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrieval.graph_rag.graph_retriever import (
    GraphRetriever,
    Neo4jConfig,
    create_graph_retriever,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
)


# ==============================================================
# Test Data Helpers
# ==============================================================


def make_config() -> Neo4jConfig:
    """Return a valid test Neo4j configuration."""

    return Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
        database="neo4j",
    )


def make_request(
    query: str = "refund policy",
    top_k: int = 5,
) -> RetrievalRequest:
    """Create a valid RetrievalRequest."""

    return RetrievalRequest(
        query=query,
        top_k=top_k,
        collection_name="supportmind",
    )


def make_driver():
    """
    Build a mocked Neo4j driver/session structure.

    Structure:

        driver
          └── session()
                └── context manager
                      └── session
    """

    driver = MagicMock()

    session = MagicMock()

    context_manager = MagicMock()

    context_manager.__enter__.return_value = session
    context_manager.__exit__.return_value = None

    driver.session.return_value = context_manager

    return driver, session


def make_record(
    *,
    element_id: str = "node-1",
    labels: list[str] | None = None,
    properties: dict | None = None,
    matched_tokens: int = 2,
    relationship_count: int = 3,
    paths: list | None = None,
):
    """Create a mocked Neo4j record."""

    record = MagicMock()

    values = {
        "neo4j_element_id": element_id,
        "labels": labels or ["Document"],
        "node_properties": properties
        or {
            "node_id": "node-1",
            "document_id": "doc-1",
            "text": "Refund policy information.",
            "title": "Refund Policy",
        },
        "matched_tokens": matched_tokens,
        "relationship_count": relationship_count,
        "paths": paths or [],
    }

    record.get.side_effect = values.get

    return record


# ==============================================================
# Neo4jConfig Tests
# ==============================================================


def test_neo4j_config_accepts_valid_data():
    config = make_config()

    config.validate()


def test_neo4j_config_rejects_empty_uri():
    config = Neo4jConfig(
        uri="",
        username="neo4j",
        password="password",
        database="neo4j",
    )

    with pytest.raises(
        ValueError,
        match="Neo4j URI must be a non-empty string",
    ):
        config.validate()


def test_neo4j_config_rejects_empty_username():
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="",
        password="password",
        database="neo4j",
    )

    with pytest.raises(
        ValueError,
        match="Neo4j username must be a non-empty string",
    ):
        config.validate()


def test_neo4j_config_rejects_empty_password():
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="",
        database="neo4j",
    )

    with pytest.raises(
        ValueError,
        match="Neo4j password must be a non-empty string",
    ):
        config.validate()


def test_neo4j_config_rejects_empty_database():
    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="",
    )

    with pytest.raises(
        ValueError,
        match="Neo4j database must be a non-empty string",
    ):
        config.validate()


# ==============================================================
# Constructor Tests
# ==============================================================


def test_graph_retriever_accepts_injected_driver():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    assert retriever._driver is driver


def test_graph_retriever_rejects_invalid_max_hops():
    driver, _ = make_driver()

    with pytest.raises(
        ValueError,
        match="max_hops must be either 1 or 2",
    ):
        GraphRetriever(
            driver=driver,
            config=make_config(),
            max_hops=3,
        )


def test_graph_retriever_rejects_boolean_max_hops():
    driver, _ = make_driver()

    with pytest.raises(
        ValueError,
        match="max_hops must be an integer",
    ):
        GraphRetriever(
            driver=driver,
            config=make_config(),
            max_hops=True,
        )


def test_graph_retriever_accepts_one_hop_configuration():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
        max_hops=1,
    )

    assert retriever.max_hops == 1


def test_graph_retriever_accepts_two_hop_configuration():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
        max_hops=2,
    )

    assert retriever.max_hops == 2


# ==============================================================
# Search Property Tests
# ==============================================================


def test_graph_retriever_uses_default_search_properties():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    assert (
        retriever.search_properties
        == GraphRetriever.DEFAULT_SEARCH_PROPERTIES
    )


def test_graph_retriever_accepts_custom_search_properties():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
        search_properties=[
            "name",
            "title",
            "custom_field",
        ],
    )

    assert retriever.search_properties == (
        "name",
        "title",
        "custom_field",
    )


def test_graph_retriever_deduplicates_search_properties():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
        search_properties=[
            "name",
            "title",
            "name",
        ],
    )

    assert retriever.search_properties == (
        "name",
        "title",
    )


def test_graph_retriever_rejects_string_search_properties():
    driver, _ = make_driver()

    with pytest.raises(
        ValueError,
        match="search_properties must be a sequence of strings",
    ):
        GraphRetriever(
            driver=driver,
            config=make_config(),
            search_properties="title",
        )


def test_graph_retriever_rejects_empty_search_properties():
    driver, _ = make_driver()

    with pytest.raises(
        ValueError,
        match="At least one search property is required",
    ):
        GraphRetriever(
            driver=driver,
            config=make_config(),
            search_properties=[],
        )


def test_graph_retriever_rejects_invalid_search_property_name():
    driver, _ = make_driver()

    with pytest.raises(
        ValueError,
        match="Invalid Neo4j property name",
    ):
        GraphRetriever(
            driver=driver,
            config=make_config(),
            search_properties=[
                "title-name",
            ],
        )


def test_graph_retriever_rejects_empty_search_property():
    driver, _ = make_driver()

    with pytest.raises(
        ValueError,
        match="Every search property must be a non-empty string",
    ):
        GraphRetriever(
            driver=driver,
            config=make_config(),
            search_properties=[
                "",
            ],
        )


# ==============================================================
# Property Tests
# ==============================================================


def test_database_property_returns_configured_database():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    assert retriever.database == "neo4j"


# ==============================================================
# Tokenization Tests
# ==============================================================


def test_tokenizer_is_case_insensitive():
    tokens = GraphRetriever._tokenize(
        "Refund POLICY"
    )

    assert tokens == [
        "refund",
        "policy",
    ]


def test_tokenizer_removes_punctuation():
    tokens = GraphRetriever._tokenize(
        "refund-policy!"
    )

    assert tokens == [
        "refund",
        "policy",
    ]


def test_tokenizer_preserves_unicode_words():
    tokens = GraphRetriever._tokenize(
        "réclamation politique"
    )

    assert tokens == [
        "réclamation",
        "politique",
    ]


def test_tokenizer_rejects_non_string_input():
    with pytest.raises(
        ValueError,
        match="Text must be a string",
    ):
        GraphRetriever._tokenize(
            123
        )


# ==============================================================
# Request Validation Tests
# ==============================================================


def test_retrieve_rejects_none_request():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    with pytest.raises(
        ValueError,
        match="Retrieval request cannot be None",
    ):
        retriever.retrieve(None)


def test_retrieve_rejects_invalid_request_type():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    with pytest.raises(
        ValueError,
        match="request must be a RetrievalRequest instance",
    ):
        retriever.retrieve(
            "refund policy"
        )


# ==============================================================
# Retrieval Tests
# ==============================================================


def test_retrieve_returns_empty_for_empty_tokenized_query():
    driver, session = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    request = make_request(
        query="!!!"
    )

    results = retriever.retrieve(
        request
    )

    assert results == []

    session.run.assert_not_called()


def test_retrieve_executes_neo4j_query():
    driver, session = make_driver()

    record = make_record()

    session.run.return_value = [
        record
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert len(results) == 1

    session.run.assert_called_once()


def test_retrieve_passes_tokenized_query_to_neo4j():
    driver, session = make_driver()

    session.run.return_value = [
        make_record()
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    retriever.retrieve(
        make_request(
            query="Refund POLICY"
        )
    )

    args, kwargs = session.run.call_args

    assert kwargs["query_tokens"] == [
        "refund",
        "policy",
    ]


def test_retrieve_passes_top_k_to_neo4j():
    driver, session = make_driver()

    session.run.return_value = [
        make_record()
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    retriever.retrieve(
        make_request(
            top_k=3
        )
    )

    args, kwargs = session.run.call_args

    assert kwargs["limit"] == 3


def test_retrieve_uses_configured_database():
    driver, session = make_driver()

    session.run.return_value = []

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    retriever.retrieve(
        make_request()
    )

    driver.session.assert_called_once_with(
        database="neo4j"
    )


def test_retrieve_returns_retrieval_results():
    driver, session = make_driver()

    session.run.return_value = [
        make_record()
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert all(
        isinstance(
            result,
            RetrievalResult,
        )
        for result in results
    )


def test_retrieve_assigns_graph_source():
    driver, session = make_driver()

    session.run.return_value = [
        make_record()
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results[0].source == RetrievalSource.GRAPH


def test_retrieve_assigns_deterministic_rank():
    driver, session = make_driver()

    session.run.return_value = [
        make_record(
            element_id="node-1",
            properties={
                "node_id": "node-1",
                "document_id": "doc-1",
                "text": "Refund policy",
            },
            matched_tokens=3,
            relationship_count=1,
        ),
        make_record(
            element_id="node-2",
            properties={
                "node_id": "node-2",
                "document_id": "doc-2",
                "text": "Payment policy",
            },
            matched_tokens=1,
            relationship_count=1,
        ),
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert [result.rank for result in results] == [
        1,
        2,
    ]


def test_retrieve_ranks_follow_graph_score():
    driver, session = make_driver()

    session.run.return_value = [
        make_record(
            element_id="node-low",
            properties={
                "node_id": "node-low",
                "document_id": "doc-low",
                "text": "Low relevance",
            },
            matched_tokens=1,
            relationship_count=0,
        ),
        make_record(
            element_id="node-high",
            properties={
                "node_id": "node-high",
                "document_id": "doc-high",
                "text": "High relevance",
            },
            matched_tokens=3,
            relationship_count=0,
        ),
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results[0].id == "node-high"
    assert results[1].id == "node-low"


def test_retrieve_respects_top_k():
    driver, session = make_driver()

    session.run.return_value = [
        make_record(
            element_id="node-1",
            properties={
                "node_id": "node-1",
                "document_id": "doc-1",
                "text": "One",
            },
            matched_tokens=3,
        ),
        make_record(
            element_id="node-2",
            properties={
                "node_id": "node-2",
                "document_id": "doc-2",
                "text": "Two",
            },
            matched_tokens=2,
        ),
        make_record(
            element_id="node-3",
            properties={
                "node_id": "node-3",
                "document_id": "doc-3",
                "text": "Three",
            },
            matched_tokens=1,
        ),
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request(
            top_k=2
        )
    )

    assert len(results) == 2


def test_retrieve_returns_empty_when_neo4j_returns_no_records():
    driver, session = make_driver()

    session.run.return_value = []

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results == []


# ==============================================================
# Result Conversion Tests
# ==============================================================


def test_convert_record_returns_retrieval_result():
    record = make_record()

    result = GraphRetriever._convert_record(
        record
    )

    assert isinstance(
        result,
        RetrievalResult,
    )


def test_convert_record_preserves_identity_fields():
    record = make_record(
        properties={
            "node_id": "node-42",
            "document_id": "doc-42",
            "text": "Refund policy",
        }
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.id == "node-42"
    assert result.node_id == "node-42"
    assert result.document_id == "doc-42"


def test_convert_record_preserves_text():
    record = make_record(
        properties={
            "node_id": "node-1",
            "document_id": "doc-1",
            "text": "Customers can request refunds.",
        }
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.text == (
        "Customers can request refunds."
    )


def test_convert_record_preserves_metadata():
    properties = {
        "node_id": "node-1",
        "document_id": "doc-1",
        "text": "Refund policy",
        "title": "Refund Policy",
        "category": "billing",
    }

    record = make_record(
        properties=properties
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.metadata["title"] == "Refund Policy"
    assert result.metadata["category"] == "billing"


def test_convert_record_adds_neo4j_metadata():
    record = make_record(
        element_id="neo4j-123"
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.metadata[
        "neo4j_element_id"
    ] == "neo4j-123"


def test_convert_record_adds_graph_labels():
    record = make_record(
        labels=[
            "Document",
            "Policy",
        ]
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.metadata[
        "graph_labels"
    ] == [
        "Document",
        "Policy",
    ]


def test_convert_record_adds_matched_token_metadata():
    record = make_record(
        matched_tokens=4
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.metadata[
        "matched_tokens"
    ] == 4


def test_convert_record_adds_relationship_metadata():
    record = make_record(
        relationship_count=7
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.metadata[
        "relationship_count"
    ] == 7


def test_convert_record_adds_graph_paths():
    paths = [
        [
            {
                "element_id": "node-1",
                "labels": ["Document"],
                "properties": {
                    "name": "Refund Policy"
                },
            }
        ]
    ]

    record = make_record(
        paths=paths
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result.metadata[
        "graph_paths"
    ] == paths


def test_convert_record_returns_none_without_neo4j_element_id():
    record = MagicMock()

    values = {
        "neo4j_element_id": None,
        "node_properties": {},
    }

    record.get.side_effect = values.get

    result = GraphRetriever._convert_record(
        record
    )

    assert result is None


# ==============================================================
# Graph Score Tests
# ==============================================================


def test_graph_score_uses_matched_tokens_and_relationships():
    score = GraphRetriever._calculate_graph_score(
        matched_tokens=3,
        relationship_count=5,
    )

    assert score == pytest.approx(
        3.5
    )


def test_graph_score_with_no_relationships():
    score = GraphRetriever._calculate_graph_score(
        matched_tokens=3,
        relationship_count=0,
    )

    assert score == 3.0


def test_graph_score_rejects_negative_matched_tokens():
    with pytest.raises(
        ValueError,
        match="matched_tokens cannot be negative",
    ):
        GraphRetriever._calculate_graph_score(
            matched_tokens=-1,
            relationship_count=0,
        )


def test_graph_score_rejects_negative_relationship_count():
    with pytest.raises(
        ValueError,
        match="relationship_count cannot be negative",
    ):
        GraphRetriever._calculate_graph_score(
            matched_tokens=1,
            relationship_count=-1,
        )


# ==============================================================
# Deterministic Ordering Tests
# ==============================================================


def test_retrieve_uses_node_id_as_deterministic_tie_breaker():
    driver, session = make_driver()

    session.run.return_value = [
        make_record(
            element_id="node-b",
            properties={
                "node_id": "node-b",
                "document_id": "doc-b",
                "text": "Policy B",
            },
            matched_tokens=2,
            relationship_count=0,
        ),
        make_record(
            element_id="node-a",
            properties={
                "node_id": "node-a",
                "document_id": "doc-a",
                "text": "Policy A",
            },
            matched_tokens=2,
            relationship_count=0,
        ),
    ]

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    results = retriever.retrieve(
        make_request()
    )

    assert [result.id for result in results] == [
        "node-a",
        "node-b",
    ]


# ==============================================================
# Connectivity Tests
# ==============================================================


def test_verify_connection_returns_true_when_neo4j_responds():
    driver, session = make_driver()

    session.run.return_value.consume.return_value = None

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    assert retriever.verify_connection() is True

    session.run.assert_called_once_with(
        "RETURN 1 AS ok"
    )


def test_verify_connection_returns_false_when_neo4j_fails():
    driver, session = make_driver()

    session.run.side_effect = RuntimeError(
        "Neo4j unavailable"
    )

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    assert retriever.verify_connection() is False


def test_close_closes_neo4j_driver():
    driver, _ = make_driver()

    retriever = GraphRetriever(
        driver=driver,
        config=make_config(),
    )

    retriever.close()

    driver.close.assert_called_once()


# ==============================================================
# Factory Tests
# ==============================================================


def test_create_graph_retriever_uses_config():
    driver, _ = make_driver()

    # The factory creates its own driver when no driver is supplied,
    # so this test only verifies the resulting configuration boundary.
    retriever = create_graph_retriever(
        config=make_config()
    )

    try:
        assert isinstance(
            retriever,
            GraphRetriever,
        )

        assert retriever.database == "neo4j"

    finally:
        retriever.close()
