
"""
test_graph_retriever.py

Standalone tests for SupportMind GraphRetriever.

These tests do not require a live Neo4j server.
Neo4j driver/session behavior is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.retrieval.graph_rag.graph_retriever import (
    GraphRetriever,
    Neo4jConfig,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalSource,
)


# ==============================================================
# Helpers
# ==============================================================


def make_config() -> Neo4jConfig:
    return Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="test-password",
        database="neo4j",
    )


def make_request(
    query: str = "payment failure",
    top_k: int = 5,
) -> RetrievalRequest:

    return RetrievalRequest(
        query=query,
        collection_name="supportmind_documents",
        top_k=top_k,
    )


def make_retriever(
    driver=None,
    **kwargs,
) -> GraphRetriever:

    if driver is None:
        driver = MagicMock()

    return GraphRetriever(
        driver=driver,
        config=make_config(),
        **kwargs,
    )


def make_record(
    *,
    node_id="node-1",
    document_id="doc-1",
    text="Payment failures may occur.",
    matched_tokens=2,
    relationship_count=3,
):

    record = MagicMock()

    values = {
        "neo4j_element_id": "4:abc:1",
        "labels": ["Document"],
        "node_properties": {
            "node_id": node_id,
            "document_id": document_id,
            "text": text,
            "title": "Payment Failure",
        },
        "matched_tokens": matched_tokens,
        "relationship_count": relationship_count,
        "paths": [],
    }

    record.get.side_effect = values.get

    return record


# ==============================================================
# Configuration
# ==============================================================


def test_neo4j_config_validates():

    config = make_config()

    config.validate()


def test_neo4j_config_rejects_empty_uri():

    config = Neo4jConfig(
        uri="",
        username="neo4j",
        password="password",
        database="neo4j",
    )

    with pytest.raises(ValueError):
        config.validate()


def test_neo4j_config_rejects_empty_username():

    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="",
        password="password",
        database="neo4j",
    )

    with pytest.raises(ValueError):
        config.validate()


def test_neo4j_config_rejects_empty_password():

    config = Neo4jConfig(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="",
        database="neo4j",
    )

    with pytest.raises(ValueError):
        config.validate()


# ==============================================================
# Constructor
# ==============================================================


def test_graph_retriever_accepts_valid_driver():

    driver = MagicMock()

    retriever = make_retriever(
        driver=driver
    )

    assert retriever.database == "neo4j"


def test_graph_retriever_rejects_invalid_max_hops():

    with pytest.raises(ValueError):

        make_retriever(
            max_hops=3
        )


def test_graph_retriever_rejects_boolean_max_hops():

    with pytest.raises(ValueError):

        make_retriever(
            max_hops=True
        )


def test_graph_retriever_accepts_one_hop():

    retriever = make_retriever(
        max_hops=1
    )

    assert retriever.max_hops == 1


def test_graph_retriever_accepts_two_hops():

    retriever = make_retriever(
        max_hops=2
    )

    assert retriever.max_hops == 2


# ==============================================================
# Search Properties
# ==============================================================


def test_graph_retriever_uses_default_search_properties():

    retriever = make_retriever()

    assert "name" in retriever.search_properties
    assert "text" in retriever.search_properties
    assert "symbol" in retriever.search_properties


def test_graph_retriever_accepts_custom_search_properties():

    retriever = make_retriever(
        search_properties=[
            "name",
            "symbol",
        ]
    )

    assert retriever.search_properties == (
        "name",
        "symbol",
    )


def test_graph_retriever_rejects_string_search_properties():

    with pytest.raises(ValueError):

        make_retriever(
            search_properties="name"
        )


def test_graph_retriever_rejects_empty_search_properties():

    with pytest.raises(ValueError):

        make_retriever(
            search_properties=[]
        )


def test_graph_retriever_rejects_invalid_property_name():

    with pytest.raises(ValueError):

        make_retriever(
            search_properties=[
                "name; MATCH (n) DETACH DELETE n"
            ]
        )


def test_graph_retriever_deduplicates_search_properties():

    retriever = make_retriever(
        search_properties=[
            "name",
            "name",
            "symbol",
        ]
    )

    assert retriever.search_properties == (
        "name",
        "symbol",
    )


# ==============================================================
# Tokenization
# ==============================================================


def test_tokenizer_is_case_insensitive():

    assert GraphRetriever._tokenize(
        "Payment FAILURE"
    ) == [
        "payment",
        "failure",
    ]


def test_tokenizer_removes_punctuation():

    assert GraphRetriever._tokenize(
        "payment-failure!"
    ) == [
        "payment",
        "failure",
    ]


def test_tokenizer_preserves_unicode_words():

    tokens = GraphRetriever._tokenize(
        "paiement échec"
    )

    assert "paiement" in tokens
    assert "échec" in tokens


def test_tokenizer_rejects_non_string():

    with pytest.raises(ValueError):

        GraphRetriever._tokenize(
            None
        )


# ==============================================================
# Request Validation
# ==============================================================


def test_retrieve_rejects_none_request():

    retriever = make_retriever()

    with pytest.raises(ValueError):

        retriever.retrieve(
            None
        )


def test_retrieve_rejects_invalid_request_type():

    retriever = make_retriever()

    with pytest.raises(ValueError):

        retriever.retrieve(
            "payment failure"
        )


def test_retrieve_returns_empty_for_tokenless_query():

    driver = MagicMock()

    retriever = make_retriever(
        driver=driver
    )

    request = make_request(
        query="!!! ???"
    )

    results = retriever.retrieve(
        request
    )

    assert results == []

    driver.session.assert_not_called()


# ==============================================================
# Graph Score
# ==============================================================


def test_graph_score_for_matching_tokens():

    score = GraphRetriever._calculate_graph_score(
        matched_tokens=2,
        relationship_count=0,
    )

    assert score == 2.0


def test_graph_score_includes_relationship_evidence():

    score = GraphRetriever._calculate_graph_score(
        matched_tokens=2,
        relationship_count=10,
    )

    assert score == 3.0


def test_graph_score_rejects_negative_token_count():

    with pytest.raises(ValueError):

        GraphRetriever._calculate_graph_score(
            matched_tokens=-1,
            relationship_count=0,
        )


def test_graph_score_rejects_negative_relationship_count():

    with pytest.raises(ValueError):

        GraphRetriever._calculate_graph_score(
            matched_tokens=1,
            relationship_count=-1,
        )


# ==============================================================
# Result Conversion
# ==============================================================


def test_convert_record_returns_retrieval_result():

    record = make_record()

    result = GraphRetriever._convert_record(
        record
    )

    assert result is not None
    assert result.id == "node-1"
    assert result.node_id == "node-1"
    assert result.document_id == "doc-1"
    assert result.text == "Payment failures may occur."
    assert result.source == RetrievalSource.GRAPH


def test_convert_record_preserves_metadata():

    record = make_record()

    result = GraphRetriever._convert_record(
        record
    )

    assert result is not None

    assert result.metadata["title"] == (
        "Payment Failure"
    )

    assert result.metadata["graph_labels"] == [
        "Document"
    ]


def test_convert_record_preserves_graph_scores():

    record = make_record(
        matched_tokens=3,
        relationship_count=5,
    )

    result = GraphRetriever._convert_record(
        record
    )

    assert result is not None

    assert result.retrieval_scores[
        "matched_tokens"
    ] == 3

    assert result.retrieval_scores[
        "relationship_count"
    ] == 5

    assert result.retrieval_scores[
        "graph_score"
    ] == 3.5


def test_convert_record_rejects_missing_element_id():

    record = MagicMock()

    values = {
        "neo4j_element_id": None,
        "node_properties": {},
        "matched_tokens": 1,
        "relationship_count": 0,
        "labels": [],
        "paths": [],
    }

    record.get.side_effect = values.get

    result = GraphRetriever._convert_record(
        record
    )

    assert result is None


# ==============================================================
# Retrieval
# ==============================================================


def test_retrieve_executes_neo4j_query():

    driver = MagicMock()

    session = MagicMock()

    result_cursor = [
        make_record()
    ]

    session.run.return_value = result_cursor

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    results = retriever.retrieve(
        make_request()
    )

    assert len(results) == 1

    session.run.assert_called_once()


def test_retrieve_assigns_graph_source():

    driver = MagicMock()

    session = MagicMock()

    session.run.return_value = [
        make_record()
    ]

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results[0].source == RetrievalSource.GRAPH


def test_retrieve_assigns_deterministic_rank():

    driver = MagicMock()

    session = MagicMock()

    session.run.return_value = [
        make_record(
            node_id="node-a",
            matched_tokens=2,
        ),
        make_record(
            node_id="node-b",
            matched_tokens=1,
        ),
    ]

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    results = retriever.retrieve(
        make_request(
            top_k=5
        )
    )

    assert results[0].rank == 1
    assert results[1].rank == 2


def test_retrieve_orders_by_graph_score():

    driver = MagicMock()

    session = MagicMock()

    session.run.return_value = [
        make_record(
            node_id="weak",
            matched_tokens=1,
            relationship_count=0,
        ),
        make_record(
            node_id="strong",
            matched_tokens=3,
            relationship_count=5,
        ),
    ]

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    results = retriever.retrieve(
        make_request(
            top_k=5
        )
    )

    assert results[0].id == "strong"
    assert results[1].id == "weak"


def test_retrieve_respects_top_k():

    driver = MagicMock()

    session = MagicMock()

    session.run.return_value = [
        make_record(
            node_id="node-1",
            matched_tokens=5,
        ),
        make_record(
            node_id="node-2",
            matched_tokens=4,
        ),
        make_record(
            node_id="node-3",
            matched_tokens=3,
        ),
    ]

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    results = retriever.retrieve(
        make_request(
            top_k=2
        )
    )

    assert len(results) == 2


def test_retrieve_allows_empty_neo4j_results():

    driver = MagicMock()

    session = MagicMock()

    session.run.return_value = []

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results == []


# ==============================================================
# Connection
# ==============================================================


def test_verify_connection_returns_true_when_neo4j_responds():

    driver = MagicMock()

    session = MagicMock()

    result = MagicMock()

    session.run.return_value = result

    driver.session.return_value.__enter__.return_value = (
        session
    )

    retriever = make_retriever(
        driver=driver
    )

    assert retriever.verify_connection() is True


def test_verify_connection_returns_false_when_neo4j_fails():

    driver = MagicMock()

    driver.session.side_effect = Exception(
        "Neo4j unavailable"
    )

    retriever = make_retriever(
        driver=driver
    )

    assert retriever.verify_connection() is False


# ==============================================================
# Driver Lifecycle
# ==============================================================


def test_close_closes_driver():

    driver = MagicMock()

    retriever = make_retriever(
        driver=driver
    )

    retriever.close()

    driver.close.assert_called_once()
