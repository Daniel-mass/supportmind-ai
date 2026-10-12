"""
Tests for retrieval_models.py

These tests verify the retrieval contracts independently
of Qdrant, Neo4j, BM25, rerankers, or the retrieval service.

SupportMind AI
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResult,
    RetrievalSource,
)


# ==============================================================
# RetrievalRequest
# ==============================================================


def test_retrieval_request_valid():
    request = RetrievalRequest(
        query="What is the refund policy?",
        collection_name="supportmind_documents",
        top_k=5,
    )

    assert request.query == "What is the refund policy?"
    assert request.collection_name == "supportmind_documents"
    assert request.top_k == 5
    assert request.query_vector is None
    assert request.filters == {}
    assert request.include_graph is False
    assert request.rerank is False


def test_retrieval_request_strips_strings():
    request = RetrievalRequest(
        query="  What is the refund policy?  ",
        collection_name="  supportmind_documents  ",
    )

    assert request.query == "What is the refund policy?"
    assert request.collection_name == "supportmind_documents"


def test_retrieval_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(
            query="",
            collection_name="supportmind_documents",
        )


def test_retrieval_request_rejects_empty_collection():
    with pytest.raises(ValidationError):
        RetrievalRequest(
            query="refund policy",
            collection_name="",
        )


def test_retrieval_request_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        RetrievalRequest(
            query="refund policy",
            collection_name="supportmind_documents",
            top_k=0,
        )


def test_retrieval_request_accepts_vector():
    request = RetrievalRequest(
        query="refund policy",
        collection_name="supportmind_documents",
        query_vector=[0.1, 0.2, 0.3],
    )

    assert request.query_vector == [0.1, 0.2, 0.3]


# ==============================================================
# RetrievalResult
# ==============================================================


def test_retrieval_result_valid():
    result = RetrievalResult(
        id="chunk-001",
        node_id="node-001",
        document_id="doc-001",
        text="Customers may request a refund within 30 days.",
        score=0.91,
        rank=1,
        source=RetrievalSource.DENSE,
    )

    assert result.id == "chunk-001"
    assert result.node_id == "node-001"
    assert result.document_id == "doc-001"
    assert result.score == 0.91
    assert result.rank == 1
    assert result.source == RetrievalSource.DENSE
    assert isinstance(result.retrieved_at, datetime)


def test_retrieval_result_supports_additional_scores():
    result = RetrievalResult(
        id="chunk-001",
        node_id="node-001",
        document_id="doc-001",
        text="Refund policy text.",
        score=0.87,
        source=RetrievalSource.HYBRID,
        retrieval_scores={
            "dense_score": 0.82,
            "bm25_score": 7.3,
            "rrf_score": 0.87,
        },
    )

    assert result.retrieval_scores["dense_score"] == 0.82
    assert result.retrieval_scores["bm25_score"] == 7.3
    assert result.retrieval_scores["rrf_score"] == 0.87


def test_retrieval_result_supports_graph_source():
    result = RetrievalResult(
        id="chunk-001",
        node_id="node-001",
        document_id="doc-001",
        text="Graph retrieved text.",
        score=0.75,
        source=RetrievalSource.GRAPH,
    )

    assert result.source == RetrievalSource.GRAPH


def test_retrieval_result_supports_reranked_source():
    result = RetrievalResult(
        id="chunk-001",
        node_id="node-001",
        document_id="doc-001",
        text="Reranked text.",
        score=0.95,
        source=RetrievalSource.RERANKED,
    )

    assert result.source == RetrievalSource.RERANKED


def test_retrieval_result_rejects_invalid_rank():
    with pytest.raises(ValidationError):
        RetrievalResult(
            id="chunk-001",
            node_id="node-001",
            document_id="doc-001",
            text="text",
            score=0.5,
            rank=0,
            source=RetrievalSource.DENSE,
        )


def test_retrieval_result_rejects_empty_id():
    with pytest.raises(ValidationError):
        RetrievalResult(
            id="",
            node_id="node-001",
            document_id="doc-001",
            text="text",
            score=0.5,
            source=RetrievalSource.DENSE,
        )


# ==============================================================
# RetrievalResponse
# ==============================================================


def test_retrieval_response_valid():
    result = RetrievalResult(
        id="chunk-001",
        node_id="node-001",
        document_id="doc-001",
        text="Refund policy text.",
        score=0.91,
        rank=1,
        source=RetrievalSource.DENSE,
    )

    response = RetrievalResponse(
        query="refund policy",
        results=[result],
        retrieval_strategy=RetrievalSource.DENSE,
        latency_ms=12.5,
    )

    assert response.query == "refund policy"
    assert len(response.results) == 1
    assert response.total_results == 1
    assert response.retrieval_strategy == RetrievalSource.DENSE
    assert response.latency_ms == 12.5


def test_retrieval_response_empty_results():
    response = RetrievalResponse(
        query="unknown query",
        results=[],
        retrieval_strategy=RetrievalSource.DENSE,
    )

    assert response.results == []
    assert response.total_results == 0


def test_retrieval_response_total_results_is_automatic():
    results = [
        RetrievalResult(
            id=f"chunk-{index}",
            node_id=f"node-{index}",
            document_id="doc-001",
            text=f"text {index}",
            score=0.9 - index * 0.1,
            rank=index,
            source=RetrievalSource.DENSE,
        )
        for index in range(1, 4)
    ]

    response = RetrievalResponse(
        query="test",
        results=results,
        retrieval_strategy=RetrievalSource.DENSE,
        total_results=999,
    )

    assert response.total_results == 3


def test_retrieval_response_rejects_negative_latency():
    with pytest.raises(ValidationError):
        RetrievalResponse(
            query="test",
            retrieval_strategy=RetrievalSource.DENSE,
            latency_ms=-1,
        )


# ==============================================================
# Model Serialization
# ==============================================================


def test_models_are_json_serializable():
    result = RetrievalResult(
        id="chunk-001",
        node_id="node-001",
        document_id="doc-001",
        text="Refund policy text.",
        score=0.91,
        source=RetrievalSource.DENSE,
    )

    response = RetrievalResponse(
        query="refund policy",
        results=[result],
        retrieval_strategy=RetrievalSource.DENSE,
    )

    data = response.model_dump()

    assert data["query"] == "refund policy"
    assert len(data["results"]) == 1
    assert data["results"][0]["source"] == "dense"


# ==============================================================
# Unknown Fields
# ==============================================================


def test_retrieval_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RetrievalRequest(
            query="refund",
            collection_name="documents",
            unknown_field="should_fail",
        )


def test_retrieval_result_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RetrievalResult(
            id="chunk-001",
            node_id="node-001",
            document_id="doc-001",
            text="text",
            score=0.5,
            source=RetrievalSource.DENSE,
            unknown_field="should_fail",
        )