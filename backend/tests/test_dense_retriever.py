from unittest.mock import Mock

import pytest

from app.retrieval.dense.dense_retriever import DenseRetriever
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


def make_request() -> RetrievalRequest:
    return RetrievalRequest(
        query="payment failure",
        collection_name="supportmind_documents",
        top_k=3,
    )


def make_vector_result(
    result_id="node-1",
    score=0.95,
):
    result = Mock()

    result.id = result_id
    result.node_id = result_id
    result.document_id = "doc-1"
    result.text = "Payment failures may occur when..."
    result.score = score
    result.metadata = {
        "source": "payment_policy.pdf",
        "page": 4,
    }
    result.embedding_model = "BAAI/bge-small-en-v1.5"
    result.provider = "qdrant"
    result.collection_name = "supportmind_documents"

    return result


# ==========================================================
# Initialization
# ==========================================================


def test_dense_retriever_uses_injected_pipeline():
    pipeline = Mock()

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    assert retriever.pipeline is pipeline


# ==========================================================
# Retrieval
# ==========================================================


def test_retrieve_calls_pipeline_search():
    pipeline = Mock()

    pipeline.search.return_value = [
        make_vector_result(
            "node-1",
            0.95,
        ),
        make_vector_result(
            "node-2",
            0.87,
        ),
    ]

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    results = retriever.retrieve(
        request=request,
        query_vector=[0.1, 0.2, 0.3],
    )

    pipeline.search.assert_called_once_with(
        collection_name="supportmind_documents",
        query_vector=[0.1, 0.2, 0.3],
        top_k=3,
        filters=request.filters,
        provider_name=request.provider,
    )

    assert len(results) == 2


def test_retrieve_assigns_dense_source():
    pipeline = Mock()

    pipeline.search.return_value = [
        make_vector_result(),
    ]

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    results = retriever.retrieve(
        make_request(),
        [0.1, 0.2, 0.3],
    )

    assert results[0].source == "dense"


def test_retrieve_assigns_rank_in_order():
    pipeline = Mock()

    pipeline.search.return_value = [
        make_vector_result("node-1", 0.95),
        make_vector_result("node-2", 0.90),
        make_vector_result("node-3", 0.85),
    ]

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    results = retriever.retrieve(
        make_request(),
        [0.1, 0.2, 0.3],
    )

    assert [result.rank for result in results] == [
        1,
        2,
        3,
    ]


def test_retrieve_preserves_scores():
    pipeline = Mock()

    pipeline.search.return_value = [
        make_vector_result("node-1", 0.91),
    ]

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    results = retriever.retrieve(
        make_request(),
        [0.1, 0.2, 0.3],
    )

    assert results[0].score == 0.91


# ==========================================================
# Validation
# ==========================================================


def test_retrieve_rejects_none_request():
    retriever = DenseRetriever(
        pipeline=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="Retrieval request cannot be None",
    ):
        retriever.retrieve(
            None,
            [0.1, 0.2],
        )


def test_retrieve_rejects_invalid_request_type():
    retriever = DenseRetriever(
        pipeline=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="RetrievalRequest",
    ):
        retriever.retrieve(
            "invalid",
            [0.1, 0.2],
        )


def test_retrieve_rejects_none_query_vector():
    retriever = DenseRetriever(
        pipeline=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="Query vector cannot be None",
    ):
        retriever.retrieve(
            make_request(),
            None,
        )


def test_retrieve_rejects_empty_query_vector():
    retriever = DenseRetriever(
        pipeline=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="Query vector cannot be empty",
    ):
        retriever.retrieve(
            make_request(),
            [],
        )


def test_retrieve_rejects_non_list_query_vector():
    retriever = DenseRetriever(
        pipeline=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="provided as a list",
    ):
        retriever.retrieve(
            make_request(),
            (0.1, 0.2),
        )


def test_retrieve_rejects_non_numeric_query_vector():
    retriever = DenseRetriever(
        pipeline=Mock(),
    )

    with pytest.raises(
        ValueError,
        match="only numeric values",
    ):
        retriever.retrieve(
            make_request(),
            [0.1, "invalid"],
        )


# ==========================================================
# Empty result handling
# ==========================================================


def test_retrieve_returns_empty_list_when_no_matches():
    pipeline = Mock()

    pipeline.search.return_value = []

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    results = retriever.retrieve(
        make_request(),
        [0.1, 0.2, 0.3],
    )

    assert results == []