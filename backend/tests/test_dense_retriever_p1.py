"""
Standalone tests for DenseRetriever.

Test scope
----------
1. Input validation
2. Query-vector validation
3. VectorFilter conversion
4. VectorPipeline delegation
5. VectorSearchResult -> RetrievalResult conversion
6. Deterministic ranking
7. Score preservation
8. Metadata preservation
9. Empty-result handling
10. Original-result immutability

Qdrant is intentionally NOT required here.
The real Qdrant connection belongs to the retrieval integration test.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
)

from app.retrieval.dense.dense_retriever import (
    DenseRetriever,
)

from app.vector_store.vector_filters import (
    FilterOperator,
    VectorFilter,
)

from app.vector_store.vector_search_result import (
    VectorSearchResult,
)


# ============================================================
# Helpers
# ============================================================


def make_request(
    *,
    query: str = "refund not received",
    collection_name: str = "supportmind_documents",
    top_k: int = 5,
    provider: str | None = None,
    filters=None,
) -> RetrievalRequest:
    """
    Create a valid RetrievalRequest using the project's
    existing retrieval contract.
    """

    data = {
        "query": query,
        "collection_name": collection_name,
        "top_k": top_k,
    }

    if provider is not None:
        data["provider"] = provider

    if filters is not None:
        data["filters"] = filters

    return RetrievalRequest(**data)


def make_vector_result(
    *,
    result_id: str = "result-1",
    node_id: str = "node-1",
    document_id: str = "doc-1",
    text: str = "Refunds are processed within five business days.",
    score: float = 0.91,
    metadata=None,
    embedding_model: str = "test-model",
    provider: str = "qdrant",
    collection_name: str = "supportmind_documents",
) -> VectorSearchResult:

    return VectorSearchResult(
        id=result_id,
        node_id=node_id,
        document_id=document_id,
        text=text,
        score=score,
        metadata=metadata or {},
        embedding_model=embedding_model,
        provider=provider,
        collection_name=collection_name,
    )


def make_pipeline(
    results=None,
):
    """
    Create a mocked VectorPipeline.
    """

    pipeline = Mock()

    pipeline.search.return_value = (
        results if results is not None else []
    )

    return pipeline


# ============================================================
# Initialization
# ============================================================


def test_dense_retriever_accepts_injected_pipeline():
    pipeline = make_pipeline()

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    assert retriever.pipeline is pipeline


def test_dense_retriever_uses_application_pipeline_when_omitted(
    monkeypatch,
):
    """
    Verify that an application-level pipeline can be used
    when no pipeline is explicitly injected.
    """

    pipeline = make_pipeline()

    monkeypatch.setattr(
        "app.retrieval.dense.dense_retriever.vector_pipeline",
        pipeline,
    )

    retriever = DenseRetriever()

    assert retriever.pipeline is pipeline


# ============================================================
# Request validation
# ============================================================


def test_retrieve_rejects_none_request():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    with pytest.raises(
        ValueError,
        match="Retrieval request cannot be None",
    ):
        retriever.retrieve(
            None,
            [0.1, 0.2, 0.3],
        )


def test_retrieve_rejects_invalid_request_type():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    with pytest.raises(
        ValueError,
        match="request must be a RetrievalRequest",
    ):
        retriever.retrieve(
            "invalid request",
            [0.1, 0.2],
        )


# ============================================================
# Query-vector validation
# ============================================================


def test_retrieve_rejects_none_query_vector():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector cannot be None",
    ):
        retriever.retrieve(
            request,
            None,
        )


def test_retrieve_rejects_non_list_query_vector():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector must be provided as a list",
    ):
        retriever.retrieve(
            request,
            "not-a-vector",
        )


def test_retrieve_rejects_empty_query_vector():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector cannot be empty",
    ):
        retriever.retrieve(
            request,
            [],
        )


def test_retrieve_rejects_non_numeric_query_vector():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector must contain only numeric values",
    ):
        retriever.retrieve(
            request,
            [0.1, "invalid", 0.3],
        )


def test_retrieve_rejects_boolean_query_vector_value():
    retriever = DenseRetriever(
        pipeline=make_pipeline(),
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector must contain only numeric values",
    ):
        retriever.retrieve(
            request,
            [0.1, True, 0.3],
        )


def test_retrieve_accepts_integer_and_float_vector_values():
    pipeline = make_pipeline()

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    retriever.retrieve(
        request,
        [1, 0.5, -0.2],
    )

    pipeline.search.assert_called_once()


# ============================================================
# VectorFilter conversion
# ============================================================


def test_build_vector_filter_returns_none_for_none():
    assert (
        DenseRetriever._build_vector_filter(None)
        is None
    )


def test_build_vector_filter_returns_none_for_empty_dict():
    assert (
        DenseRetriever._build_vector_filter({})
        is None
    )


def test_build_vector_filter_accepts_existing_vector_filter():
    vector_filter = VectorFilter()

    result = DenseRetriever._build_vector_filter(
        vector_filter,
    )

    assert result is vector_filter


def test_build_vector_filter_converts_dictionary():
    filters = {
        "conditions": [
            {
                "field": "metadata.category",
                "operator": FilterOperator.EQUAL,
                "value": "billing",
            }
        ]
    }

    result = DenseRetriever._build_vector_filter(
        filters,
    )

    assert isinstance(
        result,
        VectorFilter,
    )

    assert len(result.conditions) == 1

    assert (
        result.conditions[0].field
        == "metadata.category"
    )

    assert (
        result.conditions[0].value
        == "billing"
    )


def test_build_vector_filter_rejects_invalid_filter_type():
    with pytest.raises(
        ValueError,
        match="dictionary or VectorFilter",
    ):
        DenseRetriever._build_vector_filter(
            "invalid",
        )


def test_build_vector_filter_rejects_invalid_filter_data():
    filters = {
        "conditions": [
            {
                "field": "",
                "operator": "invalid_operator",
                "value": "billing",
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match="Invalid retrieval filters",
    ):
        DenseRetriever._build_vector_filter(
            filters,
        )


# ============================================================
# VectorPipeline delegation
# ============================================================


def test_retrieve_calls_vector_pipeline_search():
    pipeline = make_pipeline(
        results=[
            make_vector_result(),
        ]
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request(
        top_k=5,
    )

    query_vector = [
        0.1,
        0.2,
        0.3,
    ]

    retriever.retrieve(
        request,
        query_vector,
    )

    pipeline.search.assert_called_once_with(
        collection_name="supportmind_documents",
        query_vector=query_vector,
        top_k=5,
        filters=None,
        provider_name=None,
    )


def test_retrieve_passes_provider_to_pipeline():
    pipeline = make_pipeline(
        results=[
            make_vector_result(),
        ]
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request(
        provider="qdrant",
    )

    retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    pipeline.search.assert_called_once_with(
        collection_name="supportmind_documents",
        query_vector=[0.1, 0.2],
        top_k=5,
        filters=None,
        provider_name="qdrant",
    )


def test_retrieve_passes_vector_filter_to_pipeline():
    pipeline = make_pipeline(
        results=[
            make_vector_result(),
        ]
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    filters = {
        "conditions": [
            {
                "field": "metadata.source",
                "operator": "eq",
                "value": "pdf",
            }
        ]
    }

    request = make_request(
        filters=filters,
    )

    retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    call = pipeline.search.call_args

    assert call.kwargs["filters"] is not None

    assert isinstance(
        call.kwargs["filters"],
        VectorFilter,
    )


# ============================================================
# Empty retrieval
# ============================================================


def test_retrieve_returns_empty_when_pipeline_returns_empty():
    pipeline = make_pipeline(
        results=[],
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    result = retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    assert result == []


# ============================================================
# Result conversion
# ============================================================


def test_convert_result_returns_retrieval_result():
    vector_result = make_vector_result(
        score=0.87,
    )

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert isinstance(
        result,
        RetrievalResult,
    )


def test_convert_result_preserves_identity_fields():
    vector_result = make_vector_result(
        result_id="result-42",
        node_id="node-42",
        document_id="doc-42",
    )

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert result.id == "result-42"
    assert result.node_id == "node-42"
    assert result.document_id == "doc-42"


def test_convert_result_preserves_text():
    vector_result = make_vector_result(
        text="Refund requests require verification.",
    )

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert (
        result.text
        == "Refund requests require verification."
    )


def test_convert_result_preserves_metadata():
    metadata = {
        "source": "pdf",
        "page": 12,
        "category": "billing",
    }

    vector_result = make_vector_result(
        metadata=metadata,
    )

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert result.metadata == metadata


def test_convert_result_preserves_score():
    vector_result = make_vector_result(
        score=0.9345,
    )

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert result.score == pytest.approx(0.9345)


def test_convert_result_assigns_dense_source():
    vector_result = make_vector_result()

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert result.source == "dense"


def test_convert_result_adds_dense_retrieval_score():
    vector_result = make_vector_result(
        score=0.82,
    )

    result = DenseRetriever._convert_result(
        result=vector_result,
        rank=1,
    )

    assert (
        result.retrieval_scores["dense_score"]
        == pytest.approx(0.82)
    )


# ============================================================
# Ranking
# ============================================================


def test_retrieve_assigns_deterministic_ranks():
    results = [
        make_vector_result(
            result_id="result-1",
            score=0.91,
        ),
        make_vector_result(
            result_id="result-2",
            score=0.85,
        ),
        make_vector_result(
            result_id="result-3",
            score=0.72,
        ),
    ]

    pipeline = make_pipeline(
        results=results,
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    retrieved = retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    assert [item.rank for item in retrieved] == [
        1,
        2,
        3,
    ]


def test_retrieve_preserves_pipeline_result_order():
    results = [
        make_vector_result(
            result_id="result-a",
            score=0.60,
        ),
        make_vector_result(
            result_id="result-b",
            score=0.90,
        ),
    ]

    pipeline = make_pipeline(
        results=results,
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    retrieved = retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    assert [
        item.id
        for item in retrieved
    ] == [
        "result-a",
        "result-b",
    ]


# ============================================================
# Top-k
# ============================================================


def test_retrieve_passes_top_k_to_pipeline():
    pipeline = make_pipeline(
        results=[
            make_vector_result(),
        ]
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request(
        top_k=10,
    )

    retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    assert (
        pipeline.search.call_args.kwargs["top_k"]
        == 10
    )


# ============================================================
# Immutability
# ============================================================


def test_retrieve_does_not_mutate_original_vector_results():
    original = make_vector_result(
        result_id="result-1",
        score=0.88,
        metadata={
            "source": "pdf",
            "page": 5,
        },
    )

    original_snapshot = {
        "id": original.id,
        "node_id": original.node_id,
        "document_id": original.document_id,
        "text": original.text,
        "score": original.score,
        "metadata": dict(original.metadata),
    }

    pipeline = make_pipeline(
        results=[original],
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    assert original.id == original_snapshot["id"]
    assert original.node_id == original_snapshot["node_id"]
    assert (
        original.document_id
        == original_snapshot["document_id"]
    )
    assert original.text == original_snapshot["text"]
    assert original.score == original_snapshot["score"]
    assert (
        original.metadata
        == original_snapshot["metadata"]
    )


# ============================================================
# Multiple results
# ============================================================


def test_retrieve_converts_all_pipeline_results():
    results = [
        make_vector_result(
            result_id="result-1",
            score=0.95,
        ),
        make_vector_result(
            result_id="result-2",
            score=0.88,
        ),
        make_vector_result(
            result_id="result-3",
            score=0.76,
        ),
    ]

    pipeline = make_pipeline(
        results=results,
    )

    retriever = DenseRetriever(
        pipeline=pipeline,
    )

    request = make_request()

    retrieved = retriever.retrieve(
        request,
        [0.1, 0.2],
    )

    assert len(retrieved) == 3

    assert all(
        isinstance(
            result,
            RetrievalResult,
        )
        for result in retrieved
    )

    assert [
        result.source
        for result in retrieved
    ] == [
        "dense",
        "dense",
        "dense",
    ]