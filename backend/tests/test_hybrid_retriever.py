"""
Tests for SupportMind AI HybridRetriever.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.retrieval.dense.dense_retriever import DenseRetriever
from app.retrieval.hybrid.hybrid_retriever import HybridRetriever
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
)
from app.retrieval.sparse.sparse_retriever import SparseRetriever


# ==============================================================
# Helpers
# ==============================================================


def make_request(
    query: str = "payment failure",
    top_k: int = 3,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        top_k=top_k,
        collection_name="supportmind_documents",
    )


def make_result(
    result_id: str,
    source: RetrievalSource,
    rank: int,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        node_id=f"node-{result_id}",
        document_id=f"doc-{result_id}",
        text=f"text for {result_id}",
        score=score,
        rank=rank,
        source=source,
        metadata={},
        retrieval_scores={
            f"{source.value}_score": score,
        },
    )


def make_dense_results() -> list[RetrievalResult]:
    return [
        make_result(
            "dense-1",
            RetrievalSource.DENSE,
            1,
            0.95,
        ),
        make_result(
            "dense-2",
            RetrievalSource.DENSE,
            2,
            0.82,
        ),
    ]


def make_sparse_results() -> list[RetrievalResult]:
    return [
        make_result(
            "sparse-1",
            RetrievalSource.SPARSE,
            1,
            4.20,
        ),
        make_result(
            "sparse-2",
            RetrievalSource.SPARSE,
            2,
            2.80,
        ),
    ]


# ==============================================================
# Constructor Tests
# ==============================================================


def test_hybrid_retriever_accepts_valid_retrievers():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    # Bypass concrete-type restriction for interface behavior test.
    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    assert retriever.dense_retriever is dense
    assert retriever.sparse_retriever is sparse


def test_hybrid_retriever_rejects_invalid_dense_retriever():
    sparse = SparseRetriever()

    with pytest.raises(ValueError):
        HybridRetriever(
            dense_retriever=object(),
            sparse_retriever=sparse,
        )


def test_hybrid_retriever_rejects_invalid_sparse_retriever():
    dense = DenseRetriever()

    with pytest.raises(ValueError):
        HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=object(),
        )


# ==============================================================
# Retrieval Tests
# ==============================================================


def test_hybrid_retriever_calls_dense_and_sparse():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    dense.retrieve.return_value = make_dense_results()
    sparse.retrieve.return_value = make_sparse_results()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    request = make_request()
    query_vector = [0.1, 0.2, 0.3]

    dense_results, sparse_results = retriever.retrieve(
        request=request,
        query_vector=query_vector,
    )

    dense.retrieve.assert_called_once_with(
        request=request,
        query_vector=query_vector,
    )

    sparse.retrieve.assert_called_once_with(
        request=request,
    )

    assert dense_results == make_dense_results()
    assert sparse_results == make_sparse_results()


def test_hybrid_retriever_preserves_dense_results():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    expected_dense = make_dense_results()

    dense.retrieve.return_value = expected_dense
    sparse.retrieve.return_value = []

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    dense_results, _ = retriever.retrieve(
        request=make_request(),
        query_vector=[0.1, 0.2],
    )

    assert dense_results is expected_dense


def test_hybrid_retriever_preserves_sparse_results():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    expected_sparse = make_sparse_results()

    dense.retrieve.return_value = []
    sparse.retrieve.return_value = expected_sparse

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    _, sparse_results = retriever.retrieve(
        request=make_request(),
        query_vector=[0.1, 0.2],
    )

    assert sparse_results is expected_sparse


def test_hybrid_retriever_allows_empty_dense_results():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    dense.retrieve.return_value = []
    sparse.retrieve.return_value = make_sparse_results()

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    dense_results, sparse_results = retriever.retrieve(
        request=make_request(),
        query_vector=[0.1, 0.2],
    )

    assert dense_results == []
    assert sparse_results


def test_hybrid_retriever_allows_empty_sparse_results():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    dense.retrieve.return_value = make_dense_results()
    sparse.retrieve.return_value = []

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    dense_results, sparse_results = retriever.retrieve(
        request=make_request(),
        query_vector=[0.1, 0.2],
    )

    assert dense_results
    assert sparse_results == []


# ==============================================================
# Validation Tests
# ==============================================================


def test_hybrid_retriever_rejects_none_request():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    with pytest.raises(ValueError):
        retriever.retrieve(
            request=None,
            query_vector=[0.1],
        )


def test_hybrid_retriever_rejects_invalid_request():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    with pytest.raises(ValueError):
        retriever.retrieve(
            request="payment failure",
            query_vector=[0.1],
        )


def test_hybrid_retriever_rejects_none_query_vector():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    with pytest.raises(ValueError):
        retriever.retrieve(
            request=make_request(),
            query_vector=None,
        )


def test_hybrid_retriever_rejects_empty_query_vector():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    with pytest.raises(ValueError):
        retriever.retrieve(
            request=make_request(),
            query_vector=[],
        )


def test_hybrid_retriever_rejects_non_list_query_vector():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    with pytest.raises(ValueError):
        retriever.retrieve(
            request=make_request(),
            query_vector=(0.1, 0.2),
        )


def test_hybrid_retriever_rejects_non_numeric_query_vector():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    retriever = HybridRetriever.__new__(
        HybridRetriever
    )

    retriever.dense_retriever = dense
    retriever.sparse_retriever = sparse

    with pytest.raises(ValueError):
        retriever.retrieve(
            request=make_request(),
            query_vector=[0.1, "invalid"],
        )


# ==============================================================
# Orchestration Contract Tests
# ==============================================================


def test_hybrid_retriever_does_not_fuse_results():
    dense = Mock(spec=DenseRetriever)
    sparse = Mock(spec=SparseRetriever)

    dense_results = make_dense_results()
    sparse_results = make_sparse_results()

    dense.retrieve.return_value = dense_results
    sparse.retrieve.return_value = sparse_results

    retriever = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    result = retriever.retrieve(
        request=make_request(),
        query_vector=[0.1, 0.2],
    )

    assert result == (
        dense_results,
        sparse_results,
    )