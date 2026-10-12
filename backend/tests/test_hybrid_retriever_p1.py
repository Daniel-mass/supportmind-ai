"""
P1 tests for SupportMind AI hybrid retrieval and RRF fusion.

Coverage
--------
HybridRetriever
    - dependency validation
    - request validation
    - query-vector validation
    - dense retrieval delegation
    - sparse retrieval delegation
    - independent result preservation
    - deterministic behavior
    - no mutation of retrieval results

RRFFusion
    - configuration validation
    - input validation
    - RRF calculation
    - dense-only fusion
    - sparse-only fusion
    - dense + sparse fusion
    - document-level deduplication
    - metadata preservation
    - original score preservation
    - RRF score storage
    - deterministic tie-breaking
    - top-k enforcement
    - no mutation of original results
    - factory behavior

These tests intentionally use mocked retrievers/results.
They do NOT require Qdrant, Neo4j, BM25, embeddings, or an LLM.
"""

from __future__ import annotations

import pytest

from app.retrieval.dense.dense_retriever import DenseRetriever
from app.retrieval.hybrid.hybrid_retriever import HybridRetriever
from app.retrieval.hybrid.rrf_fusion import (
    RRFFusion,
    create_rrf_fusion,
)
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
    query: str = "refund policy",
    top_k: int = 5,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        top_k=top_k,
        collection_name="supportmind_documents",
    )


def make_result(
    *,
    result_id: str,
    node_id: str,
    document_id: str,
    text: str,
    score: float,
    rank: int,
    source: RetrievalSource,
    metadata: dict | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        id=result_id,
        node_id=node_id,
        document_id=document_id,
        text=text,
        score=score,
        rank=rank,
        source=source,
        metadata=metadata or {},
        retrieval_scores={
            "original_score": score,
        },
    )


class FakeDenseRetriever:
    """
    Test double deliberately implementing retrieve()
    without requiring a real DenseRetriever dependency.
    """

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def retrieve(self, request, query_vector):
        self.calls.append(
            {
                "request": request,
                "query_vector": query_vector,
            }
        )
        return list(self.results)


class FakeSparseRetriever:
    """
    Test double deliberately implementing retrieve()
    without requiring a real BM25 dependency.
    """

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def retrieve(self, request):
        self.calls.append(
            {
                "request": request,
            }
        )
        return list(self.results)


def build_real_dense_stub() -> DenseRetriever:
    """
    Create a DenseRetriever instance without requiring
    a live VectorPipeline.
    """

    class DummyPipeline:
        def search(self, **kwargs):
            return []

    return DenseRetriever(
        pipeline=DummyPipeline()
    )


def build_real_sparse_stub() -> SparseRetriever:
    """
    Create a SparseRetriever instance without requiring
    an application BM25 index.

    The HybridRetriever only validates the dependency type
    during construction. Retrieval itself is not executed
    in these construction tests.
    """

    return SparseRetriever()


# ==============================================================
# HybridRetriever
# ==============================================================


def test_hybrid_retriever_accepts_valid_dependencies():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    assert hybrid.dense_retriever is dense
    assert hybrid.sparse_retriever is sparse


def test_hybrid_retriever_rejects_invalid_dense_retriever():
    sparse = build_real_sparse_stub()

    with pytest.raises(
        ValueError,
        match="dense_retriever must be a DenseRetriever instance",
    ):
        HybridRetriever(
            dense_retriever=object(),
            sparse_retriever=sparse,
        )


def test_hybrid_retriever_rejects_invalid_sparse_retriever():
    dense = build_real_dense_stub()

    with pytest.raises(
        ValueError,
        match="sparse_retriever must be a SparseRetriever instance",
    ):
        HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=object(),
        )


def test_hybrid_retriever_rejects_none_request():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    with pytest.raises(
        ValueError,
        match="Retrieval request cannot be None",
    ):
        hybrid.retrieve(
            request=None,
            query_vector=[0.1, 0.2],
        )


def test_hybrid_retriever_rejects_invalid_request_type():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    with pytest.raises(
        ValueError,
        match="request must be a RetrievalRequest instance",
    ):
        hybrid.retrieve(
            request="invalid",
            query_vector=[0.1, 0.2],
        )


def test_hybrid_retriever_rejects_none_query_vector():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector cannot be None",
    ):
        hybrid.retrieve(
            request=request,
            query_vector=None,
        )


def test_hybrid_retriever_rejects_non_list_query_vector():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector must be provided as a list",
    ):
        hybrid.retrieve(
            request=request,
            query_vector=(0.1, 0.2),
        )


def test_hybrid_retriever_rejects_empty_query_vector():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector cannot be empty",
    ):
        hybrid.retrieve(
            request=request,
            query_vector=[],
        )


def test_hybrid_retriever_rejects_non_numeric_query_vector():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector must contain only numeric values",
    ):
        hybrid.retrieve(
            request=request,
            query_vector=[0.1, "invalid"],
        )


def test_hybrid_retriever_rejects_boolean_query_vector_value():
    dense = build_real_dense_stub()
    sparse = build_real_sparse_stub()

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
    )

    request = make_request()

    with pytest.raises(
        ValueError,
        match="Query vector must contain only numeric values",
    ):
        hybrid.retrieve(
            request=request,
            query_vector=[0.1, True],
        )


def test_hybrid_retriever_calls_dense_retriever():
    dense_result = make_result(
        result_id="d1",
        node_id="n1",
        document_id="doc1",
        text="refund policy",
        score=0.95,
        rank=1,
        source=RetrievalSource.DENSE,
    )

    sparse_result = make_result(
        result_id="s1",
        node_id="n2",
        document_id="doc2",
        text="payment policy",
        score=2.5,
        rank=1,
        source=RetrievalSource.SPARSE,
    )

    dense = FakeDenseRetriever(
        results=[dense_result]
    )
    sparse = FakeSparseRetriever(
        results=[sparse_result]
    )

    # HybridRetriever currently enforces concrete types.
    # Therefore use real wrappers and monkeypatch retrieve().
    real_dense = build_real_dense_stub()
    real_sparse = build_real_sparse_stub()

    dense_calls = []
    sparse_calls = []

    def dense_retrieve(request, query_vector):
        dense_calls.append(
            (request, query_vector)
        )
        return [dense_result]

    def sparse_retrieve(request):
        sparse_calls.append(request)
        return [sparse_result]

    real_dense.retrieve = dense_retrieve
    real_sparse.retrieve = sparse_retrieve

    hybrid = HybridRetriever(
        dense_retriever=real_dense,
        sparse_retriever=real_sparse,
    )

    request = make_request()
    vector = [0.1, 0.2, 0.3]

    dense_results, sparse_results = hybrid.retrieve(
        request=request,
        query_vector=vector,
    )

    assert len(dense_calls) == 1
    assert dense_calls[0][0] is request
    assert dense_calls[0][1] == vector

    assert dense_results == [dense_result]
    assert sparse_results == [sparse_result]


def test_hybrid_retriever_calls_sparse_retriever():
    dense_result = make_result(
        result_id="d1",
        node_id="n1",
        document_id="doc1",
        text="refund policy",
        score=0.95,
        rank=1,
        source=RetrievalSource.DENSE,
    )

    sparse_result = make_result(
        result_id="s1",
        node_id="n2",
        document_id="doc2",
        text="payment policy",
        score=2.5,
        rank=1,
        source=RetrievalSource.SPARSE,
    )

    real_dense = build_real_dense_stub()
    real_sparse = build_real_sparse_stub()

    sparse_calls = []

    real_dense.retrieve = (
        lambda request, query_vector: [dense_result]
    )

    def sparse_retrieve(request):
        sparse_calls.append(request)
        return [sparse_result]

    real_sparse.retrieve = sparse_retrieve

    hybrid = HybridRetriever(
        dense_retriever=real_dense,
        sparse_retriever=real_sparse,
    )

    request = make_request()

    _, sparse_results = hybrid.retrieve(
        request=request,
        query_vector=[0.1, 0.2],
    )

    assert len(sparse_calls) == 1
    assert sparse_calls[0] is request
    assert sparse_results == [sparse_result]


def test_hybrid_retriever_preserves_independent_result_lists():
    dense_results = [
        make_result(
            result_id="d1",
            node_id="dn1",
            document_id="doc1",
            text="refund",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            result_id="d2",
            node_id="dn2",
            document_id="doc2",
            text="payment",
            score=0.8,
            rank=2,
            source=RetrievalSource.DENSE,
        ),
    ]

    sparse_results = [
        make_result(
            result_id="s1",
            node_id="sn1",
            document_id="doc3",
            text="subscription",
            score=3.0,
            rank=1,
            source=RetrievalSource.SPARSE,
        ),
    ]

    real_dense = build_real_dense_stub()
    real_sparse = build_real_sparse_stub()

    real_dense.retrieve = (
        lambda request, query_vector: list(dense_results)
    )

    real_sparse.retrieve = (
        lambda request: list(sparse_results)
    )

    hybrid = HybridRetriever(
        dense_retriever=real_dense,
        sparse_retriever=real_sparse,
    )

    dense, sparse = hybrid.retrieve(
        request=make_request(),
        query_vector=[0.1, 0.2],
    )

    assert dense == dense_results
    assert sparse == sparse_results

    assert dense is not dense_results
    assert sparse is not sparse_results


def test_hybrid_retriever_supports_empty_dense_results():
    real_dense = build_real_dense_stub()
    real_sparse = build_real_sparse_stub()

    sparse_result = make_result(
        result_id="s1",
        node_id="sn1",
        document_id="doc1",
        text="refund",
        score=2.0,
        rank=1,
        source=RetrievalSource.SPARSE,
    )

    real_dense.retrieve = (
        lambda request, query_vector: []
    )

    real_sparse.retrieve = (
        lambda request: [sparse_result]
    )

    hybrid = HybridRetriever(
        dense_retriever=real_dense,
        sparse_retriever=real_sparse,
    )

    dense, sparse = hybrid.retrieve(
        request=make_request(),
        query_vector=[0.1],
    )

    assert dense == []
    assert sparse == [sparse_result]


def test_hybrid_retriever_supports_empty_sparse_results():
    real_dense = build_real_dense_stub()
    real_sparse = build_real_sparse_stub()

    dense_result = make_result(
        result_id="d1",
        node_id="dn1",
        document_id="doc1",
        text="refund",
        score=0.9,
        rank=1,
        source=RetrievalSource.DENSE,
    )

    real_dense.retrieve = (
        lambda request, query_vector: [dense_result]
    )

    real_sparse.retrieve = (
        lambda request: []
    )

    hybrid = HybridRetriever(
        dense_retriever=real_dense,
        sparse_retriever=real_sparse,
    )

    dense, sparse = hybrid.retrieve(
        request=make_request(),
        query_vector=[0.1],
    )

    assert dense == [dense_result]
    assert sparse == []


def test_hybrid_retriever_is_deterministic():
    dense_result = make_result(
        result_id="d1",
        node_id="dn1",
        document_id="doc1",
        text="refund",
        score=0.9,
        rank=1,
        source=RetrievalSource.DENSE,
    )

    sparse_result = make_result(
        result_id="s1",
        node_id="sn1",
        document_id="doc2",
        text="payment",
        score=2.0,
        rank=1,
        source=RetrievalSource.SPARSE,
    )

    real_dense = build_real_dense_stub()
    real_sparse = build_real_sparse_stub()

    real_dense.retrieve = (
        lambda request, query_vector: [dense_result]
    )

    real_sparse.retrieve = (
        lambda request: [sparse_result]
    )

    hybrid = HybridRetriever(
        dense_retriever=real_dense,
        sparse_retriever=real_sparse,
    )

    request = make_request()

    first = hybrid.retrieve(
        request=request,
        query_vector=[0.1, 0.2],
    )

    second = hybrid.retrieve(
        request=request,
        query_vector=[0.1, 0.2],
    )

    assert first == second


# ==============================================================
# RRF Configuration
# ==============================================================


def test_rrf_uses_default_k():
    fusion = RRFFusion()

    assert fusion.k == 60


def test_rrf_accepts_custom_k():
    fusion = RRFFusion(k=10)

    assert fusion.k == 10


def test_rrf_rejects_boolean_k():
    with pytest.raises(
        ValueError,
        match="RRF k must be a positive integer",
    ):
        RRFFusion(k=True)


def test_rrf_rejects_non_integer_k():
    with pytest.raises(
        ValueError,
        match="RRF k must be a positive integer",
    ):
        RRFFusion(k=1.5)


def test_rrf_rejects_zero_k():
    with pytest.raises(
        ValueError,
        match="RRF k must be greater than zero",
    ):
        RRFFusion(k=0)


def test_rrf_rejects_negative_k():
    with pytest.raises(
        ValueError,
        match="RRF k must be greater than zero",
    ):
        RRFFusion(k=-1)


# ==============================================================
# RRF Validation
# ==============================================================


def test_rrf_rejects_none_dense_results():
    fusion = RRFFusion()

    sparse = []

    with pytest.raises(
        ValueError,
        match="dense_results cannot be None",
    ):
        fusion.fuse(
            None,
            sparse,
            top_k=5,
        )


def test_rrf_rejects_none_sparse_results():
    fusion = RRFFusion()

    dense = []

    with pytest.raises(
        ValueError,
        match="sparse_results cannot be None",
    ):
        fusion.fuse(
            dense,
            None,
            top_k=5,
        )


def test_rrf_rejects_string_results():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="must be a sequence of RetrievalResult objects",
    ):
        fusion.fuse(
            "invalid",
            [],
            top_k=5,
        )


def test_rrf_rejects_invalid_result_objects():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="All items in dense_results must be RetrievalResult",
    ):
        fusion.fuse(
            [object()],
            [],
            top_k=5,
        )


def test_rrf_rejects_zero_top_k():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        fusion.fuse(
            [],
            [],
            top_k=0,
        )


def test_rrf_rejects_negative_top_k():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        fusion.fuse(
            [],
            [],
            top_k=-1,
        )


def test_rrf_rejects_boolean_top_k():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="top_k must be a positive integer",
    ):
        fusion.fuse(
            [],
            [],
            top_k=True,
        )


# ==============================================================
# RRF Calculation
# ==============================================================


def test_rrf_score_calculation():
    fusion = RRFFusion(k=60)

    assert fusion._rrf_score(1) == pytest.approx(
        1 / 61
    )

    assert fusion._rrf_score(2) == pytest.approx(
        1 / 62
    )


def test_rrf_score_rejects_zero_rank():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="RRF rank must be greater than zero",
    ):
        fusion._rrf_score(0)


def test_rrf_score_rejects_negative_rank():
    fusion = RRFFusion()

    with pytest.raises(
        ValueError,
        match="RRF rank must be greater than zero",
    ):
        fusion._rrf_score(-1)


# ==============================================================
# RRF Fusion Behavior
# ==============================================================


def test_rrf_returns_empty_for_empty_inputs():
    fusion = RRFFusion()

    assert fusion.fuse(
        [],
        [],
        top_k=5,
    ) == []


def test_rrf_supports_dense_only_results():
    fusion = RRFFusion(k=60)

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=0.95,
            rank=1,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            result_id="d2",
            node_id="n2",
            document_id="doc2",
            text="payment",
            score=0.90,
            rank=2,
            source=RetrievalSource.DENSE,
        ),
    ]

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert len(results) == 2
    assert results[0].document_id == "doc1"
    assert results[1].document_id == "doc2"


def test_rrf_supports_sparse_only_results():
    fusion = RRFFusion(k=60)

    sparse = [
        make_result(
            result_id="s1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=5.0,
            rank=1,
            source=RetrievalSource.SPARSE,
        ),
        make_result(
            result_id="s2",
            node_id="n2",
            document_id="doc2",
            text="payment",
            score=4.0,
            rank=2,
            source=RetrievalSource.SPARSE,
        ),
    ]

    results = fusion.fuse(
        [],
        sparse,
        top_k=5,
    )

    assert len(results) == 2
    assert results[0].document_id == "doc1"
    assert results[1].document_id == "doc2"


def test_rrf_fuses_dense_and_sparse_results():
    fusion = RRFFusion(k=60)

    dense = [
        make_result(
            result_id="d1",
            node_id="dn1",
            document_id="doc1",
            text="refund",
            score=0.95,
            rank=1,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            result_id="d2",
            node_id="dn2",
            document_id="doc2",
            text="payment",
            score=0.90,
            rank=2,
            source=RetrievalSource.DENSE,
        ),
    ]

    sparse = [
        make_result(
            result_id="s2",
            node_id="sn2",
            document_id="doc2",
            text="payment",
            score=5.0,
            rank=1,
            source=RetrievalSource.SPARSE,
        ),
        make_result(
            result_id="s3",
            node_id="sn3",
            document_id="doc3",
            text="subscription",
            score=4.0,
            rank=2,
            source=RetrievalSource.SPARSE,
        ),
    ]

    results = fusion.fuse(
        dense,
        sparse,
        top_k=5,
    )

    assert len(results) == 3

    # doc2 appears at rank 2 in dense and rank 1 in sparse.
    expected_doc2 = (
        1 / 62
        + 1 / 61
    )

    assert results[0].document_id == "doc2"
    assert results[0].retrieval_scores["rrf_score"] == pytest.approx(
        expected_doc2
    )


def test_rrf_deduplicates_same_document():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="dense-id",
            node_id="dense-node",
            document_id="doc1",
            text="dense version",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    sparse = [
        make_result(
            result_id="sparse-id",
            node_id="sparse-node",
            document_id="doc1",
            text="sparse version",
            score=3.0,
            rank=1,
            source=RetrievalSource.SPARSE,
        )
    ]

    results = fusion.fuse(
        dense,
        sparse,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].document_id == "doc1"


def test_rrf_uses_document_id_for_cross_retriever_deduplication():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="dense-id",
            node_id="dense-node",
            document_id="same-document",
            text="dense",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    sparse = [
        make_result(
            result_id="different-id",
            node_id="different-node",
            document_id="same-document",
            text="sparse",
            score=2.0,
            rank=1,
            source=RetrievalSource.SPARSE,
        )
    ]

    results = fusion.fuse(
        dense,
        sparse,
        top_k=5,
    )

    assert len(results) == 1


def test_rrf_preserves_first_seen_representative():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="dense-id",
            node_id="dense-node",
            document_id="doc1",
            text="dense representative",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
            metadata={"origin": "dense"},
        )
    ]

    sparse = [
        make_result(
            result_id="sparse-id",
            node_id="sparse-node",
            document_id="doc1",
            text="sparse representative",
            score=2.0,
            rank=1,
            source=RetrievalSource.SPARSE,
            metadata={"origin": "sparse"},
        )
    ]

    results = fusion.fuse(
        dense,
        sparse,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].source == RetrievalSource.DENSE
    assert results[0].metadata["origin"] == "dense"


def test_rrf_stores_rrf_score():
    fusion = RRFFusion(k=60)

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert "rrf_score" in results[0].retrieval_scores


def test_rrf_replaces_primary_score_with_rrf_score():
    fusion = RRFFusion(k=60)

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert results[0].score == pytest.approx(
        1 / 61
    )


def test_rrf_preserves_original_retrieval_scores():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    original_scores = dict(
        dense[0].retrieval_scores
    )

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert results[0].retrieval_scores[
        "original_score"
    ] == original_scores["original_score"]

    assert "rrf_score" in results[0].retrieval_scores


def test_rrf_assigns_deterministic_final_ranks():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="one",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            result_id="d2",
            node_id="n2",
            document_id="doc2",
            text="two",
            score=0.8,
            rank=2,
            source=RetrievalSource.DENSE,
        ),
    ]

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert [result.rank for result in results] == [1, 2]


def test_rrf_respects_top_k():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id=f"d{i}",
            node_id=f"n{i}",
            document_id=f"doc{i}",
            text=f"text {i}",
            score=1.0 - i * 0.01,
            rank=i,
            source=RetrievalSource.DENSE,
        )
        for i in range(1, 6)
    ]

    results = fusion.fuse(
        dense,
        [],
        top_k=2,
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_rrf_deterministic_tie_breaking_uses_first_seen_order():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="one",
            score=1.0,
            rank=1,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            result_id="d2",
            node_id="n2",
            document_id="doc2",
            text="two",
            score=1.0,
            rank=2,
            source=RetrievalSource.DENSE,
        ),
    ]

    sparse = [
        make_result(
            result_id="s1",
            node_id="sn1",
            document_id="doc3",
            text="three",
            score=1.0,
            rank=1,
            source=RetrievalSource.SPARSE,
        ),
    ]

    results = fusion.fuse(
        dense,
        sparse,
        top_k=5,
    )

    # doc1 and doc3 both receive rank-1 contributions.
    # doc1 was encountered first.
    assert results[0].document_id == "doc1"


def test_rrf_does_not_mutate_original_results():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    original_score = dense[0].score
    original_rank = dense[0].rank
    original_scores = dict(
        dense[0].retrieval_scores
    )

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert results[0] is not dense[0]

    assert dense[0].score == original_score
    assert dense[0].rank == original_rank
    assert dense[0].retrieval_scores == original_scores


def test_rrf_preserves_metadata():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="d1",
            node_id="n1",
            document_id="doc1",
            text="refund",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
            metadata={
                "page": 7,
                "category": "policy",
            },
        )
    ]

    results = fusion.fuse(
        dense,
        [],
        top_k=5,
    )

    assert results[0].metadata == {
        "page": 7,
        "category": "policy",
    }


def test_rrf_preserves_identity_fields():
    fusion = RRFFusion()

    dense = [
        make_result(
            result_id="d1",
            node_id="node1",
            document_id="doc1",
            text="refund policy",
            score=0.9,
            rank=1,
            source=RetrievalSource.DENSE,
        )
    ]

    result = fusion.fuse(
        dense,
        [],
        top_k=5,
    )[0]

    assert result.id == "d1"
    assert result.node_id == "node1"
    assert result.document_id == "doc1"
    assert result.text == "refund policy"


def test_rrf_factory_uses_default_k():
    fusion = create_rrf_fusion()

    assert isinstance(
        fusion,
        RRFFusion,
    )

    assert fusion.k == RRFFusion.DEFAULT_K


def test_rrf_factory_accepts_custom_k():
    fusion = create_rrf_fusion(k=20)

    assert isinstance(
        fusion,
        RRFFusion,
    )

    assert fusion.k == 20