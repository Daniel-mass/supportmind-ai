"""
Tests for SupportMind AI SparseRetriever.
"""

from __future__ import annotations

import pytest

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalSource,
)
from app.retrieval.sparse.sparse_retriever import (
    SparseDocument,
    SparseRetriever,
)


# ==============================================================
# Fixtures / Helpers
# ==============================================================


def make_document(
    document_id: str,
    text: str,
    *,
    node_id: str | None = None,
    metadata: dict | None = None,
) -> SparseDocument:
    """
    Create a test SparseDocument.
    """

    return SparseDocument(
        id=f"chunk-{document_id}",
        node_id=node_id or f"node-{document_id}",
        document_id=document_id,
        text=text,
        metadata=metadata or {},
    )


def make_request(
    query: str = "payment failure",
    top_k: int = 3,
) -> RetrievalRequest:
    """
    Create a standard retrieval request.
    """

    return RetrievalRequest(
        query=query,
        top_k=top_k,
        collection_name="supportmind_documents",
    )


def make_documents() -> list[SparseDocument]:
    """
    Standard BM25 test corpus.
    """

    return [
        make_document(
            "payment-policy",
            (
                "Payment failures can occur when a payment "
                "method is declined or unavailable."
            ),
            metadata={
                "category": "payment",
                "source": "policy",
            },
        ),
        make_document(
            "refund-policy",
            (
                "Customers can request a refund for eligible "
                "payments under the refund policy."
            ),
            metadata={
                "category": "refund",
                "source": "policy",
            },
        ),
        make_document(
            "subscription-guide",
            (
                "Subscription failures may prevent customers "
                "from accessing premium features."
            ),
            metadata={
                "category": "subscription",
                "source": "guide",
            },
        ),
    ]


# ==============================================================
# Document Tests
# ==============================================================


def test_sparse_document_requires_valid_id():
    with pytest.raises(ValueError):
        SparseDocument(
            id="",
            node_id="node-1",
            document_id="doc-1",
            text="payment failure",
        )


def test_sparse_document_requires_valid_node_id():
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="",
            document_id="doc-1",
            text="payment failure",
        )


def test_sparse_document_requires_valid_document_id():
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="",
            text="payment failure",
        )


def test_sparse_document_requires_text_string():
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="doc-1",
            text=None,
        )


# ==============================================================
# Index Tests
# ==============================================================


def test_sparse_retriever_builds_index():
    retriever = SparseRetriever()

    assert retriever.is_indexed is False
    assert retriever.document_count == 0

    retriever.build_index(
        make_documents()
    )

    assert retriever.is_indexed is True
    assert retriever.document_count == 3


def test_sparse_retriever_builds_index_from_constructor():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    assert retriever.is_indexed is True
    assert retriever.document_count == 3


def test_build_index_rejects_none_documents():
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index(None)


def test_build_index_rejects_empty_documents():
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index([])


def test_build_index_rejects_invalid_document_type():
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index(
            ["not-a-document"]
        )


def test_constructor_rejects_index_without_documents():
    with pytest.raises(ValueError):
        SparseRetriever(
            index=object()
        )


# ==============================================================
# Retrieval Tests
# ==============================================================


def test_retrieve_returns_matching_documents():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request(
            query="payment failure"
        )
    )

    assert results
    assert results[0].document_id == "payment-policy"


def test_retrieve_orders_by_bm25_score():
    retriever = SparseRetriever(
        documents=[
            make_document(
                "one",
                "payment payment payment failure",
            ),
            make_document(
                "two",
                "payment failure",
            ),
            make_document(
                "three",
                "refund request",
            ),
        ]
    )

    results = retriever.retrieve(
        make_request(
            query="payment failure",
            top_k=3,
        )
    )

    assert len(results) == 2
    assert (
        results[0].score
        >= results[1].score
    )


def test_retrieve_assigns_sparse_source():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results
    assert all(
        result.source == RetrievalSource.SPARSE
        for result in results
    )


def test_retrieve_assigns_deterministic_rank():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request()
    )

    assert [
        result.rank
        for result in results
    ] == list(
        range(
            1,
            len(results) + 1,
        )
    )


def test_retrieve_preserves_metadata():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request()
    )

    payment_result = next(
        result
        for result in results
        if result.document_id == "payment-policy"
    )

    assert payment_result.metadata == {
        "category": "payment",
        "source": "policy",
    }


def test_retrieve_preserves_bm25_score():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request()
    )

    assert results

    for result in results:
        assert (
            result.retrieval_scores["bm25_score"]
            == result.score
        )


def test_retrieve_respects_top_k():
    retriever = SparseRetriever(
        documents=[
            make_document(
                "one",
                "payment failure payment",
            ),
            make_document(
                "two",
                "payment failure",
            ),
            make_document(
                "three",
                "payment issue",
            ),
        ]
    )

    results = retriever.retrieve(
        make_request(
            query="payment",
            top_k=2,
        )
    )

    assert len(results) <= 2


def test_retrieve_returns_empty_for_no_matches():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request(
            query="quantum spaceship",
        )
    )

    assert results == []


def test_retrieve_returns_empty_for_tokenless_query():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    results = retriever.retrieve(
        make_request(
            query="!!!",
        )
    )

    assert results == []


# ==============================================================
# Validation Tests
# ==============================================================


def test_retrieve_rejects_none_request():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    with pytest.raises(ValueError):
        retriever.retrieve(None)


def test_retrieve_rejects_invalid_request_type():
    retriever = SparseRetriever(
        documents=make_documents()
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            "payment failure"
        )


def test_retrieve_rejects_unbuilt_index():
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.retrieve(
            make_request()
        )


# ==============================================================
# Tokenization Tests
# ==============================================================


def test_tokenizer_is_case_insensitive():
    tokens = SparseRetriever._tokenize(
        "Payment FAILURE"
    )

    assert tokens == [
        "payment",
        "failure",
    ]


def test_tokenizer_removes_punctuation():
    tokens = SparseRetriever._tokenize(
        "Payment, failure!"
    )

    assert tokens == [
        "payment",
        "failure",
    ]


def test_tokenizer_preserves_unicode_words():
    tokens = SparseRetriever._tokenize(
        "payment café"
    )

    assert tokens == [
        "payment",
        "café",
    ]