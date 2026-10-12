"""
P1 tests for SupportMind AI SparseRetriever.

Scope
-----
These tests verify the standalone BM25 sparse retrieval component.

They intentionally do NOT test:
- Qdrant
- Neo4j
- dense retrieval
- hybrid/RRF
- reranking
- context building
- retrieval service
- orchestrator

Component under test
--------------------
SparseRetriever
    |
    v
rank-bm25 / BM25Okapi
    |
    v
RetrievalResult
"""

from __future__ import annotations

import pytest

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
)
from app.retrieval.sparse.sparse_retriever import (
    SparseDocument,
    SparseRetriever,
)


# ==============================================================
# Fixtures
# ==============================================================


@pytest.fixture
def documents() -> list[SparseDocument]:
    """Small deterministic corpus for BM25 tests."""

    return [
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="doc-1",
            text="refund policy allows customers to request a refund",
            metadata={
                "source": "refund_policy.pdf",
                "page": 1,
                "category": "policy",
            },
        ),
        SparseDocument(
            id="chunk-2",
            node_id="node-2",
            document_id="doc-2",
            text="subscription cancellation policy explains cancellation rules",
            metadata={
                "source": "subscription_policy.pdf",
                "page": 2,
                "category": "policy",
            },
        ),
        SparseDocument(
            id="chunk-3",
            node_id="node-3",
            document_id="doc-3",
            text="payment failures can occur when a payment method is declined",
            metadata={
                "source": "payment_policy.pdf",
                "page": 3,
                "category": "payment",
            },
        ),
        SparseDocument(
            id="chunk-4",
            node_id="node-4",
            document_id="doc-4",
            text="customers can troubleshoot login and account access problems",
            metadata={
                "source": "troubleshooting_guide.pdf",
                "page": 4,
                "category": "troubleshooting",
            },
        ),
    ]


@pytest.fixture
def retriever(
    documents: list[SparseDocument],
) -> SparseRetriever:
    """Initialized sparse retriever."""

    return SparseRetriever(
        documents=documents,
    )


@pytest.fixture
def request_factory():
    """Create valid RetrievalRequest objects."""

    def _create(
        query: str,
        top_k: int = 5,
    ) -> RetrievalRequest:
        return RetrievalRequest(
            query=query,
            top_k=top_k,
            collection_name="supportmind_documents",
        )

    return _create


# ==============================================================
# SparseDocument Tests
# ==============================================================


def test_sparse_document_accepts_valid_data() -> None:
    document = SparseDocument(
        id="chunk-1",
        node_id="node-1",
        document_id="doc-1",
        text="refund policy",
        metadata={"page": 1},
    )

    assert document.id == "chunk-1"
    assert document.node_id == "node-1"
    assert document.document_id == "doc-1"
    assert document.text == "refund policy"
    assert document.metadata == {"page": 1}


def test_sparse_document_defaults_metadata_to_empty_dict() -> None:
    document = SparseDocument(
        id="chunk-1",
        node_id="node-1",
        document_id="doc-1",
        text="refund policy",
    )

    assert document.metadata == {}


def test_sparse_document_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        SparseDocument(
            id="",
            node_id="node-1",
            document_id="doc-1",
            text="refund policy",
        )


def test_sparse_document_rejects_whitespace_id() -> None:
    with pytest.raises(ValueError):
        SparseDocument(
            id="   ",
            node_id="node-1",
            document_id="doc-1",
            text="refund policy",
        )


def test_sparse_document_rejects_empty_node_id() -> None:
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="",
            document_id="doc-1",
            text="refund policy",
        )


def test_sparse_document_rejects_empty_document_id() -> None:
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="",
            text="refund policy",
        )


def test_sparse_document_rejects_non_string_text() -> None:
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="doc-1",
            text=123,  # type: ignore[arg-type]
        )


def test_sparse_document_rejects_non_dict_metadata() -> None:
    with pytest.raises(ValueError):
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="doc-1",
            text="refund policy",
            metadata=[],  # type: ignore[arg-type]
        )


# ==============================================================
# Retriever Initialization Tests
# ==============================================================


def test_retriever_can_be_created_without_dependencies() -> None:
    retriever = SparseRetriever()

    assert retriever.is_indexed is False
    assert retriever.document_count == 0


def test_retriever_builds_index_from_documents(
    documents: list[SparseDocument],
) -> None:
    retriever = SparseRetriever(
        documents=documents,
    )

    assert retriever.is_indexed is True
    assert retriever.document_count == len(documents)


def test_retriever_rejects_index_without_documents() -> None:
    fake_index = object()

    with pytest.raises(ValueError):
        SparseRetriever(
            index=fake_index,  # type: ignore[arg-type]
        )


def test_retriever_accepts_prebuilt_index_with_documents(
    documents: list[SparseDocument],
) -> None:
    first = SparseRetriever(
        documents=documents,
    )

    # Access the already-built BM25 index only to verify
    # the constructor's pre-built-index contract.
    index = first._index

    retriever = SparseRetriever(
        index=index,
        documents=documents,
    )

    assert retriever.is_indexed is True
    assert retriever.document_count == len(documents)


# ==============================================================
# Index Management Tests
# ==============================================================


def test_build_index_replaces_existing_index(
    retriever: SparseRetriever,
) -> None:
    replacement_documents = [
        SparseDocument(
            id="new-1",
            node_id="new-node-1",
            document_id="new-doc-1",
            text="incident escalation procedure",
        ),
        SparseDocument(
            id="new-2",
            node_id="new-node-2",
            document_id="new-doc-2",
            text="technical incident response",
        ),
    ]

    retriever.build_index(
        replacement_documents,
    )

    assert retriever.is_indexed is True
    assert retriever.document_count == 2


def test_build_index_rejects_none_documents() -> None:
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index(
            None,  # type: ignore[arg-type]
        )


def test_build_index_rejects_string_documents() -> None:
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index(
            "not documents",  # type: ignore[arg-type]
        )


def test_build_index_rejects_invalid_document_objects() -> None:
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index(
            [
                "invalid-document",  # type: ignore[list-item]
            ]
        )


def test_build_index_rejects_empty_document_collection() -> None:
    retriever = SparseRetriever()

    with pytest.raises(ValueError):
        retriever.build_index([])


def test_build_index_rejects_document_without_tokens() -> None:
    retriever = SparseRetriever()

    documents = [
        SparseDocument(
            id="chunk-1",
            node_id="node-1",
            document_id="doc-1",
            text="!!! ??? ...",
        )
    ]

    with pytest.raises(ValueError):
        retriever.build_index(documents)


# ==============================================================
# Request Validation Tests
# ==============================================================


def test_retrieve_rejects_none_request(
    retriever: SparseRetriever,
) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve(
            None,  # type: ignore[arg-type]
        )


def test_retrieve_rejects_invalid_request_type(
    retriever: SparseRetriever,
) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve(
            "invalid request",  # type: ignore[arg-type]
        )


def test_retrieve_rejects_query_before_unindexed_search() -> None:
    retriever = SparseRetriever()

    request = RetrievalRequest(
        query="refund",
        top_k=5,
        collection_name="supportmind_documents",
    )

    with pytest.raises(ValueError, match="index has not been built"):
        retriever.retrieve(request)


# ==============================================================
# Tokenization Tests
# ==============================================================


def test_tokenizer_is_case_insensitive() -> None:
    assert SparseRetriever._tokenize(
        "Refund POLICY"
    ) == [
        "refund",
        "policy",
    ]


def test_tokenizer_removes_punctuation() -> None:
    assert SparseRetriever._tokenize(
        "refund, policy! cancellation."
    ) == [
        "refund",
        "policy",
        "cancellation",
    ]


def test_tokenizer_preserves_unicode_words() -> None:
    tokens = SparseRetriever._tokenize(
        "payment café policy"
    )

    assert tokens == [
        "payment",
        "café",
        "policy",
    ]


def test_tokenizer_rejects_non_string_input() -> None:
    with pytest.raises(ValueError):
        SparseRetriever._tokenize(
            123  # type: ignore[arg-type]
        )


# ==============================================================
# Basic Retrieval Tests
# ==============================================================


def test_retrieve_returns_matching_documents(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund policy"
    )

    results = retriever.retrieve(request)

    assert results
    assert all(
        isinstance(result, RetrievalResult)
        for result in results
    )


def test_retrieve_returns_empty_for_no_lexical_match(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "quantum spaceship database"
    )

    results = retriever.retrieve(request)

    assert results == []


def test_retrieve_respects_top_k(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "policy",
        top_k=2,
    )

    results = retriever.retrieve(request)

    assert len(results) <= 2


def test_retrieve_returns_at_most_document_count(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "policy",
        top_k=100,
    )

    results = retriever.retrieve(request)

    assert len(results) <= retriever.document_count


# ==============================================================
# Result Contract Tests
# ==============================================================


def test_retrieve_returns_retrieval_results(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund"
    )

    results = retriever.retrieve(request)

    assert results

    for result in results:
        assert isinstance(
            result,
            RetrievalResult,
        )


def test_retrieve_assigns_sparse_source(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund"
    )

    results = retriever.retrieve(request)

    assert results
    assert all(
        result.source == RetrievalSource.SPARSE
        for result in results
    )


def test_retrieve_preserves_document_identity(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund"
    )

    results = retriever.retrieve(request)

    assert results

    result = results[0]

    assert result.id == "chunk-1"
    assert result.node_id == "node-1"
    assert result.document_id == "doc-1"


def test_retrieve_preserves_document_text(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund"
    )

    results = retriever.retrieve(request)

    assert results

    assert results[0].text == (
        "refund policy allows customers to request a refund"
    )


def test_retrieve_preserves_metadata(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund"
    )

    results = retriever.retrieve(request)

    assert results

    assert results[0].metadata == {
        "source": "refund_policy.pdf",
        "page": 1,
        "category": "policy",
    }


def test_retrieve_adds_bm25_score(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund"
    )

    results = retriever.retrieve(request)

    assert results

    for result in results:
        assert "bm25_score" in result.retrieval_scores
        assert (
            result.retrieval_scores["bm25_score"]
            == result.score
        )


def test_retrieve_scores_are_numeric(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund policy"
    )

    results = retriever.retrieve(request)

    assert results

    for result in results:
        assert isinstance(
            result.score,
            float,
        )


# ==============================================================
# Ranking Tests
# ==============================================================


def test_retrieve_assigns_deterministic_ranks(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "policy"
    )

    results = retriever.retrieve(request)

    assert results

    assert [
        result.rank
        for result in results
    ] == list(
        range(
            1,
            len(results) + 1,
        )
    )


def test_retrieve_ranks_follow_score_order(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "refund policy"
    )

    results = retriever.retrieve(request)

    assert results

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_retrieve_is_deterministic(
    retriever: SparseRetriever,
    request_factory,
) -> None:
    request = request_factory(
        "policy"
    )

    first = retriever.retrieve(request)
    second = retriever.retrieve(request)

    assert [
        result.id
        for result in first
    ] == [
        result.id
        for result in second
    ]

    assert [
        result.rank
        for result in first
    ] == [
        result.rank
        for result in second
    ]

    assert [
        result.score
        for result in first
    ] == [
        result.score
        for result in second
    ]


# ==============================================================
# Negative BM25 Score Behavior
# ==============================================================


def test_negative_bm25_scores_do_not_automatically_remove_matches(
    request_factory,
) -> None:
    """
    Regression test for the explicit sparse-retrieval design.

    A lexical match must survive even if rank-bm25 produces
    a negative score.
    """

    documents = [
        SparseDocument(
            id="doc-a",
            node_id="node-a",
            document_id="parent-a",
            text="common policy",
        ),
        SparseDocument(
            id="doc-b",
            node_id="node-b",
            document_id="parent-b",
            text="common policy",
        ),
    ]

    retriever = SparseRetriever(
        documents=documents,
    )

    request = request_factory(
        "common"
    )

    results = retriever.retrieve(request)

    assert len(results) == 2

    assert {
        result.id
        for result in results
    } == {
        "doc-a",
        "doc-b",
    }


# ==============================================================
# Tie-Breaking Tests
# ==============================================================


def test_equal_scores_use_original_document_order() -> None:
    documents = [
        SparseDocument(
            id="first",
            node_id="node-first",
            document_id="doc-first",
            text="shared",
        ),
        SparseDocument(
            id="second",
            node_id="node-second",
            document_id="doc-second",
            text="shared",
        ),
    ]

    retriever = SparseRetriever(
        documents=documents,
    )

    request = RetrievalRequest(
        query="shared",
        top_k=5,
        collection_name="supportmind_documents",
    )

    results = retriever.retrieve(request)

    assert [
        result.id
        for result in results
    ] == [
        "first",
        "second",
    ]


# ==============================================================
# Result Conversion Tests
# ==============================================================


def test_convert_result_returns_retrieval_result() -> None:
    document = SparseDocument(
        id="chunk-1",
        node_id="node-1",
        document_id="doc-1",
        text="refund policy",
        metadata={"page": 1},
    )

    result = SparseRetriever._convert_result(
        document=document,
        score=2.5,
        rank=1,
    )

    assert isinstance(
        result,
        RetrievalResult,
    )


def test_convert_result_preserves_score_and_rank() -> None:
    document = SparseDocument(
        id="chunk-1",
        node_id="node-1",
        document_id="doc-1",
        text="refund policy",
    )

    result = SparseRetriever._convert_result(
        document=document,
        score=-1.25,
        rank=3,
    )

    assert result.score == -1.25
    assert result.rank == 3


def test_convert_result_copies_metadata() -> None:
    metadata = {
        "page": 5,
        "category": "policy",
    }

    document = SparseDocument(
        id="chunk-1",
        node_id="node-1",
        document_id="doc-1",
        text="refund policy",
        metadata=metadata,
    )

    result = SparseRetriever._convert_result(
        document=document,
        score=1.0,
        rank=1,
    )

    assert result.metadata == metadata
    assert result.metadata is not metadata


# ==============================================================
# Mutation Safety Tests
# ==============================================================


def test_retrieve_does_not_mutate_original_documents(
    documents: list[SparseDocument],
    retriever: SparseRetriever,
    request_factory,
) -> None:
    original = [
        (
            document.id,
            document.node_id,
            document.document_id,
            document.text,
            dict(document.metadata),
        )
        for document in documents
    ]

    request = request_factory(
        "policy"
    )

    retriever.retrieve(request)

    after = [
        (
            document.id,
            document.node_id,
            document.document_id,
            document.text,
            dict(document.metadata),
        )
        for document in documents
    ]

    assert after == original


def test_build_index_does_not_mutate_input_collection(
    documents: list[SparseDocument],
) -> None:
    original = list(documents)

    retriever = SparseRetriever()

    retriever.build_index(documents)

    assert documents == original


# ==============================================================
# Factory Test
# ==============================================================


def test_create_sparse_retriever_factory(
    documents: list[SparseDocument],
) -> None:
    from app.retrieval.sparse.sparse_retriever import (
        create_sparse_retriever,
    )

    retriever = create_sparse_retriever(
        documents
    )

    assert isinstance(
        retriever,
        SparseRetriever,
    )

    assert retriever.is_indexed is True
    assert retriever.document_count == len(documents)