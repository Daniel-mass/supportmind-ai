from __future__ import annotations

from unittest.mock import Mock

import pytest

from sentence_transformers import MultiVectorEncoder

from app.retrieval.models.retrieval_models import (
    RetrievalResult,
    RetrievalSource,
)

from app.retrieval.reranker.colbert_reranker import (
    ColBERTReranker,
    create_colbert_reranker,
)


# ==============================================================
# Helpers
# ==============================================================


def make_result(
    result_id: str,
    text: str,
    rank: int,
    score: float = 0.5,
) -> RetrievalResult:

    return RetrievalResult(
        id=result_id,
        node_id=f"node-{result_id}",
        document_id=f"document-{result_id}",
        text=text,
        score=score,
        rank=rank,
        source=RetrievalSource.HYBRID,
        metadata={
            "category": "test",
        },
        retrieval_scores={
            "rrf_score": score,
        },
    )


class FakeColBERTModel:
    """
    Deterministic fake MultiVectorEncoder.

    The fake returns predefined similarity scores.
    """

    def __init__(
        self,
        scores: list[float],
    ) -> None:

        self.scores = scores

        self.query_calls = []
        self.document_calls = []
        self.similarity_calls = []

    def encode_query(
        self,
        queries,
    ):

        self.query_calls.append(
            queries
        )

        return [
            "fake-query-embedding"
        ]

    def encode_document(
        self,
        documents,
    ):

        self.document_calls.append(
            documents
        )

        return [
            f"fake-doc-{index}"
            for index in range(
                len(documents)
            )
        ]

    def similarity(
        self,
        query_embeddings,
        document_embeddings,
    ):

        self.similarity_calls.append(
            (
                query_embeddings,
                document_embeddings,
            )
        )

        return [
            self.scores
        ]


def make_reranker(
    scores: list[float],
) -> tuple[
    ColBERTReranker,
    FakeColBERTModel,
]:

    fake_model = FakeColBERTModel(
        scores
    )

    reranker = object.__new__(
        ColBERTReranker
    )

    reranker.model_name = (
        ColBERTReranker.DEFAULT_MODEL_NAME
    )

    reranker._model = fake_model

    return (
        reranker,
        fake_model,
    )


# ==============================================================
# Configuration
# ==============================================================


def test_colbert_uses_default_model_name():

    assert (
        ColBERTReranker.DEFAULT_MODEL_NAME
        == "colbert-ir/colbertv2.0"
    )


def test_colbert_rejects_empty_model_name():

    with pytest.raises(ValueError):

        ColBERTReranker._validate_model_name(
            ""
        )


def test_colbert_rejects_non_string_model_name():

    with pytest.raises(ValueError):

        ColBERTReranker._validate_model_name(
            None
        )


# ==============================================================
# Query validation
# ==============================================================


def test_rerank_rejects_none_query():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            None,
            [],
        )


def test_rerank_rejects_empty_query():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "   ",
            [],
        )


def test_rerank_rejects_non_string_query():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            123,
            [],
        )


# ==============================================================
# Candidate validation
# ==============================================================


def test_rerank_rejects_none_results():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "payment failure",
            None,
        )


def test_rerank_rejects_string_results():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "payment failure",
            "invalid",
        )


def test_rerank_rejects_invalid_result_type():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "payment failure",
            [Mock()],
        )


def test_rerank_allows_empty_results():

    reranker, fake_model = make_reranker(
        []
    )

    results = reranker.rerank(
        "payment failure",
        [],
    )

    assert results == []

    assert (
        fake_model.query_calls
        == []
    )

    assert (
        fake_model.document_calls
        == []
    )


# ==============================================================
# top_k validation
# ==============================================================


def test_rerank_rejects_zero_top_k():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "payment",
            [],
            top_k=0,
        )


def test_rerank_rejects_negative_top_k():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "payment",
            [],
            top_k=-1,
        )


def test_rerank_rejects_boolean_top_k():

    reranker, _ = make_reranker(
        []
    )

    with pytest.raises(ValueError):

        reranker.rerank(
            "payment",
            [],
            top_k=True,
        )


# ==============================================================
# Model interaction
# ==============================================================


def test_rerank_encodes_query_once():

    reranker, fake_model = make_reranker(
        [0.9]
    )

    results = [
        make_result(
            "one",
            "payment failure",
            1,
        )
    ]

    reranker.rerank(
        "payment failure",
        results,
    )

    assert (
        fake_model.query_calls
        == [
            ["payment failure"]
        ]
    )


def test_rerank_encodes_all_documents():

    reranker, fake_model = make_reranker(
        [0.9, 0.5]
    )

    results = [
        make_result(
            "one",
            "payment failure",
            1,
        ),
        make_result(
            "two",
            "refund policy",
            2,
        ),
    ]

    reranker.rerank(
        "payment",
        results,
    )

    assert (
        fake_model.document_calls
        == [
            [
                "payment failure",
                "refund policy",
            ]
        ]
    )


def test_rerank_uses_similarity():

    reranker, fake_model = make_reranker(
        [0.9, 0.4]
    )

    results = [
        make_result(
            "one",
            "payment failure",
            1,
        ),
        make_result(
            "two",
            "refund",
            2,
        ),
    ]

    reranker.rerank(
        "payment",
        results,
    )

    assert (
        len(
            fake_model.similarity_calls
        )
        == 1
    )


# ==============================================================
# Ranking
# ==============================================================


def test_rerank_orders_by_colbert_score():

    reranker, _ = make_reranker(
        [
            0.2,
            0.9,
            0.5,
        ]
    )

    results = [
        make_result(
            "one",
            "weak",
            1,
        ),
        make_result(
            "two",
            "strong",
            2,
        ),
        make_result(
            "three",
            "medium",
            3,
        ),
    ]

    reranked = reranker.rerank(
        "payment",
        results,
    )

    assert [
        result.id
        for result in reranked
    ] == [
        "two",
        "three",
        "one",
    ]


def test_rerank_assigns_deterministic_ranks():

    reranker, _ = make_reranker(
        [
            0.2,
            0.9,
            0.5,
        ]
    )

    results = [
        make_result("one", "weak", 1),
        make_result("two", "strong", 2),
        make_result("three", "medium", 3),
    ]

    reranked = reranker.rerank(
        "payment",
        results,
    )

    assert [
        result.rank
        for result in reranked
    ] == [
        1,
        2,
        3,
    ]


def test_rerank_is_deterministic_for_equal_scores():

    reranker, _ = make_reranker(
        [
            0.5,
            0.5,
            0.5,
        ]
    )

    results = [
        make_result(
            "one",
            "first",
            1,
        ),
        make_result(
            "two",
            "second",
            2,
        ),
        make_result(
            "three",
            "third",
            3,
        ),
    ]

    reranked = reranker.rerank(
        "query",
        results,
    )

    assert [
        result.id
        for result in reranked
    ] == [
        "one",
        "two",
        "three",
    ]


# ==============================================================
# top_k
# ==============================================================


def test_rerank_respects_top_k():

    reranker, _ = make_reranker(
        [
            0.2,
            0.9,
            0.5,
        ]
    )

    results = [
        make_result("one", "weak", 1),
        make_result("two", "strong", 2),
        make_result("three", "medium", 3),
    ]

    reranked = reranker.rerank(
        "query",
        results,
        top_k=2,
    )

    assert len(reranked) == 2

    assert [
        result.id
        for result in reranked
    ] == [
        "two",
        "three",
    ]


def test_rerank_top_k_larger_than_results():

    reranker, _ = make_reranker(
        [
            0.8,
            0.6,
        ]
    )

    results = [
        make_result(
            "one",
            "first",
            1,
        ),
        make_result(
            "two",
            "second",
            2,
        ),
    ]

    reranked = reranker.rerank(
        "query",
        results,
        top_k=10,
    )

    assert len(reranked) == 2


# ==============================================================
# Score handling
# ==============================================================


def test_rerank_adds_colbert_score():

    reranker, _ = make_reranker(
        [0.85]
    )

    results = [
        make_result(
            "one",
            "payment failure",
            1,
        )
    ]

    reranked = reranker.rerank(
        "payment",
        results,
    )

    assert (
        reranked[0]
        .retrieval_scores[
            "colbert_score"
        ]
        == pytest.approx(
            0.85
        )
    )


def test_rerank_preserves_existing_scores():

    reranker, _ = make_reranker(
        [0.85]
    )

    results = [
        make_result(
            "one",
            "payment failure",
            1,
            score=0.42,
        )
    ]

    reranked = reranker.rerank(
        "payment",
        results,
    )

    scores = (
        reranked[0]
        .retrieval_scores
    )

    assert (
        scores["rrf_score"]
        == 0.42
    )

    assert (
        scores["colbert_score"]
        == pytest.approx(
            0.85
        )
    )


# ==============================================================
# Metadata preservation
# ==============================================================


def test_rerank_preserves_metadata():

    reranker, _ = make_reranker(
        [0.9]
    )

    results = [
        make_result(
            "one",
            "payment failure",
            1,
        )
    ]

    reranked = reranker.rerank(
        "payment",
        results,
    )

    assert (
        reranked[0].metadata
        == results[0].metadata
    )


def test_rerank_preserves_text():

    reranker, _ = make_reranker(
        [0.9]
    )

    results = [
        make_result(
            "one",
            "original document text",
            1,
        )
    ]

    reranked = reranker.rerank(
        "payment",
        results,
    )

    assert (
        reranked[0].text
        == "original document text"
    )


# ==============================================================
# Immutability
# ==============================================================


def test_rerank_does_not_mutate_original():

    reranker, _ = make_reranker(
        [0.9]
    )

    original = make_result(
        "one",
        "payment failure",
        1,
    )

    original_scores = dict(
        original.retrieval_scores
    )

    original_rank = (
        original.rank
    )

    reranked = reranker.rerank(
        "payment",
        [original],
    )

    assert (
        original.rank
        == original_rank
    )

    assert (
        original.retrieval_scores
        == original_scores
    )

    assert (
        "colbert_score"
        not in original.retrieval_scores
    )

    assert (
        "colbert_score"
        in reranked[0].retrieval_scores
    )


# ==============================================================
# Score validation
# ==============================================================


def test_score_validation_accepts_integer():

    assert (
        ColBERTReranker._validate_score(
            1
        )
        == 1.0
    )


def test_score_validation_accepts_float():

    assert (
        ColBERTReranker._validate_score(
            0.75
        )
        == 0.75
    )


def test_score_validation_rejects_boolean():

    with pytest.raises(ValueError):

        ColBERTReranker._validate_score(
            True
        )


def test_score_validation_rejects_non_numeric():

    with pytest.raises(ValueError):

        ColBERTReranker._validate_score(
            "0.5"
        )


# ==============================================================
# Factory
# ==============================================================


def test_create_colbert_reranker_uses_default_model():

    fake_model = FakeColBERTModel(
        []
    )

    reranker = object.__new__(
        ColBERTReranker
    )

    reranker.model_name = (
        ColBERTReranker.DEFAULT_MODEL_NAME
    )

    reranker._model = fake_model

    assert (
        reranker.model_name
        == "colbert-ir/colbertv2.0"
    )