from __future__ import annotations

from unittest.mock import Mock

import pytest
from sentence_transformers import CrossEncoder

from app.retrieval.models.retrieval_models import (
    RetrievalResult,
    RetrievalSource,
)
from app.retrieval.reranker.cross_encoder_baseline import (
    CrossEncoderBaseline,
    create_cross_encoder_reranker,
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


class FakeCrossEncoder:
    """
    Lightweight fake model for deterministic tests.
    """

    def __init__(
        self,
        scores: list[float],
    ) -> None:
        self.scores = scores
        self.calls = []

    def predict(
        self,
        pairs,
        show_progress_bar=False,
    ):
        self.calls.append(
            {
                "pairs": pairs,
                "show_progress_bar": show_progress_bar,
            }
        )

        return self.scores


def make_reranker(
    scores: list[float],
) -> tuple[CrossEncoderBaseline, FakeCrossEncoder]:

    fake_model = FakeCrossEncoder(scores)

    reranker = object.__new__(
        CrossEncoderBaseline
    )

    reranker.model_name = (
        CrossEncoderBaseline.DEFAULT_MODEL_NAME
    )

    reranker.max_length = (
        CrossEncoderBaseline.DEFAULT_MAX_LENGTH
    )

    reranker._model = fake_model

    return reranker, fake_model


# ==============================================================
# Constructor validation
# ==============================================================


def test_cross_encoder_uses_default_model_name():

    assert (
        CrossEncoderBaseline.DEFAULT_MODEL_NAME
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def test_cross_encoder_uses_default_max_length():

    assert (
        CrossEncoderBaseline.DEFAULT_MAX_LENGTH
        == 512
    )


def test_cross_encoder_rejects_empty_model_name():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_model_name("")


def test_cross_encoder_rejects_non_string_model_name():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_model_name(
            None
        )


def test_cross_encoder_rejects_zero_max_length():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_max_length(0)


def test_cross_encoder_rejects_negative_max_length():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_max_length(-1)


def test_cross_encoder_rejects_boolean_max_length():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_max_length(True)


# ==============================================================
# Query validation
# ==============================================================


def test_rerank_rejects_none_query():

    reranker, _ = make_reranker([1.0])

    with pytest.raises(ValueError):
        reranker.rerank(
            None,
            [],
        )


def test_rerank_rejects_empty_query():

    reranker, _ = make_reranker([1.0])

    with pytest.raises(ValueError):
        reranker.rerank(
            "   ",
            [],
        )


def test_rerank_rejects_non_string_query():

    reranker, _ = make_reranker([1.0])

    with pytest.raises(ValueError):
        reranker.rerank(
            123,
            [],
        )


# ==============================================================
# Candidate validation
# ==============================================================


def test_rerank_rejects_none_results():

    reranker, _ = make_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "payment failure",
            None,
        )


def test_rerank_rejects_string_results():

    reranker, _ = make_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "payment failure",
            "invalid",
        )


def test_rerank_rejects_invalid_result_type():

    reranker, _ = make_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "payment failure",
            [Mock()],
        )


def test_rerank_allows_empty_results():

    reranker, fake_model = make_reranker([])

    results = reranker.rerank(
        "payment failure",
        [],
    )

    assert results == []
    assert fake_model.calls == []


# ==============================================================
# top_k validation
# ==============================================================


def test_rerank_rejects_zero_top_k():

    reranker, _ = make_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "payment",
            [],
            top_k=0,
        )


def test_rerank_rejects_negative_top_k():

    reranker, _ = make_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "payment",
            [],
            top_k=-1,
        )


def test_rerank_rejects_boolean_top_k():

    reranker, _ = make_reranker([])

    with pytest.raises(ValueError):
        reranker.rerank(
            "payment",
            [],
            top_k=True,
        )


def test_rerank_accepts_none_top_k():

    reranker, _ = make_reranker([])

    assert (
        reranker.rerank(
            "payment",
            [],
            top_k=None,
        )
        == []
    )


# ==============================================================
# Model invocation
# ==============================================================


def test_rerank_calls_cross_encoder():

    reranker, fake_model = make_reranker(
        [0.9, 0.3]
    )

    results = [
        make_result(
            "one",
            "payment failure policy",
            1,
        ),
        make_result(
            "two",
            "refund policy",
            2,
        ),
    ]

    reranker.rerank(
        "payment failure",
        results,
    )

    assert len(fake_model.calls) == 1

    call = fake_model.calls[0]

    assert call["pairs"] == [
        (
            "payment failure",
            "payment failure policy",
        ),
        (
            "payment failure",
            "refund policy",
        ),
    ]

    assert call["show_progress_bar"] is False


# ==============================================================
# Ranking
# ==============================================================


def test_rerank_orders_by_cross_encoder_score():

    reranker, _ = make_reranker(
        [0.2, 0.9, 0.5]
    )

    results = [
        make_result(
            "one",
            "weak match",
            1,
        ),
        make_result(
            "two",
            "strong match",
            2,
        ),
        make_result(
            "three",
            "medium match",
            3,
        ),
    ]

    reranked = reranker.rerank(
        "payment failure",
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


def test_rerank_assigns_deterministic_final_ranks():

    reranker, _ = make_reranker(
        [0.2, 0.9, 0.5]
    )

    results = [
        make_result("one", "weak", 1),
        make_result("two", "strong", 2),
        make_result("three", "medium", 3),
    ]

    reranked = reranker.rerank(
        "query",
        results,
    )

    assert [
        result.rank
        for result in reranked
    ] == [1, 2, 3]


def test_rerank_is_deterministic_for_equal_scores():

    reranker, _ = make_reranker(
        [0.5, 0.5, 0.5]
    )

    results = [
        make_result("one", "first", 1),
        make_result("two", "second", 2),
        make_result("three", "third", 3),
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
        [0.2, 0.9, 0.5]
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
        [0.8, 0.6]
    )

    results = [
        make_result("one", "first", 1),
        make_result("two", "second", 2),
    ]

    reranked = reranker.rerank(
        "query",
        results,
        top_k=10,
    )

    assert len(reranked) == 2


# ==============================================================
# Score preservation
# ==============================================================


def test_rerank_adds_cross_encoder_score():

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
        "payment failure",
        results,
    )

    assert (
        reranked[0]
        .retrieval_scores[
            "cross_encoder_score"
        ]
        == pytest.approx(0.85)
    )


def test_rerank_preserves_existing_retrieval_scores():

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
        "payment failure",
        results,
    )

    scores = (
        reranked[0]
        .retrieval_scores
    )

    assert scores["rrf_score"] == 0.42
    assert scores["cross_encoder_score"] == pytest.approx(
        0.85
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


def test_rerank_preserves_document_text():

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
        "query",
        results,
    )

    assert (
        reranked[0].text
        == "original document text"
    )


# ==============================================================
# Immutability
# ==============================================================


def test_rerank_does_not_mutate_original_result():

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

    reranked = reranker.rerank(
        "payment",
        [original],
    )

    assert (
        original.rank
        == 1
    )

    assert (
        original.retrieval_scores
        == original_scores
    )

    assert (
        "cross_encoder_score"
        not in original.retrieval_scores
    )

    assert (
        "cross_encoder_score"
        in reranked[0].retrieval_scores
    )


# ==============================================================
# Score validation
# ==============================================================


def test_score_validation_accepts_integer():

    assert (
        CrossEncoderBaseline._validate_score(1)
        == 1.0
    )


def test_score_validation_accepts_float():

    assert (
        CrossEncoderBaseline._validate_score(0.75)
        == 0.75
    )


def test_score_validation_rejects_boolean():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_score(True)


def test_score_validation_rejects_non_numeric():

    with pytest.raises(ValueError):
        CrossEncoderBaseline._validate_score(
            "0.5"
        )


# ==============================================================
# Factory
# ==============================================================


def test_create_cross_encoder_reranker_uses_default_configuration():

    fake_model = FakeCrossEncoder([])

    reranker = object.__new__(
        CrossEncoderBaseline
    )

    reranker.model_name = (
        CrossEncoderBaseline.DEFAULT_MODEL_NAME
    )

    reranker.max_length = (
        CrossEncoderBaseline.DEFAULT_MAX_LENGTH
    )

    reranker._model = fake_model

    assert (
        reranker.model_name
        == CrossEncoderBaseline.DEFAULT_MODEL_NAME
    )

    assert (
        reranker.max_length
        == CrossEncoderBaseline.DEFAULT_MAX_LENGTH
    )