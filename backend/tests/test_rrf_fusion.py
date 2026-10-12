"""
Tests for SupportMind AI RRF Fusion.
"""

from __future__ import annotations

import pytest

from app.retrieval.hybrid.rrf_fusion import (
    RRFFusion,
    create_rrf_fusion,
)
from app.retrieval.models.retrieval_models import (
    RetrievalResult,
    RetrievalSource,
)


# ==============================================================
# Fixtures / Helpers
# ==============================================================


def make_result(
    document_id: str,
    *,
    rank: int,
    score: float = 0.5,
    source: RetrievalSource = RetrievalSource.DENSE,
    node_id: str | None = None,
    metadata: dict | None = None,
) -> RetrievalResult:
    """
    Create a RetrievalResult for testing.
    """

    return RetrievalResult(
        id=f"chunk-{document_id}",
        node_id=node_id or f"node-{document_id}",
        document_id=document_id,
        text=f"Text for {document_id}",
        score=score,
        rank=rank,
        source=source,
        metadata=metadata or {},
        retrieval_scores={
            "original_score": score,
        },
    )


def make_dense_results() -> list[RetrievalResult]:
    """
    Standard dense result list.
    """

    return [
        make_result(
            "doc-a",
            rank=1,
            score=0.95,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            "doc-b",
            rank=2,
            score=0.90,
            source=RetrievalSource.DENSE,
        ),
        make_result(
            "doc-c",
            rank=3,
            score=0.85,
            source=RetrievalSource.DENSE,
        ),
    ]


def make_sparse_results() -> list[RetrievalResult]:
    """
    Standard sparse result list.
    """

    return [
        make_result(
            "doc-b",
            rank=1,
            score=2.5,
            source=RetrievalSource.SPARSE,
        ),
        make_result(
            "doc-a",
            rank=2,
            score=2.0,
            source=RetrievalSource.SPARSE,
        ),
        make_result(
            "doc-d",
            rank=3,
            score=1.5,
            source=RetrievalSource.SPARSE,
        ),
    ]


# ==============================================================
# Constructor Tests
# ==============================================================


def test_rrf_fusion_uses_default_k():
    fusion = RRFFusion()

    assert fusion.k == 60


def test_rrf_fusion_accepts_custom_k():
    fusion = RRFFusion(k=10)

    assert fusion.k == 10


def test_rrf_fusion_rejects_zero_k():
    with pytest.raises(ValueError):
        RRFFusion(k=0)


def test_rrf_fusion_rejects_negative_k():
    with pytest.raises(ValueError):
        RRFFusion(k=-1)


def test_rrf_fusion_rejects_non_integer_k():
    with pytest.raises(ValueError):
        RRFFusion(k=1.5)


def test_rrf_fusion_rejects_boolean_k():
    with pytest.raises(ValueError):
        RRFFusion(k=True)


# ==============================================================
# Basic Fusion Tests
# ==============================================================


def test_fuse_dense_and_sparse_results():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=make_dense_results(),
        sparse_results=make_sparse_results(),
        top_k=5,
    )

    assert len(results) == 4


def test_fuse_returns_empty_for_empty_inputs():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=[],
        sparse_results=[],
        top_k=5,
    )

    assert results == []


def test_fuse_allows_empty_dense_results():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=[],
        sparse_results=make_sparse_results(),
        top_k=5,
    )

    assert len(results) == 3


def test_fuse_allows_empty_sparse_results():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=make_dense_results(),
        sparse_results=[],
        top_k=5,
    )

    assert len(results) == 3


# ==============================================================
# RRF Score Tests
# ==============================================================


def test_rrf_score_for_rank_one():
    fusion = RRFFusion(k=60)

    score = fusion._rrf_score(1)

    assert score == pytest.approx(
        1 / 61
    )


def test_rrf_score_for_rank_two():
    fusion = RRFFusion(k=60)

    score = fusion._rrf_score(2)

    assert score == pytest.approx(
        1 / 62
    )


def test_rrf_score_for_rank_three():
    fusion = RRFFusion(k=60)

    score = fusion._rrf_score(3)

    assert score == pytest.approx(
        1 / 63
    )


def test_rrf_score_rejects_zero_rank():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion._rrf_score(0)


# ==============================================================
# Overlap / Deduplication Tests
# ==============================================================


def test_fuse_deduplicates_overlapping_documents():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=make_dense_results(),
        sparse_results=make_sparse_results(),
        top_k=10,
    )

    document_ids = [
        result.document_id
        for result in results
    ]

    assert len(document_ids) == len(
        set(document_ids)
    )


def test_overlapping_document_receives_two_rrf_contributions():
    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        dense_results=[
            make_result(
                "doc-a",
                rank=1,
                source=RetrievalSource.DENSE,
            )
        ],
        sparse_results=[
            make_result(
                "doc-a",
                rank=1,
                source=RetrievalSource.SPARSE,
            )
        ],
        top_k=5,
    )

    assert len(results) == 1

    expected = (
        (1 / 61)
        + (1 / 61)
    )

    assert results[0].score == pytest.approx(
        expected
    )

    assert results[0].retrieval_scores[
        "rrf_score"
    ] == pytest.approx(expected)


def test_non_overlapping_documents_receive_single_contribution():
    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        dense_results=[
            make_result(
                "dense-only",
                rank=1,
                source=RetrievalSource.DENSE,
            )
        ],
        sparse_results=[
            make_result(
                "sparse-only",
                rank=1,
                source=RetrievalSource.SPARSE,
            )
        ],
        top_k=5,
    )

    assert len(results) == 2

    for result in results:
        assert result.score == pytest.approx(
            1 / 61
        )


# ==============================================================
# Ranking Tests
# ==============================================================


def test_fuse_ranks_results_by_rrf_score():
    fusion = RRFFusion(k=60)

    results = fusion.fuse(
        dense_results=[
            make_result(
                "doc-a",
                rank=1,
            ),
            make_result(
                "doc-b",
                rank=2,
            ),
        ],
        sparse_results=[
            make_result(
                "doc-b",
                rank=1,
                source=RetrievalSource.SPARSE,
            ),
            make_result(
                "doc-a",
                rank=2,
                source=RetrievalSource.SPARSE,
            ),
        ],
        top_k=2,
    )

    assert results[0].document_id in {
        "doc-a",
        "doc-b",
    }

    assert results[0].score >= results[1].score


def test_fuse_assigns_deterministic_final_ranks():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=make_dense_results(),
        sparse_results=make_sparse_results(),
        top_k=5,
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


def test_fuse_is_deterministic():
    fusion = RRFFusion()

    dense = make_dense_results()
    sparse = make_sparse_results()

    first = fusion.fuse(
        dense_results=dense,
        sparse_results=sparse,
        top_k=5,
    )

    second = fusion.fuse(
        dense_results=dense,
        sparse_results=sparse,
        top_k=5,
    )

    assert [
        result.document_id
        for result in first
    ] == [
        result.document_id
        for result in second
    ]

    assert [
        result.score
        for result in first
    ] == pytest.approx([
        result.score
        for result in second
    ])


# ==============================================================
# top_k Tests
# ==============================================================


def test_fuse_respects_top_k():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=make_dense_results(),
        sparse_results=make_sparse_results(),
        top_k=2,
    )

    assert len(results) == 2


def test_fuse_top_k_larger_than_results():
    fusion = RRFFusion()

    results = fusion.fuse(
        dense_results=make_dense_results(),
        sparse_results=make_sparse_results(),
        top_k=100,
    )

    assert len(results) == 4


def test_fuse_rejects_zero_top_k():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=[],
            sparse_results=[],
            top_k=0,
        )


def test_fuse_rejects_negative_top_k():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=[],
            sparse_results=[],
            top_k=-1,
        )


def test_fuse_rejects_non_integer_top_k():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=[],
            sparse_results=[],
            top_k=1.5,
        )


# ==============================================================
# Validation Tests
# ==============================================================


def test_fuse_rejects_none_dense_results():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=None,
            sparse_results=[],
            top_k=5,
        )


def test_fuse_rejects_none_sparse_results():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=[],
            sparse_results=None,
            top_k=5,
        )


def test_fuse_rejects_invalid_dense_result_type():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=["invalid"],
            sparse_results=[],
            top_k=5,
        )


def test_fuse_rejects_invalid_sparse_result_type():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=[],
            sparse_results=["invalid"],
            top_k=5,
        )


def test_fuse_rejects_string_dense_results():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results="invalid",
            sparse_results=[],
            top_k=5,
        )


def test_fuse_rejects_string_sparse_results():
    fusion = RRFFusion()

    with pytest.raises(ValueError):
        fusion.fuse(
            dense_results=[],
            sparse_results="invalid",
            top_k=5,
        )


# ==============================================================
# Preservation Tests
# ==============================================================


def test_fuse_preserves_metadata():
    fusion = RRFFusion()

    dense = [
        make_result(
            "doc-a",
            rank=1,
            metadata={
                "category": "payment",
                "source": "policy",
            },
        )
    ]

    results = fusion.fuse(
        dense_results=dense,
        sparse_results=[],
        top_k=5,
    )

    assert results[0].metadata == {
        "category": "payment",
        "source": "policy",
    }


def test_fuse_preserves_document_text():
    fusion = RRFFusion()

    dense = [
        make_result(
            "doc-a",
            rank=1,
        )
    ]

    results = fusion.fuse(
        dense_results=dense,
        sparse_results=[],
        top_k=5,
    )

    assert results[0].text == "Text for doc-a"


def test_fuse_adds_rrf_score_without_removing_existing_scores():
    fusion = RRFFusion()

    dense = [
        make_result(
            "doc-a",
            rank=1,
            score=0.95,
        )
    ]

    results = fusion.fuse(
        dense_results=dense,
        sparse_results=[],
        top_k=5,
    )

    assert results[0].retrieval_scores[
        "original_score"
    ] == 0.95

    assert "rrf_score" in (
        results[0].retrieval_scores
    )


def test_fuse_does_not_mutate_original_result():
    fusion = RRFFusion()

    original = make_result(
        "doc-a",
        rank=1,
        score=0.95,
    )

    original_rank = original.rank
    original_score = original.score

    fusion.fuse(
        dense_results=[original],
        sparse_results=[],
        top_k=5,
    )

    assert original.rank == original_rank
    assert original.score == original_score


# ==============================================================
# Factory Tests
# ==============================================================


def test_create_rrf_fusion_uses_default_configuration():
    fusion = create_rrf_fusion()

    assert isinstance(
        fusion,
        RRFFusion,
    )

    assert fusion.k == 60


def test_create_rrf_fusion_accepts_custom_k():
    fusion = create_rrf_fusion(k=20)

    assert fusion.k == 20