"""
Unit tests for deterministic Graph-RAG groundedness validation.
"""

from __future__ import annotations

import pytest

from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGContext,
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GroundingStatus,
)
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def context() -> GraphRAGContext:
    return GraphRAGContext(
        query="Why was ticket ABC-123 delayed?",
        evidence=[
            GraphRAGEvidence(
                evidence_id="E1",
                source=(
                    GraphRAGEvidenceSource
                    .RETRIEVAL_GRAPH
                ),
                evidence_type=(
                    GraphRAGEvidenceType
                    .GRAPH_FACT
                ),
                text=(
                    "Ticket ABC-123 was assigned "
                    "to the Payments team."
                ),
                source_reference="ticket:ABC-123",
            ),
            GraphRAGEvidence(
                evidence_id="E2",
                source=(
                    GraphRAGEvidenceSource
                    .HELPDESK_GRAPH
                ),
                evidence_type=(
                    GraphRAGEvidenceType
                    .WORKFLOW_EVENT
                ),
                text=(
                    "Ticket ABC-123 spent 18 hours "
                    "waiting for approval."
                ),
                source_reference=(
                    "workflow:ABC-123:1"
                ),
            ),
            GraphRAGEvidence(
                evidence_id="E3",
                source=(
                    GraphRAGEvidenceSource
                    .ORGANIZATION_INTELLIGENCE
                ),
                evidence_type=(
                    GraphRAGEvidenceType
                    .METRIC
                ),
                text=(
                    "High-priority tickets have "
                    "elevated approval wait time."
                ),
                source_reference=(
                    "organization:approval-wait"
                ),
            ),
        ],
        selected_count=3,
        character_count=240,
    )


@pytest.fixture
def empty_context() -> GraphRAGContext:
    return GraphRAGContext(
        query="Why was ticket ABC-123 delayed?",
        evidence=[],
        selected_count=0,
        character_count=0,
    )


@pytest.fixture
def validator() -> GroundednessValidator:
    return GroundednessValidator()


# ============================================================================
# Citation extraction
# ============================================================================


def test_extract_citation_ids(
    validator: GroundednessValidator,
) -> None:
    answer = (
        "The ticket waited for approval [E2] "
        "and was assigned to Payments [E1]."
    )

    assert validator.extract_citation_ids(
        answer
    ) == [
        "E2",
        "E1",
    ]


def test_extract_citation_ids_preserves_duplicates(
    validator: GroundednessValidator,
) -> None:
    answer = (
        "The ticket was delayed [E2]. "
        "The same delay is confirmed [E2]."
    )

    assert validator.extract_citation_ids(
        answer
    ) == [
        "E2",
        "E2",
    ]


def test_extract_citation_ids_returns_empty_for_no_citations(
    validator: GroundednessValidator,
) -> None:
    assert (
        validator.extract_citation_ids(
            "There is insufficient evidence."
        )
        == []
    )


# ============================================================================
# Fully grounded
# ============================================================================


def test_validate_returns_grounded_for_valid_citations(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "The ticket was delayed during "
            "customer approval for 18 hours [E2] "
            "and was assigned to Payments [E1]."
        ),
        context=context,
    )

    assert result.status == (
        GroundingStatus.GROUNDED
    )

    assert result.valid is True

    assert result.cited_evidence_ids == [
        "E2",
        "E1",
    ]

    assert result.invalid_citation_ids == []


def test_validate_removes_duplicate_citation_ids(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "The ticket was delayed [E2]. "
            "This was related to approval [E2]."
        ),
        context=context,
    )

    assert result.status == (
        GroundingStatus.GROUNDED
    )

    assert result.cited_evidence_ids == [
        "E2"
    ]


# ============================================================================
# Missing citations
# ============================================================================


def test_validate_detects_missing_citations(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "The ticket was delayed during "
            "customer approval."
        ),
        context=context,
    )

    assert result.status == (
        GroundingStatus.PARTIALLY_GROUNDED
    )

    assert result.valid is False

    assert result.cited_evidence_ids == []

    assert (
        "no valid evidence citations"
        in result.issues[0]
    )


# ============================================================================
# Invalid citations
# ============================================================================


def test_validate_detects_unknown_evidence_id(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "The ticket was delayed [E99]."
        ),
        context=context,
    )

    assert result.status == (
        GroundingStatus.INVALID
    )

    assert result.valid is False

    assert result.invalid_citation_ids == [
        "E99"
    ]


def test_validate_detects_multiple_unknown_ids(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "The ticket was delayed [E99] "
            "because of [E100]."
        ),
        context=context,
    )

    assert result.status == (
        GroundingStatus.INVALID
    )

    assert result.invalid_citation_ids == [
        "E99",
        "E100",
    ]


def test_validate_detects_malformed_evidence_citation(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "The ticket was delayed [Eabc]."
        ),
        context=context,
    )

    assert result.status == (
        GroundingStatus.INVALID
    )

    assert result.invalid_citation_ids == [
        "Eabc"
    ]


# ============================================================================
# Empty evidence
# ============================================================================


def test_validate_returns_insufficient_evidence_when_context_is_empty(
    validator: GroundednessValidator,
    empty_context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer=(
            "I cannot determine the cause "
            "from the available information."
        ),
        context=empty_context,
    )

    assert result.status == (
        GroundingStatus.INSUFFICIENT_EVIDENCE
    )

    assert result.valid is False

    assert result.cited_evidence_ids == []

    assert (
        result.metadata["evidence_count"]
        == 0
    )


# ============================================================================
# Input validation
# ============================================================================


def test_validate_rejects_empty_answer(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    with pytest.raises(
        ValueError,
        match="answer cannot be empty",
    ):
        validator.validate(
            answer="   ",
            context=context,
        )


def test_validate_rejects_invalid_answer_type(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    with pytest.raises(
        TypeError,
        match="answer must be a string",
    ):
        validator.validate(
            answer=123,  # type: ignore[arg-type]
            context=context,
        )


def test_validate_rejects_invalid_context(
    validator: GroundednessValidator,
) -> None:
    with pytest.raises(
        TypeError,
        match="GraphRAGContext",
    ):
        validator.validate(
            answer="Some answer [E1].",
            context="invalid",  # type: ignore[arg-type]
        )


# ============================================================================
# Metadata
# ============================================================================


def test_validate_returns_grounding_metadata(
    validator: GroundednessValidator,
    context: GraphRAGContext,
) -> None:
    result = validator.validate(
        answer="Approval caused the delay [E2].",
        context=context,
    )

    assert result.metadata == {
        "evidence_count": 3,
        "citation_count": 1,
    }