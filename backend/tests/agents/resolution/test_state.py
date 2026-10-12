from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentError,
    ResolutionAgentState,
    ResolutionAgentStatus,
    ResolutionClassificationSignal,
    ResolutionEvidence,
    ResolutionRecommendation,
    ResolutionType,
)


# ============================================================================
# ResolutionClassificationSignal
# ============================================================================


def test_classification_signal_defaults() -> None:
    signal = ResolutionClassificationSignal(
        name="priority",
        value="High",
        source="priority_detection",
    )

    assert signal.name == "priority"
    assert signal.value == "High"
    assert signal.confidence is None
    assert signal.source == "priority_detection"
    assert signal.metadata == {}


def test_classification_signal_accepts_confidence() -> None:
    signal = ResolutionClassificationSignal(
        name="sentiment",
        value="negative",
        confidence=0.91,
        source="sentiment_predictor",
    )

    assert signal.confidence == 0.91


def test_classification_signal_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ResolutionClassificationSignal(
            name="priority",
            value="High",
            confidence=1.5,
            source="priority_detection",
        )


def test_classification_signal_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ResolutionClassificationSignal(
            name="priority",
            value="High",
            source="priority_detection",
            unexpected_field="not_allowed",
        )


# ============================================================================
# ResolutionEvidence
# ============================================================================


def test_resolution_evidence_defaults() -> None:
    evidence = ResolutionEvidence(
        evidence_id="E1",
        source="retrieval",
        content="Refunds are allowed under the applicable policy.",
    )

    assert evidence.evidence_id == "E1"
    assert evidence.source == "retrieval"
    assert evidence.content.startswith("Refunds")
    assert evidence.source_reference is None
    assert evidence.score is None
    assert evidence.metadata == {}


def test_resolution_evidence_accepts_metadata() -> None:
    evidence = ResolutionEvidence(
        evidence_id="E2",
        source="graph_rag",
        content="Ticket priority is High.",
        source_reference="ticket:123",
        score=0.87,
        metadata={
            "ticket_id": "123",
        },
    )

    assert evidence.source_reference == "ticket:123"
    assert evidence.score == 0.87
    assert evidence.metadata["ticket_id"] == "123"


def test_resolution_evidence_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        ResolutionEvidence(
            evidence_id="E1",
            source="retrieval",
            content="",
        )


# ============================================================================
# ResolutionRecommendation
# ============================================================================


def test_resolution_recommendation_defaults() -> None:
    recommendation = ResolutionRecommendation(
        recommendation="Provide the customer with the applicable refund policy.",
        rationale="The retrieved policy evidence supports this response.",
    )

    assert (
        recommendation.resolution_type
        == ResolutionType.UNKNOWN
    )

    assert (
        recommendation.recommendation
        == "Provide the customer with the applicable refund policy."
    )

    assert (
        recommendation.rationale
        == "The retrieved policy evidence supports this response."
    )

    assert recommendation.evidence_ids == []
    assert recommendation.confidence is None
    assert recommendation.requires_human_review is False
    assert recommendation.proposed_actions == []


def test_resolution_recommendation_accepts_structured_output() -> None:
    recommendation = ResolutionRecommendation(
        resolution_type=ResolutionType.POLICY_BASED,
        recommendation="Apply the refund policy.",
        rationale="The customer's request matches the documented policy.",
        evidence_ids=["E1", "E4"],
        confidence=0.88,
        requires_human_review=True,
        proposed_actions=[
            "Verify transaction details.",
            "Confirm refund eligibility.",
        ],
    )

    assert (
        recommendation.resolution_type
        == ResolutionType.POLICY_BASED
    )

    assert recommendation.evidence_ids == [
        "E1",
        "E4",
    ]

    assert recommendation.confidence == 0.88
    assert recommendation.requires_human_review is True
    assert len(recommendation.proposed_actions) == 2


def test_resolution_recommendation_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ResolutionRecommendation(
            recommendation="Investigate the issue.",
            rationale="Evidence is insufficient.",
            confidence=-0.1,
        )


# ============================================================================
# ResolutionAgentError
# ============================================================================


def test_resolution_agent_error() -> None:
    error = ResolutionAgentError(
        source="resolution_agent",
        error_type="insufficient_evidence",
        message="No reliable evidence was available.",
        recoverable=True,
    )

    assert error.source == "resolution_agent"
    assert error.error_type == "insufficient_evidence"
    assert error.message == "No reliable evidence was available."
    assert error.recoverable is True


# ============================================================================
# ResolutionAgentState
# ============================================================================


def test_resolution_agent_state_defaults() -> None:
    state = ResolutionAgentState(
        request="Customer reports a duplicate charge."
    )

    assert (
        state.status
        == ResolutionAgentStatus.PENDING
    )

    assert state.ticket_id is None
    assert state.classification_signals == []
    assert state.evidence == []
    assert state.recommendation is None
    assert state.iteration_count == 0
    assert state.uncertainty_detected is False
    assert state.uncertainty_reasons == []
    assert state.errors == []
    assert state.execution_metadata == {}

    assert state.successful is False
    assert state.has_evidence is False
    assert state.has_recommendation is False


def test_resolution_agent_state_accepts_complete_state() -> None:
    signal = ResolutionClassificationSignal(
        name="priority",
        value="High",
        confidence=0.82,
        source="priority_detection",
    )

    evidence = ResolutionEvidence(
        evidence_id="E1",
        source="retrieval",
        content="Duplicate charges require transaction verification.",
    )

    recommendation = ResolutionRecommendation(
        resolution_type=ResolutionType.BILLING,
        recommendation="Verify the duplicate transaction before refund processing.",
        rationale="The retrieved billing evidence supports transaction verification.",
        evidence_ids=["E1"],
        confidence=0.84,
        requires_human_review=True,
    )

    state = ResolutionAgentState(
        request="Customer reports a duplicate charge.",
        ticket_id="TICKET-1001",
        classification_signals=[signal],
        evidence=[evidence],
        recommendation=recommendation,
        status=ResolutionAgentStatus.COMPLETED,
        iteration_count=1,
        uncertainty_detected=True,
        uncertainty_reasons=[
            "Priority confidence below threshold."
        ],
        execution_metadata={
            "evidence_count": 1,
        },
    )

    assert state.ticket_id == "TICKET-1001"

    assert len(
        state.classification_signals
    ) == 1

    assert state.classification_signals[0].value == "High"

    assert state.has_evidence is True
    assert state.has_recommendation is True
    assert state.successful is True

    assert (
        state.uncertainty_reasons[0]
        == "Priority confidence below threshold."
    )

    assert (
        state.execution_metadata["evidence_count"]
        == 1
    )


def test_resolution_agent_state_status_helpers() -> None:
    state = ResolutionAgentState(
        request="Investigate the customer's billing issue."
    )

    assert state.successful is False

    state.status = ResolutionAgentStatus.RUNNING

    assert state.successful is False

    state.status = ResolutionAgentStatus.COMPLETED

    assert state.successful is True


def test_resolution_agent_state_rejects_negative_iteration_count() -> None:
    with pytest.raises(ValidationError):
        ResolutionAgentState(
            request="Investigate the issue.",
            iteration_count=-1,
        )


def test_resolution_agent_state_rejects_empty_request() -> None:
    with pytest.raises(ValidationError):
        ResolutionAgentState(
            request="",
        )


def test_resolution_agent_state_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ResolutionAgentState(
            request="Investigate the issue.",
            unknown_field="not_allowed",
        )