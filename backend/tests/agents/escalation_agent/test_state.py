#test_state.py
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationAgentStatus,
    EscalationClassificationSignal,
    EscalationEvidence,
    EscalationRecommendation,
    EscalationResolutionSignal,
    EscalationSignal,
    EscalationType,
)


def test_escalation_signal_creation() -> None:
    signal = EscalationSignal(
        name="sla_breach_risk",
        value="high",
        confidence=0.91,
        source="workflow_intelligence",
    )

    assert signal.name == "sla_breach_risk"
    assert signal.value == "high"
    assert signal.confidence == 0.91
    assert signal.source == "workflow_intelligence"
    assert signal.metadata == {}


def test_escalation_classification_signal_creation() -> None:
    signal = EscalationClassificationSignal(
        name="priority",
        value="High",
        confidence=0.84,
        source="priority_detection_predictor",
    )

    assert signal.name == "priority"
    assert signal.value == "High"
    assert signal.confidence == 0.84


def test_escalation_resolution_signal_creation() -> None:
    resolution = EscalationResolutionSignal(
        resolution_type="troubleshooting",
        recommendation="Guide the customer through account recovery.",
        rationale="The evidence supports self-service recovery.",
        confidence=0.88,
        requires_human_review=False,
        proposed_actions=[
            "Verify account recovery state",
            "Guide password reset",
        ],
    )

    assert resolution.resolution_type == "troubleshooting"
    assert resolution.confidence == 0.88
    assert resolution.requires_human_review is False
    assert len(resolution.proposed_actions) == 2


def test_escalation_evidence_creation() -> None:
    evidence = EscalationEvidence(
        evidence_id="E1",
        source="graph_rag",
        content="Authentication incidents can require specialist review.",
        source_reference="incident_reports.pdf",
        score=0.92,
    )

    assert evidence.evidence_id == "E1"
    assert evidence.source == "graph_rag"
    assert evidence.score == 0.92


def test_escalation_recommendation_creation() -> None:
    recommendation = EscalationRecommendation(
        escalation_type=EscalationType.SPECIALIST_ESCALATION,
        should_escalate=True,
        reason="The issue requires specialist investigation.",
        rationale="Operational evidence indicates repeated unsuccessful resolution attempts.",
        recommended_team="Authentication Support",
        recommended_path="specialist_queue",
        evidence_ids=["E1", "E2"],
        confidence=0.87,
        requires_human_review=True,
        proposed_actions=[
            "Route ticket to authentication specialist queue",
        ],
    )

    assert recommendation.escalation_type is EscalationType.SPECIALIST_ESCALATION
    assert recommendation.should_escalate is True
    assert recommendation.recommended_team == "Authentication Support"
    assert recommendation.recommended_path == "specialist_queue"
    assert recommendation.evidence_ids == ["E1", "E2"]
    assert recommendation.requires_human_review is True


def test_escalation_agent_state_defaults() -> None:
    state = EscalationAgentState(
        request="Customer cannot log into the application."
    )

    assert state.status is EscalationAgentStatus.PENDING
    assert state.iteration_count == 0
    assert state.classification_signals == []
    assert state.operational_signals == []
    assert state.evidence == []
    assert state.resolution is None
    assert state.recommendation is None
    assert state.uncertainty_detected is False
    assert state.uncertainty_reasons == []
    assert state.errors == []

    assert state.successful is False
    assert state.has_evidence is False
    assert state.has_operational_signals is False
    assert state.has_recommendation is False


def test_escalation_agent_state_with_full_context() -> None:
    classification_signal = EscalationClassificationSignal(
        name="priority",
        value="High",
        confidence=0.82,
        source="priority_detection_predictor",
    )

    operational_signal = EscalationSignal(
        name="sla_breach_risk",
        value="high",
        confidence=0.94,
        source="workflow_intelligence",
    )

    resolution = EscalationResolutionSignal(
        resolution_type="troubleshooting",
        recommendation="Guide the customer through account recovery.",
        rationale="Evidence supports account recovery.",
        confidence=0.86,
        requires_human_review=False,
    )

    evidence = EscalationEvidence(
        evidence_id="E1",
        source="graph_rag",
        content="Repeated authentication failures may require specialist review.",
    )

    state = EscalationAgentState(
        request="Customer cannot log into the application.",
        ticket_id="TICKET-001",
        classification_signals=[classification_signal],
        resolution=resolution,
        operational_signals=[operational_signal],
        evidence=[evidence],
    )

    assert state.ticket_id == "TICKET-001"
    assert len(state.classification_signals) == 1
    assert len(state.operational_signals) == 1
    assert len(state.evidence) == 1
    assert state.resolution is not None

    assert state.has_evidence is True
    assert state.has_operational_signals is True


def test_escalation_state_successful_property() -> None:
    state = EscalationAgentState(
        request="Customer cannot log into the application.",
        status=EscalationAgentStatus.COMPLETED,
    )

    assert state.successful is True


def test_escalation_state_recommendation_property() -> None:
    recommendation = EscalationRecommendation(
        escalation_type=EscalationType.NO_ESCALATION,
        should_escalate=False,
        reason="The issue can be resolved through self-service.",
        rationale="Available evidence supports the existing resolution path.",
    )

    state = EscalationAgentState(
        request="Customer needs help resetting a password.",
        recommendation=recommendation,
    )

    assert state.has_recommendation is True
    assert state.recommendation is recommendation


def test_invalid_signal_confidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EscalationSignal(
            name="sla_breach_risk",
            value="high",
            confidence=1.5,
            source="workflow_intelligence",
        )


def test_invalid_recommendation_confidence_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EscalationRecommendation(
            escalation_type=EscalationType.HUMAN_REVIEW,
            should_escalate=True,
            reason="Human review is required.",
            rationale="Insufficient evidence.",
            confidence=-0.1,
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EscalationSignal(
            name="priority",
            value="High",
            source="classifier",
            unexpected_field="not_allowed",
        )


def test_empty_request_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EscalationAgentState(request="")