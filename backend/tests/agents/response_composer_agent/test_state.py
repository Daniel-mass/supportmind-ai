from __future__ import annotations

import pytest

from app.agents.response_composer_agent.contracts.state import (
    ComposedResponse,
    ResponseAudience,
    ResponseClassificationSignal,
    ResponseComposerAgentState,
    ResponseComposerAgentStatus,
    ResponseEvidence,
    ResponseEscalationSignal,
    ResponseResolutionSignal,
    ResponseTone,
)


def test_state_defaults() -> None:
    state = ResponseComposerAgentState(
        request="Test request."
    )

    assert (
        state.status
        is ResponseComposerAgentStatus.PENDING
    )
    assert state.iteration_count == 0

    assert (
        state.audience
        is ResponseAudience.CUSTOMER
    )

    assert (
        state.tone
        is ResponseTone.PROFESSIONAL
    )

    assert state.classification_signals == []
    assert state.evidence == []
    assert state.resolution is None
    assert state.escalation is None
    assert state.composed_response is None

    assert state.uncertainty_detected is False
    assert state.uncertainty_reasons == []
    assert state.errors == []
    assert state.execution_metadata == {}

    assert state.successful is False
    assert state.has_evidence is False
    assert state.has_resolution is False
    assert state.has_escalation is False
    assert state.has_response is False


def test_state_requires_request() -> None:
    with pytest.raises(ValueError):
        ResponseComposerAgentState(
            request=""
        )


def test_classification_signal() -> None:
    signal = ResponseClassificationSignal(
        name="category",
        value="Account",
        confidence=0.91,
        source="classification_agent",
    )

    assert signal.name == "category"
    assert signal.value == "Account"
    assert signal.confidence == 0.91
    assert signal.source == "classification_agent"


def test_classification_signal_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError):
        ResponseClassificationSignal(
            name="category",
            value="Account",
            confidence=1.5,
            source="classification_agent",
        )


def test_response_evidence() -> None:
    evidence = ResponseEvidence(
        evidence_id="E1",
        source="account_management_guide",
        content=(
            "Customers can reset account credentials "
            "through the account recovery flow."
        ),
        source_reference="account_management_guide.pdf",
        score=0.94,
    )

    assert evidence.evidence_id == "E1"
    assert evidence.source == (
        "account_management_guide"
    )
    assert evidence.score == 0.94


def test_resolution_signal() -> None:
    resolution = ResponseResolutionSignal(
        resolution_type="account_action",
        recommendation=(
            "Guide the customer through account recovery."
        ),
        rationale=(
            "The supplied evidence supports "
            "self-service recovery."
        ),
        confidence=0.91,
        requires_human_review=False,
        proposed_actions=[
            "Use the account recovery flow.",
            "Retry login.",
        ],
    )

    assert resolution.resolution_type == (
        "account_action"
    )

    assert resolution.confidence == 0.91
    assert resolution.requires_human_review is False
    assert len(resolution.proposed_actions) == 2


def test_escalation_signal() -> None:
    escalation = ResponseEscalationSignal(
        escalation_type="no_escalation",
        should_escalate=False,
        reason=(
            "The supplied resolution supports "
            "self-service recovery."
        ),
        rationale=(
            "No supplied operational signal requires "
            "escalation."
        ),
        confidence=0.90,
        requires_human_review=False,
    )

    assert escalation.escalation_type == (
        "no_escalation"
    )
    assert escalation.should_escalate is False
    assert escalation.confidence == 0.90


def test_composed_response() -> None:
    response = ComposedResponse(
        response=(
            "Please use the account recovery flow "
            "and try signing in again."
        ),
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.PROFESSIONAL,
        evidence_ids=["E1"],
        escalation_required=False,
        human_review_required=False,
    )

    assert response.response
    assert (
        response.audience
        is ResponseAudience.CUSTOMER
    )
    assert (
        response.tone
        is ResponseTone.PROFESSIONAL
    )
    assert response.evidence_ids == ["E1"]
    assert response.escalation_required is False
    assert response.human_review_required is False


def test_state_with_upstream_outputs() -> None:
    state = ResponseComposerAgentState(
        request=(
            "I cannot log into my account."
        ),
        ticket_id="d1z0:186",
        classification_signals=[
            ResponseClassificationSignal(
                name="category",
                value="Account",
                confidence=0.91,
                source="classification_agent",
            )
        ],
        evidence=[
            ResponseEvidence(
                evidence_id="E1",
                source="account_management_guide",
                content="Use account recovery.",
                score=0.94,
            )
        ],
        resolution=ResponseResolutionSignal(
            resolution_type="account_action",
            recommendation=(
                "Use the account recovery flow."
            ),
            rationale=(
                "Supported by account management evidence."
            ),
            confidence=0.91,
        ),
        escalation=ResponseEscalationSignal(
            escalation_type="no_escalation",
            should_escalate=False,
            reason="Self-service resolution is supported.",
            rationale="No escalation signal was supplied.",
            confidence=0.90,
        ),
    )

    assert state.ticket_id == "d1z0:186"
    assert state.has_evidence is True
    assert state.has_resolution is True
    assert state.has_escalation is True
    assert state.has_response is False