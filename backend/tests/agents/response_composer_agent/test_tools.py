from __future__ import annotations

from app.agents.response_composer_agent.contracts.state import (
    ResponseAudience,
    ResponseClassificationSignal,
    ResponseComposerAgentState,
    ResponseEscalationSignal,
    ResponseEvidence,
    ResponseResolutionSignal,
    ResponseTone,
)
from app.agents.response_composer_agent.tools import (
    build_composition_context,
    summarize_upstream_signals,
    validate_composed_response,
    validate_evidence_references,
)


def build_test_state() -> ResponseComposerAgentState:
    """Create a representative composer state for unit tests."""

    return ResponseComposerAgentState(
        request="I cannot log into my account.",
        ticket_id="ticket-001",
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.PROFESSIONAL,
        classification_signals=[
            ResponseClassificationSignal(
                name="category",
                value="Account",
                confidence=0.91,
                source="classification_agent",
            ),
            ResponseClassificationSignal(
                name="sentiment",
                value="negative",
                confidence=0.88,
                source="sentiment_predictor",
            ),
        ],
        evidence=[
            ResponseEvidence(
                evidence_id="E1",
                source="account_management_guide",
                content=(
                    "Customers can reset their password "
                    "through the account recovery flow."
                ),
                source_reference="account_management_guide.pdf",
                score=0.94,
            ),
            ResponseEvidence(
                evidence_id="E2",
                source="faq_documents",
                content=(
                    "Account recovery may require "
                    "identity verification."
                ),
                source_reference="faq_documents.pdf",
                score=0.87,
            ),
        ],
        resolution=ResponseResolutionSignal(
            resolution_type="account_action",
            recommendation=(
                "Guide the customer through account recovery."
            ),
            rationale=(
                "The available account documentation "
                "supports the recovery flow."
            ),
            confidence=0.90,
            requires_human_review=False,
            proposed_actions=[
                "Provide account recovery instructions."
            ],
        ),
        escalation=ResponseEscalationSignal(
            escalation_type="account_support",
            should_escalate=False,
            reason="No escalation indicator was supplied.",
            rationale=(
                "The available resolution can be "
                "communicated directly."
            ),
            recommended_team=None,
            recommended_path=None,
            evidence_ids=["E1"],
            confidence=0.86,
            requires_human_review=False,
            proposed_actions=[],
        ),
    )


def test_validate_evidence_references_accepts_existing_ids():
    state = build_test_state()

    valid, missing = validate_evidence_references(
        evidence_ids=["E1", "E2"],
        state=state,
    )

    assert valid is True
    assert missing == []


def test_validate_evidence_references_rejects_unknown_ids():
    state = build_test_state()

    valid, missing = validate_evidence_references(
        evidence_ids=["E1", "E999"],
        state=state,
    )

    assert valid is False
    assert missing == ["E999"]


def test_validate_evidence_references_handles_empty_list():
    state = build_test_state()

    valid, missing = validate_evidence_references(
        evidence_ids=[],
        state=state,
    )

    assert valid is True
    assert missing == []


def test_summarize_upstream_signals():
    state = build_test_state()

    summary = summarize_upstream_signals(state)

    assert summary["request"] == (
        "I cannot log into my account."
    )
    assert summary["ticket_id"] == "ticket-001"
    assert summary["audience"] == "customer"
    assert summary["tone"] == "professional"

    assert len(summary["classification"]) == 2

    assert summary["classification"][0]["name"] == "category"
    assert summary["classification"][0]["value"] == "Account"
    assert (
        summary["classification"][0]["confidence"]
        == 0.91
    )

    assert len(summary["evidence"]) == 2
    assert summary["evidence"][0]["evidence_id"] == "E1"
    assert (
        summary["evidence"][0]["source"]
        == "account_management_guide"
    )

    assert summary["resolution"] is not None
    assert (
        summary["resolution"]["resolution_type"]
        == "account_action"
    )
    assert (
        summary["resolution"]["recommendation"]
        == "Guide the customer through account recovery."
    )

    assert summary["escalation"] is not None
    assert (
        summary["escalation"]["should_escalate"]
        is False
    )
    assert (
        summary["escalation"]["evidence_ids"]
        == ["E1"]
    )


def test_summarize_upstream_signals_handles_missing_optional_outputs():
    state = ResponseComposerAgentState(
        request="How do I update my email address?"
    )

    summary = summarize_upstream_signals(state)

    assert summary["classification"] == []
    assert summary["evidence"] == []
    assert summary["resolution"] is None
    assert summary["escalation"] is None


def test_build_composition_context_contains_constraints():
    state = build_test_state()

    context = build_composition_context(state)

    assert "constraints" in context

    constraints = context["constraints"]

    assert constraints["do_not_invent_facts"] is True
    assert constraints["do_not_invent_evidence"] is True
    assert constraints["do_not_change_resolution"] is True
    assert (
        constraints["do_not_change_escalation_decision"]
        is True
    )
    assert constraints["do_not_execute_actions"] is True


def test_build_composition_context_preserves_upstream_data():
    state = build_test_state()

    context = build_composition_context(state)

    assert context["request"] == (
        "I cannot log into my account."
    )

    assert context["ticket_id"] == "ticket-001"

    assert context["classification"][0]["value"] == "Account"

    assert context["evidence"][0]["evidence_id"] == "E1"

    assert (
        context["resolution"]["resolution_type"]
        == "account_action"
    )

    assert (
        context["escalation"]["should_escalate"]
        is False
    )


def test_validate_composed_response_accepts_valid_response():
    state = build_test_state()

    valid, errors = validate_composed_response(
        response=(
            "You can use the account recovery flow "
            "to regain access to your account."
        ),
        evidence_ids=["E1"],
        state=state,
    )

    assert valid is True
    assert errors == []


def test_validate_composed_response_rejects_empty_response():
    state = build_test_state()

    valid, errors = validate_composed_response(
        response="   ",
        evidence_ids=[],
        state=state,
    )

    assert valid is False
    assert "Response must not be empty." in errors


def test_validate_composed_response_rejects_unknown_evidence():
    state = build_test_state()

    valid, errors = validate_composed_response(
        response=(
            "Please follow the account recovery process."
        ),
        evidence_ids=["E999"],
        state=state,
    )

    assert valid is False
    assert "Unknown evidence IDs: E999" in errors


def test_validate_composed_response_rejects_empty_response_and_unknown_evidence():
    state = build_test_state()

    valid, errors = validate_composed_response(
        response="",
        evidence_ids=["E999"],
        state=state,
    )

    assert valid is False
    assert "Response must not be empty." in errors
    assert "Unknown evidence IDs: E999" in errors