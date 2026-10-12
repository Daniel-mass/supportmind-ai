from __future__ import annotations

import pytest

from app.agents.classification_agent.contracts.state import (
    ClassificationType,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionClassificationSignal,
    ResolutionEvidence,
)
from app.agents.resolution_agent.prompts import (
    RESOLUTION_AGENT_SYSTEM_PROMPT,
    build_resolution_context,
    build_resolution_prompt,
    build_resolution_system_prompt,
)


def build_state() -> ResolutionAgentState:
    return ResolutionAgentState(
        request="The customer was charged twice for the same subscription.",
        ticket_id="TICKET-1001",
        classification_signals=[
            ResolutionClassificationSignal(
                name=ClassificationType.CATEGORY.value,
                value="Billing",
                confidence=0.91,
                source="ticket_category_predictor",
                metadata={"model": "supportmind_ticket_category"},
            ),
            ResolutionClassificationSignal(
                name=ClassificationType.PRIORITY.value,
                value="High",
                confidence=0.82,
                source="priority_detection_predictor",
                metadata={"model": "priority_detection"},
            ),
            ResolutionClassificationSignal(
                name=ClassificationType.SENTIMENT.value,
                value="negative",
                confidence=0.95,
                source="sentiment_predictor",
            ),
        ],
        evidence=[
            ResolutionEvidence(
                evidence_id="E1",
                source="retrieval_sparse",
                content="Duplicate subscription charges should be reviewed "
                "against the applicable billing policy before refund processing.",
                source_reference="billing_policy.pdf",
                score=0.91,
                metadata={"evidence_type": "document"},
            ),
            ResolutionEvidence(
                evidence_id="E2",
                source="organization_intelligence",
                content="The support system records repeated billing-related "
                "tickets associated with duplicate charges.",
                source_reference="organization_metrics",
                score=0.74,
                metadata={"evidence_type": "metric"},
            ),
        ],
        uncertainty_detected=True,
        uncertainty_reasons=[
            "Priority prediction confidence below preferred threshold."
        ],
        iteration_count=2,
    )


def test_system_prompt_is_non_empty():
    assert RESOLUTION_AGENT_SYSTEM_PROMPT
    assert RESOLUTION_AGENT_SYSTEM_PROMPT.strip()


def test_system_prompt_defines_resolution_agent_role():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "resolution agent" in prompt
    assert "resolution recommendation" in prompt


def test_system_prompt_requires_evidence_based_reasoning():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "evidence-based" in prompt
    assert "retrieved knowledge" in prompt
    assert "grounded" in prompt


def test_system_prompt_treats_ml_predictions_as_advisory():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "advisory" in prompt
    assert "low-confidence" in prompt


def test_system_prompt_prevents_fabricated_information():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "do not invent" in prompt
    assert "fabricate evidence" in prompt


def test_system_prompt_prevents_unauthorized_actions():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "customer-impacting action" in prompt
    assert "silently execute" in prompt


def test_system_prompt_requires_human_review_when_appropriate():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "human review" in prompt


def test_system_prompt_defines_resolution_types():
    prompt = RESOLUTION_AGENT_SYSTEM_PROMPT.lower()

    assert "informational" in prompt
    assert "troubleshooting" in prompt
    assert "policy_based" in prompt
    assert "billing" in prompt
    assert "technical" in prompt
    assert "escalation_recommended" in prompt


def test_build_resolution_context_contains_request():
    state = build_state()

    context = build_resolution_context(state)

    assert state.request in context
    assert "TICKET-1001" in context


def test_build_resolution_context_contains_classification_signals():
    state = build_state()

    context = build_resolution_context(state)

    assert "Billing" in context
    assert "High" in context
    assert "negative" in context
    assert "0.9100" in context
    assert "0.8200" in context
    assert "0.9500" in context


def test_build_resolution_context_preserves_signal_source():
    state = build_state()

    context = build_resolution_context(state)

    assert "ticket_category_predictor" in context
    assert "priority_detection_predictor" in context
    assert "sentiment_predictor" in context


def test_build_resolution_context_contains_evidence():
    state = build_state()

    context = build_resolution_context(state)

    assert "[E1]" in context
    assert "[E2]" in context
    assert "billing_policy.pdf" in context
    assert "organization_metrics" in context


def test_build_resolution_context_preserves_evidence_metadata():
    state = build_state()

    context = build_resolution_context(state)

    assert "document" in context
    assert "metric" in context


def test_build_resolution_context_contains_uncertainty():
    state = build_state()

    context = build_resolution_context(state)

    assert "Uncertainty detected: True" in context
    assert "Priority prediction confidence below preferred threshold." in context


def test_build_resolution_context_contains_errors_when_present():
    state = build_state()

    from app.agents.resolution_agent.contracts.state import ResolutionAgentError

    state.errors.append(
        ResolutionAgentError(
            source="retrieval_agent",
            error_type="retrieval_timeout",
            message="Retrieval service timed out.",
            recoverable=True,
        )
    )

    context = build_resolution_context(state)

    assert "retrieval_agent" in context
    assert "retrieval_timeout" in context
    assert "Retrieval service timed out." in context


def test_build_resolution_context_handles_empty_signals_and_evidence():
    state = ResolutionAgentState(
        request="I need help with my account."
    )

    context = build_resolution_context(state)

    assert "No classification signals available." in context
    assert "No resolution evidence available." in context
    assert "No errors reported." in context


def test_build_resolution_prompt_contains_context():
    state = build_state()

    prompt = build_resolution_prompt(state)

    assert state.request in prompt
    assert "[E1]" in prompt
    assert "[E2]" in prompt
    assert "Billing" in prompt


def test_build_resolution_prompt_contains_resolution_task():
    state = build_state()

    prompt = build_resolution_prompt(state).lower()

    assert "evidence-based resolution recommendation" in prompt
    assert "appropriate resolution type" in prompt
    assert "human review" in prompt


def test_build_resolution_prompt_requires_evidence_ids():
    state = build_state()

    prompt = build_resolution_prompt(state).lower()

    assert "evidence ids" in prompt
    assert "cite supporting evidence" in prompt


def test_build_resolution_prompt_rejects_fabrication():
    state = build_state()

    prompt = build_resolution_prompt(state).lower()

    assert "do not invent information" in prompt


def test_build_resolution_system_prompt_returns_constant():
    assert build_resolution_system_prompt() == RESOLUTION_AGENT_SYSTEM_PROMPT


def test_prompt_builders_reject_invalid_state():
    with pytest.raises(TypeError):
        build_resolution_context("invalid")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        build_resolution_prompt("invalid")  # type: ignore[arg-type]