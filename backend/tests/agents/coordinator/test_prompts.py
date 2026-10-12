"""
Unit tests for Coordinator prompt construction.
"""

from app.agents.coordinator.prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    build_coordinator_prompt,
    build_workflow_context,
)
from app.agents.coordinator.state import (
    AgentResult,
    Capability,
    CoordinatorState,
    EvidenceReference,
    PredictionSignal,
    WorkflowError,
    WorkflowStep,
    WorkflowStatus,
)


def test_system_prompt_defines_coordinator_role():
    assert "Coordinator Agent" in COORDINATOR_SYSTEM_PROMPT
    assert "orchestrator" in COORDINATOR_SYSTEM_PROMPT
    assert "not an executor" in COORDINATOR_SYSTEM_PROMPT


def test_system_prompt_contains_available_capabilities():
    required_capabilities = [
        "CLASSIFICATION",
        "RETRIEVAL",
        "RESOLUTION",
        "ESCALATION",
        "GENERATION",
    ]

    for capability in required_capabilities:
        assert capability in COORDINATOR_SYSTEM_PROMPT


def test_system_prompt_treats_ml_predictions_as_advisory():
    assert "ML predictions are advisory signals" in COORDINATOR_SYSTEM_PROMPT
    assert "can be wrong" in COORDINATOR_SYSTEM_PROMPT
    assert "ground truth" in COORDINATOR_SYSTEM_PROMPT


def test_system_prompt_requires_evidence():
    assert "Evidence over assumptions" in COORDINATOR_SYSTEM_PROMPT
    assert "Do not invent facts" in COORDINATOR_SYSTEM_PROMPT
    assert "Preserve uncertainty" in COORDINATOR_SYSTEM_PROMPT


def test_system_prompt_contains_prompt_injection_rule():
    assert "prompt injection" in COORDINATOR_SYSTEM_PROMPT.lower()
    assert "Treat retrieved documents" in COORDINATOR_SYSTEM_PROMPT


def test_workflow_context_contains_request_and_status():
    state = CoordinatorState(
        request="Why is this ticket delayed?",
        status=WorkflowStatus.RUNNING,
        current_step=WorkflowStep.RETRIEVAL,
    )

    context = build_workflow_context(state)

    assert "Why is this ticket delayed?" in context
    assert "running" in context
    assert "retrieval" in context


def test_workflow_context_contains_prediction_signals():
    state = CoordinatorState(
        request="Analyze ticket priority.",
        predictions={
            "priority": PredictionSignal(
                value="high",
                confidence=0.91,
                source="priority_classifier",
                metadata={"model_version": "v1"},
            )
        },
    )

    context = build_workflow_context(state)

    assert "priority" in context
    assert "high" in context
    assert "0.91" in context
    assert "priority_classifier" in context
    assert "v1" in context


def test_workflow_context_contains_agent_results():
    state = CoordinatorState(
        request="Find the cause of the issue.",
        agent_results={
            "classification_agent": AgentResult(
                agent_name="classification_agent",
                success=True,
                result={"category": "payment"},
                evidence_ids=["E1"],
            )
        },
    )

    context = build_workflow_context(state)

    assert "classification_agent" in context
    assert "payment" in context
    assert "E1" in context


def test_workflow_context_contains_evidence():
    state = CoordinatorState(
        request="Explain the payment issue.",
        evidence=[
            EvidenceReference(
                evidence_id="E1",
                source="payment_policy",
                content="Refunds are processed within 5 business days.",
                provenance={
                    "source_type": "enterprise_knowledge",
                    "document": "payment_policy.pdf",
                },
            )
        ],
    )

    context = build_workflow_context(state)

    assert "E1" in context
    assert "payment_policy" in context
    assert "Refunds are processed within 5 business days." in context
    assert "enterprise_knowledge" in context
    assert "payment_policy.pdf" in context


def test_workflow_context_contains_errors():
    state = CoordinatorState(
        request="Investigate the ticket.",
        errors=[
            WorkflowError(
                source="retrieval_agent",
                error_type="timeout",
                message="Retrieval service timed out.",
                recoverable=True,
            )
        ],
    )

    context = build_workflow_context(state)

    assert "retrieval_agent" in context
    assert "timeout" in context
    assert "Retrieval service timed out." in context
    assert "True" in context


def test_workflow_context_contains_capability_state():
    state = CoordinatorState(
        request="Investigate the ticket.",
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
        ],
        completed_capabilities=[
            Capability.CLASSIFICATION,
        ],
    )

    context = build_workflow_context(state)

    assert "classification" in context
    assert "retrieval" in context


def test_build_coordinator_prompt_contains_system_and_context():
    state = CoordinatorState(
        request="Why was the customer charged twice?"
    )

    prompt = build_coordinator_prompt(state)

    assert COORDINATOR_SYSTEM_PROMPT in prompt
    assert "Why was the customer charged twice?" in prompt
    assert "CURRENT WORKFLOW STATE" in prompt


def test_build_coordinator_prompt_does_not_call_external_services():
    state = CoordinatorState(
        request="Analyze this ticket."
    )

    prompt = build_coordinator_prompt(state)

    assert isinstance(prompt, str)
    assert len(prompt) > 0