"""
Unit tests for SupportMind Agentic Layer evaluation metrics.

These tests verify that deterministic metrics correctly interpret
the real CoordinatorState and AgentResult contracts.

No LLM, Ollama, MLflow server, Qdrant, or Neo4j is required.
"""

from app.agents.coordinator.state import (
    AgentResult,
    CoordinatorState,
    EvidenceReference,
    WorkflowStatus,
)
from backend.mlflow.evaluation.metrics import (
    calculate_all_metrics,
    calculate_classification_success,
    calculate_evidence_validity,
    calculate_human_review_required,
    calculate_retrieval_success,
    calculate_workflow_success,
)

def build_successful_state() -> CoordinatorState:
    """Build a realistic completed CoordinatorState for testing."""

    return CoordinatorState(
        request="Why was my payment declined?",
        status=WorkflowStatus.COMPLETED,
        agent_results={
            "classification_agent": AgentResult(
                agent_name="classification_agent",
                success=True,
                result={
                    "category": "payment",
                    "priority": "high",
                },
            ),
            "retrieval_agent": AgentResult(
                agent_name="retrieval_agent",
                success=True,
                result={
                    "strategy": "hybrid",
                },
                evidence_ids=["E1", "E2"],
            ),
            "resolution_agent": AgentResult(
                agent_name="resolution_agent",
                success=True,
                result={
                    "resolution_type": "payment_retry",
                },
                evidence_ids=["E1"],
            ),
            "escalation_agent": AgentResult(
                agent_name="escalation_agent",
                success=True,
                result={
                    "should_escalate": False,
                },
                evidence_ids=["E1"],
            ),
            "response_composer_agent": AgentResult(
                agent_name="response_composer_agent",
                success=True,
                result={
                    "response": (
                        "Please retry the payment using "
                        "another supported method."
                    ),
                    "human_review_required": False,
                },
                evidence_ids=["E1"],
            ),
        },
        evidence=[
            EvidenceReference(
                evidence_id="E1",
                source="payment_policy",
                content="Supported payment methods are listed here.",
            ),
            EvidenceReference(
                evidence_id="E2",
                source="payment_troubleshooting",
                content="Payment troubleshooting guidance.",
            ),
        ],
        final_response=(
            "Please retry the payment using another "
            "supported method."
        ),
    )


def test_workflow_success_metric() -> None:
    """Completed workflow should produce a success metric of 1.0."""

    state = build_successful_state()

    assert calculate_workflow_success(state) == 1.0


def test_classification_success_metric() -> None:
    """Successful classification agent should produce 1.0."""

    state = build_successful_state()

    assert calculate_classification_success(state) == 1.0


def test_retrieval_success_metric() -> None:
    """Successful retrieval agent should produce 1.0."""

    state = build_successful_state()

    assert calculate_retrieval_success(state) == 1.0


def test_evidence_validity_metric() -> None:
    """All referenced evidence IDs should be valid."""

    state = build_successful_state()

    assert calculate_evidence_validity(state) == 1.0


def test_invalid_evidence_metric() -> None:
    """An invalid evidence ID should reduce evidence validity."""

    state = build_successful_state()

    state.agent_results[
        "resolution_agent"
    ].evidence_ids = ["E1", "E999"]

    assert calculate_evidence_validity(state) == 5/6
def test_human_review_not_required() -> None:
    """A response without human review should produce 0.0."""

    state = build_successful_state()

    assert calculate_human_review_required(state) == 0.0


def test_human_review_required() -> None:
    """A response requiring human review should produce 1.0."""

    state = build_successful_state()

    state.agent_results[
        "response_composer_agent"
    ].result["human_review_required"] = True

    assert calculate_human_review_required(state) == 1.0


def test_all_metrics() -> None:
    """All currently supported metrics should be calculated."""

    state = build_successful_state()

    metrics = calculate_all_metrics(state)

    assert metrics == {
        "workflow_success": 1.0,
        "classification_success": 1.0,
        "retrieval_success": 1.0,
        "resolution_success": 1.0,
        "escalation_success": 1.0,
        "response_generation_success": 1.0,
        "evidence_validity": 1.0,
        "human_review_required": 0.0,
    }