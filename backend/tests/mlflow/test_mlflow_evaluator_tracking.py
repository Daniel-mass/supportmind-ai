"""
Integration test for the SupportMind MLflow evaluator.

This test verifies that:

    CoordinatorState
        ↓
    evaluator.py
        ↓
    MLflow Tracking Server
        ↓
    logged run

No real agents or LLM calls are executed.
"""

import mlflow

from app.agents.coordinator.state import (
    AgentResult,
    CoordinatorState,
    EvidenceReference,
    WorkflowStatus,
)
from backend.mlflow.evaluation.evaluator import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    evaluate_and_track,
)


def build_evaluation_state() -> CoordinatorState:
    """Build a realistic completed CoordinatorState."""

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
                content=(
                    "Supported payment methods are listed here."
                ),
            ),
            EvidenceReference(
                evidence_id="E2",
                source="payment_troubleshooting",
                content=(
                    "Payment troubleshooting guidance."
                ),
            ),
        ],
        final_response=(
            "Please retry the payment using another "
            "supported method."
        ),
    )


def test_mlflow_evaluator_tracking() -> None:
    """
    Verify that the evaluator creates an MLflow run
    and records the expected metrics.
    """

    state = build_evaluation_state()

    result = evaluate_and_track(
        state,
        run_name="evaluator_tracking_test",
    )

    assert result["run_id"]
    assert result["experiment_id"]

    assert result["metrics"] == {
        "workflow_success": 1.0,
        "classification_success": 1.0,
        "retrieval_success": 1.0,
        "resolution_success": 1.0,
        "escalation_success": 1.0,
        "response_generation_success": 1.0,
        "evidence_validity": 1.0,
        "human_review_required": 0.0,
    }

    # Verify the run actually exists in MLflow.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(
        MLFLOW_EXPERIMENT_NAME
    )

    assert experiment is not None

    run = mlflow.get_run(result["run_id"])

    assert run.info.run_id == result["run_id"]

    assert run.data.metrics["workflow_success"] == 1.0
    assert run.data.metrics["classification_success"] == 1.0
    assert run.data.metrics["retrieval_success"] == 1.0
    assert run.data.metrics["resolution_success"] == 1.0
    assert run.data.metrics["escalation_success"] == 1.0
    assert run.data.metrics["response_generation_success"] == 1.0
    assert run.data.metrics["evidence_validity"] == 1.0
    assert run.data.metrics["human_review_required"] == 0.0

    assert run.data.params["project"] == "supportmind-ai"
    assert run.data.params["layer"] == "agentic"
    assert run.data.params["evaluation_type"] == "workflow"

    assert run.data.tags["project"] == "supportmind-ai"
    assert run.data.tags["layer"] == "agentic"
    assert run.data.tags["test_type"] == (
        "agentic_workflow_evaluation"
    )
    assert run.data.tags["environment"] == "local"
    assert run.data.tags["workflow_status"] == "completed"