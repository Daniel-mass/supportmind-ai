from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
)
from app.agents.orchestrator.orchestrator import AgentOrchestrator
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionAgentStatus,
    ResolutionRecommendation,
    ResolutionType,
)


def build_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        coordinator=Mock(),
        router=Mock(),
        classification_agent=Mock(),
        retrieval_agent=Mock(),
        resolution_agent=Mock(),
        escalation_agent=Mock(),
        response_composer_agent=Mock(),
    )


def build_resolution_result() -> ResolutionAgentState:
    return ResolutionAgentState(
        request="Customer is asking about a subscription refund.",
        status=ResolutionAgentStatus.COMPLETED,
        recommendation=ResolutionRecommendation(
            resolution_type=ResolutionType.BILLING,
            recommendation=(
                "Apply the applicable subscription refund policy "
                "after verifying the customer's subscription status."
            ),
            rationale=(
                "The request concerns a subscription refund and "
                "requires policy-based billing handling."
            ),
            evidence_ids=["E1", "E2"],
            confidence=0.80,
            requires_human_review=True,
            proposed_actions=[
                "Verify subscription status",
                "Apply the refund policy if eligible",
            ],
        ),
    )


def test_merge_resolution_result_accepts_valid_recommendation() -> None:
    orchestrator = build_orchestrator()

    state = CoordinatorState(
        request="Customer is asking about a subscription refund.",
    )

    result = build_resolution_result()

    orchestrator._merge_resolution_result(
        state=state,
        result=result,
    )

    assert "resolution_agent" in state.agent_results

    agent_result = state.agent_results["resolution_agent"]

    assert agent_result.success is True
    assert agent_result.agent_name == "resolution_agent"

    assert agent_result.evidence_ids == ["E1", "E2"]

    assert agent_result.result["resolution_type"] == "billing"

    assert agent_result.result["recommendation"] == (
        "Apply the applicable subscription refund policy "
        "after verifying the customer's subscription status."
    )

    assert agent_result.result["confidence"] == 0.80

    assert Capability.RESOLUTION in state.completed_capabilities


def test_merge_resolution_result_preserves_evidence_ids() -> None:
    orchestrator = build_orchestrator()

    state = CoordinatorState(
        request="Customer is asking about a subscription refund.",
    )

    result = build_resolution_result()

    orchestrator._merge_resolution_result(
        state=state,
        result=result,
    )

    agent_result = state.agent_results["resolution_agent"]

    assert agent_result.evidence_ids == ["E1", "E2"]

    assert agent_result.result["evidence_ids"] == [
        "E1",
        "E2",
    ]


def test_merge_resolution_result_rejects_missing_recommendation() -> None:
    orchestrator = build_orchestrator()

    state = CoordinatorState(
        request="Customer is asking about a subscription refund.",
    )

    result = ResolutionAgentState(
        request="Customer is asking about a subscription refund.",
        status=ResolutionAgentStatus.COMPLETED,
    )

    with pytest.raises(
        RuntimeError,
        match="structured recommendation",
    ):
        orchestrator._merge_resolution_result(
            state=state,
            result=result,
        )

    assert "resolution_agent" not in state.agent_results
    assert Capability.RESOLUTION not in state.completed_capabilities


def test_merge_resolution_result_records_failed_agent() -> None:
    orchestrator = build_orchestrator()

    state = CoordinatorState(
        request="Customer is asking about a subscription refund.",
    )

    result = build_resolution_result()

    result.status = ResolutionAgentStatus.FAILED

    with pytest.raises(
        RuntimeError,
        match="Resolution Agent failed",
    ):
        orchestrator._merge_resolution_result(
            state=state,
            result=result,
        )

    assert "resolution_agent" in state.agent_results

    agent_result = state.agent_results["resolution_agent"]

    assert agent_result.agent_name == "resolution_agent"
    assert agent_result.success is False
    assert agent_result.evidence_ids == ["E1", "E2"]

    assert Capability.RESOLUTION not in state.completed_capabilities