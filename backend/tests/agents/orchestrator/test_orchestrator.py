"""
Unit tests for the SupportMind Agent Orchestrator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationResult,
    ClassificationState,
    ClassificationStatus,
    ClassificationType,
)
from app.agents.coordinator.router import RoutingDecision
from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
    WorkflowStatus,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationAgentStatus,
    EscalationRecommendation,
    EscalationType,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionAgentStatus,
    ResolutionRecommendation,
    ResolutionType,
)
from app.agents.response_composer_agent.contracts.state import (
    ComposedResponse,
    ResponseComposerAgentState,
    ResponseComposerAgentStatus,
)
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
    RetrievalAgentStatus,
)
from app.agents.orchestrator.orchestrator import (
    AgentOrchestrator,
)


# ============================================================================
# Fake Coordinator
# ============================================================================


class FakeCoordinator:
    """Fake Coordinator used for orchestrator unit tests."""

    def __init__(self) -> None:
        self.run = AsyncMock(
            side_effect=self._run
        )

    async def _run(
        self,
        state: CoordinatorState,
    ) -> CoordinatorState:
        state.status = WorkflowStatus.RUNNING
        return state


# ============================================================================
# Fake Classification Agent
# ============================================================================


class FakeClassificationAgent:
    """Fake Classification Agent using the real state contract."""

    def run(
        self,
        state: ClassificationState,
    ) -> ClassificationState:
        state.status = ClassificationStatus.COMPLETED

        state.predictions = ClassificationResult(
            category=ClassificationPrediction(
                classification_type=ClassificationType.CATEGORY,
                value="billing",
                confidence=0.95,
                source="fake_classification_model",
            ),
            predictions_completed=1,
            predictions_requested=1,
            uncertainty_detected=False,
        )

        state.completed_predictions = [
            ClassificationType.CATEGORY,
        ]

        state.errors = []

        return state


# ============================================================================
# Fake Retrieval Agent
# ============================================================================


class FakeRetrievalAgent:
    """Fake Retrieval Agent using the real state contract."""

    async def run(
        self,
        state: RetrievalAgentState,
    ) -> RetrievalAgentState:
        state.status = RetrievalAgentStatus.COMPLETED

        state.evidence_ids = ["E1"]

        state.execution_metadata = {
            "evidence_count": 1,
            "source": "fake_retrieval",
        }

        state.errors = []

        return state


# ============================================================================
# Fake Resolution Agent
# ============================================================================


class FakeResolutionAgent:
    """Fake Resolution Agent using the real state contract."""

    async def run(
        self,
        state: ResolutionAgentState,
    ) -> ResolutionAgentState:
        state.status = ResolutionAgentStatus.COMPLETED

        state.recommendation = ResolutionRecommendation(
            resolution_type=ResolutionType.POLICY_BASED,
            recommendation=(
                "Explain the refund policy."
            ),
            rationale=(
                "The retrieved policy supports "
                "the recommendation."
            ),
            evidence_ids=["E1"],
            confidence=0.90,
            requires_human_review=False,
            proposed_actions=[],
        )

        state.errors = []

        return state


# ============================================================================
# Fake Escalation Agent
# ============================================================================


class FakeEscalationAgent:
    """Fake Escalation Agent using the real state contract."""

    async def run(
        self,
        state: EscalationAgentState,
    ) -> EscalationAgentState:
        state.status = EscalationAgentStatus.COMPLETED

        state.recommendation = EscalationRecommendation(
            escalation_type=EscalationType.NO_ESCALATION,
            should_escalate=False,
            reason=(
                "No escalation is required."
            ),
            rationale=(
                "The issue can be handled normally."
            ),
            recommended_team=None,
            recommended_path=None,
            evidence_ids=["E1"],
            confidence=0.95,
            requires_human_review=False,
            proposed_actions=[],
        )

        state.errors = []

        return state


# ============================================================================
# Fake Response Composer Agent
# ============================================================================


class FakeResponseComposerAgent:
    """Fake Response Composer Agent using the real state contract."""

    async def run(
        self,
        state: ResponseComposerAgentState,
    ) -> ResponseComposerAgentState:
        state.status = (
            ResponseComposerAgentStatus.COMPLETED
        )

        state.composed_response = ComposedResponse(
            response=(
                "Your refund request is covered by "
                "our refund policy."
            ),
            evidence_ids=["E1"],
            escalation_required=False,
            human_review_required=False,
        )

        state.errors = []

        return state


# ============================================================================
# Fake Router
# ============================================================================


class FakeRouter:
    """
    Deterministic router used to test the complete
    orchestration sequence.
    """

    def __init__(self) -> None:
        self.capabilities = [
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
            Capability.RESOLUTION,
            Capability.ESCALATION,
            Capability.GENERATION,
        ]

    def route(
        self,
        state: CoordinatorState,
    ) -> RoutingDecision:
        for capability in self.capabilities:
            if (
                capability
                not in state.completed_capabilities
            ):
                return RoutingDecision(
                    capability=capability,
                    reason=(
                        f"Execute {capability.value}."
                    ),
                    should_continue=True,
                    should_terminate=False,
                )

        return RoutingDecision(
            capability=None,
            reason="All capabilities completed.",
            should_continue=False,
            should_terminate=True,
        )


# ============================================================================
# Orchestrator Factory
# ============================================================================


def build_orchestrator() -> AgentOrchestrator:
    """Build an orchestrator using deterministic fake agents."""

    return AgentOrchestrator(
        coordinator=FakeCoordinator(),
        router=FakeRouter(),
        classification_agent=(
            FakeClassificationAgent()
        ),
        retrieval_agent=FakeRetrievalAgent(),
        resolution_agent=FakeResolutionAgent(),
        escalation_agent=FakeEscalationAgent(),
        response_composer_agent=(
            FakeResponseComposerAgent()
        ),
        max_iterations=10,
    )


# ============================================================================
# Complete Workflow
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_executes_complete_workflow():
    orchestrator = build_orchestrator()

    state = CoordinatorState(
        request=(
            "What is NovaCare's refund policy?"
        )
    )

    result = await orchestrator.run(state)

    # ------------------------------------------------------------------
    # If the workflow failed, expose the REAL orchestrator error.
    # ------------------------------------------------------------------

    assert result.status == WorkflowStatus.COMPLETED, (
        "Orchestrator failed.\n"
        f"Workflow errors: {result.errors}\n"
        f"Agent results: {result.agent_results}\n"
    )

    # ------------------------------------------------------------------
    # Every capability must execute exactly once.
    # ------------------------------------------------------------------

    assert result.completed_capabilities == [
        Capability.CLASSIFICATION,
        Capability.RETRIEVAL,
        Capability.RESOLUTION,
        Capability.ESCALATION,
        Capability.GENERATION,
    ]

    # ------------------------------------------------------------------
    # Every specialized agent must produce a result.
    # ------------------------------------------------------------------

    assert (
        "classification_agent"
        in result.agent_results
    )

    assert (
        "retrieval_agent"
        in result.agent_results
    )

    assert (
        "resolution_agent"
        in result.agent_results
    )

    assert (
        "escalation_agent"
        in result.agent_results
    )

    assert (
        "response_composer_agent"
        in result.agent_results
    )

    # ------------------------------------------------------------------
    # Every agent result must be successful.
    # ------------------------------------------------------------------

    for agent_name in [
        "classification_agent",
        "retrieval_agent",
        "resolution_agent",
        "escalation_agent",
        "response_composer_agent",
    ]:
        assert (
            result.agent_results[agent_name].success
            is True
        )

    # ------------------------------------------------------------------
    # Final response.
    # ------------------------------------------------------------------

    assert result.final_response == (
        "Your refund request is covered by "
        "our refund policy."
    )


# ============================================================================
# Agent Failure
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_stops_when_agent_fails():
    class FailingClassificationAgent:
        def run(
            self,
            state: ClassificationState,
        ) -> ClassificationState:
            state.status = ClassificationStatus.FAILED

            return state

    orchestrator = AgentOrchestrator(
        coordinator=FakeCoordinator(),
        router=FakeRouter(),
        classification_agent=(
            FailingClassificationAgent()
        ),
        retrieval_agent=FakeRetrievalAgent(),
        resolution_agent=FakeResolutionAgent(),
        escalation_agent=FakeEscalationAgent(),
        response_composer_agent=(
            FakeResponseComposerAgent()
        ),
    )

    state = CoordinatorState(
        request="Test request"
    )

    result = await orchestrator.run(state)

    assert result.status == WorkflowStatus.FAILED

    assert (
        Capability.CLASSIFICATION
        not in result.completed_capabilities
    )

    assert result.errors


# ============================================================================
# Invalid State
# ============================================================================


@pytest.mark.asyncio
async def test_orchestrator_rejects_invalid_state():
    orchestrator = build_orchestrator()

    with pytest.raises(TypeError):
        await orchestrator.run(
            "invalid state"
        )


# ============================================================================
# Invalid Max Iterations
# ============================================================================


def test_orchestrator_rejects_invalid_max_iterations():
    with pytest.raises(ValueError):
        AgentOrchestrator(
            coordinator=FakeCoordinator(),
            router=FakeRouter(),
            classification_agent=(
                FakeClassificationAgent()
            ),
            retrieval_agent=FakeRetrievalAgent(),
            resolution_agent=FakeResolutionAgent(),
            escalation_agent=FakeEscalationAgent(),
            response_composer_agent=(
                FakeResponseComposerAgent()
            ),
            max_iterations=0,
        )
