"""
Focused test exposing the Resolution Agent ↔ Orchestrator
contract mismatch.

This test does NOT run:
    - Ollama
    - Qdrant
    - Neo4j
    - Graph-RAG
    - Phoenix
    - the full multi-agent workflow

It verifies only the contract between:

    ResolutionAgentState
        ↓
    AgentOrchestrator._merge_resolution_result()

The Resolution Agent contract allows a completed execution
without creating an unvalidated ResolutionRecommendation.

The Orchestrator currently rejects that state because it
requires a structured recommendation.
"""

from __future__ import annotations

import pytest

from app.agents.coordinator.state import CoordinatorState
from app.agents.orchestrator.orchestrator import AgentOrchestrator
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
)


def test_resolution_agent_contract_conflicts_with_orchestrator() -> None:
    """
    Expose the current Resolution Agent ↔ Orchestrator contract mismatch.

    ResolutionAgentState represents a valid Resolution Agent execution
    with no structured recommendation.

    The Orchestrator currently rejects that result because it requires
    result.recommendation to be non-None.
    """

    # --------------------------------------------------------------
    # Arrange
    # --------------------------------------------------------------

    coordinator_state = CoordinatorState(
        request="Customer is asking about a subscription refund.",
    )

    resolution_result = ResolutionAgentState(
        request="Customer is asking about a subscription refund.",
    )

    # --------------------------------------------------------------
    # Act + Assert
    # --------------------------------------------------------------

    with pytest.raises(
        RuntimeError,
        match=(
            "Resolution Agent completed without a "
            "structured recommendation."
        ),
    ):
        AgentOrchestrator._merge_resolution_result(
            AgentOrchestrator.__new__(AgentOrchestrator),
            state=coordinator_state,
            result=resolution_result,
        )