"""
#agents/coordinator/router.py
Coordinator routing logic.

This module decides which SupportMind capability should execute next.
It does not execute agents, tools, ML models, retrieval, or LLM calls.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
    WorkflowStatus,
)


class RoutingDecision(BaseModel):
    """
    Structured decision produced by the Coordinator router.
    """

    model_config = ConfigDict(extra="forbid")

    capability: Capability | None = None
    reason: str = Field(min_length=1)
    should_continue: bool
    should_terminate: bool


class CoordinatorRouter:
    """
    Deterministic routing layer for the Coordinator.

    The router controls which capabilities are allowed to execute next.
    It does not execute the selected capability.
    """

    def route(self, state: CoordinatorState) -> RoutingDecision:
        """
        Determine the next capability for the workflow.

        Routing priority:
        1. Failed workflows terminate.
        2. Completed workflows terminate.
        3. Explicitly requested capabilities are processed in order.
        4. Generation is selected after requested capabilities complete.
        5. Otherwise the workflow terminates.
        """

        if state.status == WorkflowStatus.FAILED:
            return RoutingDecision(
                capability=None,
                reason="Workflow has failed and cannot continue.",
                should_continue=False,
                should_terminate=True,
            )

        if state.status == WorkflowStatus.COMPLETED:
            return RoutingDecision(
                capability=None,
                reason="Workflow is already completed.",
                should_continue=False,
                should_terminate=True,
            )

        completed = set(state.completed_capabilities)

        for capability in state.requested_capabilities:
            if capability not in completed:
                return RoutingDecision(
                    capability=capability,
                    reason=(
                        f"Capability '{capability.value}' was requested "
                        "and has not yet been completed."
                    ),
                    should_continue=True,
                    should_terminate=False,
                )

        if (
            state.requested_capabilities
            and Capability.GENERATION not in completed
        ):
            return RoutingDecision(
                capability=Capability.GENERATION,
                reason=(
                    "All requested capabilities are complete; "
                    "generation can now produce the final response."
                ),
                should_continue=True,
                should_terminate=False,
            )

        return RoutingDecision(
            capability=None,
            reason="No remaining capability requires execution.",
            should_continue=False,
            should_terminate=True,
        )