"""
Coordinator Agent.

This module provides the AutoGen-backed Coordinator Agent for SupportMind AI.

Responsibilities:
- Accept a CoordinatorState.
- Build the Coordinator prompt.
- Execute one Coordinator reasoning turn through AutoGen.
- Extract the textual reasoning result.
- Return the updated CoordinatorState.

The Coordinator does NOT execute specialized agents.

IMPORTANT:
WorkflowStatus.COMPLETED here means that the Coordinator's
current reasoning turn completed successfully.

The Orchestrator is responsible for interpreting that result
and continuing the multi-agent workflow.
"""

from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from app.agents.coordinator.prompts import (
    COORDINATOR_SYSTEM_PROMPT,
    build_coordinator_prompt,
)
from app.agents.coordinator.state import (
    CoordinatorState,
    WorkflowError,
    WorkflowStatus,
)


COORDINATOR_AGENT_NAME = "coordinator_agent"


class CoordinatorAgent:
    """
    AutoGen-backed Coordinator Agent.

    The Coordinator performs one reasoning turn.

    A successful Coordinator reasoning turn is marked
    WorkflowStatus.COMPLETED to preserve the Coordinator's
    existing state contract.

    The Orchestrator will normalize this back to RUNNING
    when additional capabilities still need to execute.
    """

    def __init__(
        self,
        model_client: Any,
        *,
        name: str = COORDINATOR_AGENT_NAME,
    ) -> None:
        if model_client is None:
            raise ValueError(
                "model_client must not be None."
            )

        if not name.strip():
            raise ValueError(
                "name must not be empty."
            )

        self.name = name

        self._agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            system_message=COORDINATOR_SYSTEM_PROMPT,
        )

    @property
    def agent(self) -> AssistantAgent:
        """Return the underlying AutoGen AssistantAgent."""

        return self._agent

    def build_task(
        self,
        state: CoordinatorState,
    ) -> str:
        """
        Build the state-aware task sent to AutoGen.
        """

        if not isinstance(
            state,
            CoordinatorState,
        ):
            raise TypeError(
                "state must be an instance of CoordinatorState."
            )

        return build_coordinator_prompt(
            state
        )

    async def run(
        self,
        state: CoordinatorState,
    ) -> CoordinatorState:
        """
        Execute one Coordinator reasoning turn.

        A successful Coordinator reasoning turn:

        1. increments iteration_count;
        2. sets status to RUNNING while executing;
        3. executes AutoGen;
        4. extracts the textual response;
        5. stores the response;
        6. marks the Coordinator turn COMPLETED.

        The Orchestrator is responsible for deciding whether
        the complete multi-agent workflow is finished.
        """

        if not isinstance(
            state,
            CoordinatorState,
        ):
            raise TypeError(
                "state must be an instance of CoordinatorState."
            )

        # One iteration represents one Coordinator reasoning turn.
        state.iteration_count += 1

        state.status = WorkflowStatus.RUNNING

        task = self.build_task(state)

        try:
            result = await self._agent.run(
                task=task
            )

            response = self._extract_final_response(
                result
            )

            if response is None:
                state.status = WorkflowStatus.FAILED

                state.errors.append(
                    WorkflowError(
                        source=self.name,
                        error_type="empty_agent_response",
                        message=(
                            "Coordinator Agent completed without "
                            "producing a textual response."
                        ),
                        recoverable=False,
                    )
                )

                return state

            # Store the Coordinator's reasoning output.
            state.final_response = response

            # IMPORTANT:
            # COMPLETED means the Coordinator reasoning turn
            # completed successfully. The Orchestrator will
            # normalize this to RUNNING if more capabilities
            # need to execute.
            state.status = WorkflowStatus.COMPLETED

            return state

        except Exception as exc:
            state.status = WorkflowStatus.FAILED
            state.final_response = None

            state.errors.append(
                WorkflowError(
                    source=self.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=False,
                )
            )

            return state

    @staticmethod
    def _extract_final_response(
        result: Any,
    ) -> str | None:
        """
        Extract the final textual response from AutoGen's TaskResult.
        """

        messages = getattr(
            result,
            "messages",
            None,
        )

        if not messages:
            return None

        for message in reversed(messages):
            if isinstance(
                message,
                TextMessage,
            ):
                content = message.content

                if (
                    isinstance(content, str)
                    and content.strip()
                ):
                    return content.strip()

        return None