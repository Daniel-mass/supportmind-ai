"""
# agents.py

AutoGen-based Resolution Agent.

The Resolution Agent reasons over existing SupportMind signals and evidence
and produces a textual resolution recommendation.

It does not:
- perform retrieval
- run classification models
- execute customer-impacting actions
- make unsupported claims about unavailable evidence
"""

from __future__ import annotations

import json
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentError,
    ResolutionAgentState,
    ResolutionAgentStatus,
)
from app.agents.resolution_agent.output_parser import (
    ResolutionOutputParser,
)
from app.agents.resolution_agent.prompts import (
    RESOLUTION_AGENT_SYSTEM_PROMPT,
    build_resolution_prompt,
)


RESOLUTION_AGENT_NAME = "resolution_agent"


class ResolutionAgent:
    """AutoGen wrapper for the SupportMind Resolution Agent."""

    def __init__(
        self,
        model_client: Any,
        *,
        name: str = RESOLUTION_AGENT_NAME,
    ) -> None:
        if model_client is None:
            raise ValueError("model_client must not be None.")

        if not name or not name.strip():
            raise ValueError("name must not be empty.")

        self.name = name

        self._agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            system_message=RESOLUTION_AGENT_SYSTEM_PROMPT,
        )

    @property
    def agent(self) -> AssistantAgent:
        """Return the underlying AutoGen AssistantAgent."""
        return self._agent

    def build_task(
        self,
        state: ResolutionAgentState,
    ) -> str:
        """Build the task prompt from the current resolution state."""
        if not isinstance(state, ResolutionAgentState):
            raise TypeError(
                "state must be an instance of ResolutionAgentState."
            )

        return build_resolution_prompt(state)

    async def run(
        self,
        state: ResolutionAgentState,
    ) -> ResolutionAgentState:
        if not isinstance(state, ResolutionAgentState):
            raise TypeError(
                "state must be an instance of ResolutionAgentState."
            )

        state.status = ResolutionAgentStatus.RUNNING
        state.iteration_count += 1

        task = self.build_task(state)

        try:
            result = await self._agent.run(task=task)

            response = self._extract_final_response(result)

            if response is None:
                state.status = ResolutionAgentStatus.FAILED

                state.execution_metadata[
                    "resolution_failure"
                ] = {
                    "stage": "response_extraction",
                    "error_type": "empty_agent_response",
                    "error_message": (
                        "Resolution Agent completed without "
                        "producing a textual response."
                    ),
                }

                state.errors.append(
                    ResolutionAgentError(
                        source=self.name,
                        error_type="empty_agent_response",
                        message=(
                            "Resolution Agent completed without "
                            "producing a textual response."
                        ),
                        recoverable=False,
                    )
                )

                return state

            # Preserve the exact LLM response before any parsing
            # or validation occurs.
            #
            # This is intentionally stored before
            # ResolutionOutputParser.parse() because parsing may fail.
            state.execution_metadata[
                "agent_response"
            ] = response

            valid_evidence_ids = {
                evidence.evidence_id
                for evidence in state.evidence
            }

            recommendation = ResolutionOutputParser.parse(
                response,
                valid_evidence_ids=valid_evidence_ids,
            )

            state.recommendation = recommendation
            state.status = ResolutionAgentStatus.COMPLETED

            return state

        except Exception as exc:
            state.status = ResolutionAgentStatus.FAILED
            state.recommendation = None

            # ---------------------------------------------------------
            # Diagnostic boundary
            # ---------------------------------------------------------
            #
            # Do not allow the underlying exception details to disappear.
            #
            # The original LLM response is already preserved in
            # execution_metadata["agent_response"] before parsing.
            #
            # We additionally record the failure stage and, when
            # available, the structured Pydantic validation errors.
            # ---------------------------------------------------------

            failure_diagnostics: dict[str, Any] = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }

            agent_response = state.execution_metadata.get(
                "agent_response"
            )

            if agent_response is not None:
                failure_diagnostics[
                    "agent_response"
                ] = agent_response

            # ResolutionOutputParser wraps Pydantic ValidationError
            # inside ValueError. The string representation contains
            # the structured validation details generated by the parser.
            if (
                isinstance(exc, ValueError)
                and "schema validation" in str(exc)
            ):
                failure_diagnostics["stage"] = (
                    "pydantic_schema_validation"
                )

                # The parser deliberately embeds the exact Pydantic
                # validation errors as JSON after this marker.
                marker = "Validation errors: "

                if marker in str(exc):
                    raw_validation_errors = str(exc).split(
                        marker,
                        1,
                    )[1]

                    try:
                        failure_diagnostics[
                            "validation_errors"
                        ] = json.loads(
                            raw_validation_errors
                        )
                    except (json.JSONDecodeError, TypeError):
                        # Never allow diagnostic parsing itself to
                        # hide or replace the original failure.
                        failure_diagnostics[
                            "validation_errors"
                        ] = raw_validation_errors

            elif isinstance(exc, ValueError):
                failure_diagnostics["stage"] = (
                    "resolution_output_validation"
                )

            else:
                failure_diagnostics["stage"] = (
                    "resolution_agent_execution"
                )

            state.execution_metadata[
                "resolution_failure"
            ] = failure_diagnostics

            # Keep the workflow's existing error contract unchanged.
            state.errors.append(
                ResolutionAgentError(
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
        Extract the final textual response from an AutoGen task result.

        Messages are inspected in reverse order so the latest textual
        response is preferred.
        """
        messages = getattr(result, "messages", None)

        if not messages:
            return None

        for message in reversed(messages):
            if isinstance(message, TextMessage):
                content = message.content

                if (
                    isinstance(content, str)
                    and content.strip()
                ):
                    return content.strip()

        return None