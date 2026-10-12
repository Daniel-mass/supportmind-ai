from __future__ import annotations

import inspect
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

from app.agents.escalation_agent.contracts.state import (
    EscalationAgentError,
    EscalationAgentState,
    EscalationAgentStatus,
)
from app.agents.escalation_agent.output_parser import (
    EscalationOutputParser,
)
from app.agents.escalation_agent.prompts import (
    ESCALATION_AGENT_SYSTEM_PROMPT,
    build_escalation_prompt,
)


ESCALATION_AGENT_NAME = "escalation_agent"


class EscalationAgent:
    """
    Agent responsible for reasoning about whether a support case
    should be escalated.

    The agent is a decision-support component.

    It does NOT:
    - execute escalations,
    - modify tickets,
    - assign teams,
    - recreate Intelligence Layer calculations,
    - invent operational signals,
    - invent evidence,
    - override supplied evidence with unsupported assumptions.

    It consumes signals and evidence already present in
    EscalationAgentState and uses the LLM to produce a structured
    EscalationRecommendation.
    """

    def __init__(
        self,
        model_client: Any,
        *,
        name: str = ESCALATION_AGENT_NAME,
    ) -> None:
        if model_client is None:
            raise ValueError(
                "model_client must not be None."
            )

        if not name or not name.strip():
            raise ValueError(
                "name must not be empty."
            )

        self.name = name

        self._agent = AssistantAgent(
            name=self.name,
            model_client=model_client,
            system_message=ESCALATION_AGENT_SYSTEM_PROMPT,
        )

    @property
    def agent(self) -> AssistantAgent:
        """Return the underlying AutoGen AssistantAgent."""
        return self._agent

    def build_task(
        self,
        state: EscalationAgentState,
    ) -> str:
        """
        Build the deterministic task prompt from the current state.
        """

        if not isinstance(
            state,
            EscalationAgentState,
        ):
            raise TypeError(
                "state must be an instance of EscalationAgentState."
            )

        return build_escalation_prompt(state)

    async def run(
        self,
        state: EscalationAgentState,
    ) -> EscalationAgentState:
        """
        Run the Escalation Agent against the supplied state.

        The method updates the existing state object and returns it.

        Successful execution requires BOTH:

        1. A textual LLM response.
        2. A valid EscalationRecommendation parsed from that response.

        A textual response alone is NOT considered successful.
        """

        if not isinstance(
            state,
            EscalationAgentState,
        ):
            raise TypeError(
                "state must be an instance of EscalationAgentState."
            )

        state.status = EscalationAgentStatus.RUNNING
        state.iteration_count += 1

        task = self.build_task(state)

        try:
            result = await self._agent.run(
                task=task
            )

            response = self._extract_final_response(
                result
            )

            if response is None:
                state.status = (
                    EscalationAgentStatus.FAILED
                )

                state.errors.append(
                    EscalationAgentError(
                        source=self.name,
                        error_type="empty_agent_response",
                        message=(
                            "Escalation Agent completed without "
                            "producing a textual response."
                        ),
                        recoverable=False,
                    )
                )

                return state

            # ----------------------------------------------------------
            # Preserve the raw model response for diagnostics.
            # ----------------------------------------------------------

            state.execution_metadata[
                "agent_response"
            ] = response

            state.execution_metadata[
                "raw_agent_response"
            ] = response

            # ----------------------------------------------------------
            # Build the canonical evidence-ID set.
            # ----------------------------------------------------------

            valid_evidence_ids = {
                evidence.evidence_id
                for evidence in state.evidence
            }

            state.execution_metadata[
                "valid_evidence_ids"
            ] = sorted(valid_evidence_ids)

            # ----------------------------------------------------------
            # Parse and validate structured recommendation.
            # ----------------------------------------------------------

            recommendation = (
                EscalationOutputParser.parse(
                    response,
                    valid_evidence_ids=valid_evidence_ids,
                )
            )

            # ----------------------------------------------------------
            # Store the structured recommendation.
            # ----------------------------------------------------------

            state.recommendation = recommendation

            # ----------------------------------------------------------
            # Successful completion means the structured contract
            # was actually satisfied.
            # ----------------------------------------------------------

            state.status = (
                EscalationAgentStatus.COMPLETED
            )

            return state

        except Exception as exc:
            state.status = (
                EscalationAgentStatus.FAILED
            )

            # Keep the raw response when available.
            # This is valuable for debugging parser failures.
            state.errors.append(
                EscalationAgentError(
                    source=self.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=False,
                )
            )

            # Record structured validation diagnostics.
            state.execution_metadata[
                "structured_output_error"
            ] = str(exc)

            return state

    @staticmethod
    def _extract_final_response(
        result: Any,
    ) -> str | None:
        """
        Extract the latest non-empty textual AutoGen response.
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

    async def close(self) -> None:
        """
        Close the underlying model client when supported.

        This method is intentionally defensive because AutoGen
        model clients may expose synchronous or asynchronous
        close methods depending on the implementation.
        """

        model_client = getattr(
            self._agent,
            "_model_client",
            None,
        )

        if model_client is None:
            return

        close_method = getattr(
            model_client,
            "close",
            None,
        )

        if close_method is None:
            return

        result = close_method()

        if inspect.isawaitable(result):
            await result