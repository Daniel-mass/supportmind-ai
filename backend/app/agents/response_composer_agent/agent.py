#agent.py
from __future__ import annotations

from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.response_composer_agent.contracts.state import (
    ComposedResponse,
    ResponseComposerAgentError,
    ResponseComposerAgentState,
)
from app.agents.response_composer_agent.prompts import (
    build_composition_instruction,
    build_system_prompt,
)
from app.agents.response_composer_agent.tools import (
    build_composition_context,
    validate_composed_response,
)


class ResponseComposerAgent:
    """
    AutoGen-based Response Composer Agent.

    Responsibilities:
    - consume validated upstream outputs
    - compose the final user-facing response
    - preserve upstream resolution and escalation decisions
    - validate evidence references structurally

    This agent does NOT:
    - classify tickets
    - retrieve evidence
    - calculate intelligence metrics
    - create a new resolution
    - create a new escalation decision
    - execute actions
    """

    name = "response_composer_agent"

    def __init__(
        self,
        model_client: OllamaChatCompletionClient,
    ) -> None:
        self.model_client = model_client

        self.agent = AssistantAgent(
            name=self.name,
            model_client=self.model_client,
            system_message=build_system_prompt(),
        )

    def build_task(
        self,
        state: ResponseComposerAgentState,
    ) -> str:
        """
        Build the runtime task for the AutoGen agent.

        All business information comes from the existing state.
        """

        context = build_composition_context(state)

        instruction = build_composition_instruction(
            audience=state.audience,
            tone=state.tone,
        )

        return (
            f"{instruction}\n\n"
            "SUPPORTMIND CONTEXT:\n"
            f"{context}\n\n"
            "Compose only the final response text."
        )

    async def run(
        self,
        state: ResponseComposerAgentState,
    ) -> ResponseComposerAgentState:
        """
        Execute the Response Composer Agent.

        The supplied state is updated in place and returned.
        """

        state.status = state.status.RUNNING
        state.iteration_count += 1

        try:
            task = self.build_task(state)

            result = await self.agent.run(
                task=task,
                cancellation_token=CancellationToken(),
            )

            response_text = self._extract_response_text(result)

            if not response_text:
                raise ValueError(
                    "Response Composer Agent returned an empty response."
                )

            evidence_ids = self._extract_evidence_ids(
                state=state,
                response_text=response_text,
            )

            valid, validation_errors = validate_composed_response(
                response=response_text,
                evidence_ids=evidence_ids,
                state=state,
            )

            if not valid:
                raise ValueError(
                    "; ".join(validation_errors)
                )

            escalation_required = (
                state.escalation.should_escalate
                if state.escalation is not None
                else False
            )

            human_review_required = (
                (
                    state.resolution.requires_human_review
                    if state.resolution is not None
                    else False
                )
                or (
                    state.escalation.requires_human_review
                    if state.escalation is not None
                    else False
                )
            )

            state.composed_response = ComposedResponse(
                response=response_text,
                audience=state.audience,
                tone=state.tone,
                evidence_ids=evidence_ids,
                escalation_required=escalation_required,
                human_review_required=human_review_required,
                metadata={
                    "agent": self.name,
                    "iteration": state.iteration_count,
                },
            )

            state.status = state.status.COMPLETED

            state.execution_metadata.update(
                {
                    "agent": self.name,
                    "response_length": len(response_text),
                    "evidence_count": len(evidence_ids),
                }
            )

            return state

        except Exception as exc:
            state.status = state.status.FAILED

            state.errors.append(
                ResponseComposerAgentError(
                    source=self.name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=False,
                )
            )

            return state

    @staticmethod
    def _extract_response_text(
        result: Any,
    ) -> str:
        """
        Extract the final text response from AutoGen's result.

        Only TextMessage content is accepted.
        """

        messages = getattr(result, "messages", None)

        if not messages:
            return ""

        for message in reversed(messages):
            if isinstance(message, TextMessage):
                content = message.content

                if isinstance(content, str):
                    return content.strip()

        return ""

    @staticmethod
    def _extract_evidence_ids(
        state: ResponseComposerAgentState,
        response_text: str,
    ) -> list[str]:
        """
        Determine evidence IDs that are explicitly referenced
        in the composed response.

        Only IDs already present in state.evidence can be returned.

        This method does not create or infer new evidence IDs.
        """

        available_ids = [
            evidence.evidence_id
            for evidence in state.evidence
        ]

        referenced_ids: list[str] = []

        for evidence_id in available_ids:
            if evidence_id in response_text:
                referenced_ids.append(evidence_id)

        return referenced_ids