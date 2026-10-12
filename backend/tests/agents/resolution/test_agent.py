from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from autogen_agentchat.messages import TextMessage

from app.agents.resolution_agent.agents import (
    RESOLUTION_AGENT_NAME,
    ResolutionAgent,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentError,
    ResolutionAgentState,
    ResolutionAgentStatus,
    ResolutionType,
)


class FakeModelClient:
    """Minimal model-client stand-in for constructor tests."""

    pass


class FakeAutoGenAgent:
    """Fake AutoGen agent used to test ResolutionAgent.run()."""

    def __init__(
        self,
        *,
        result: Any = None,
        exception: Exception | None = None,
    ) -> None:
        self.result = result
        self.exception = exception
        self.last_task: str | None = None

    async def run(self, *, task: str) -> Any:
        self.last_task = task

        if self.exception is not None:
            raise self.exception

        return self.result


def build_state() -> ResolutionAgentState:
    return ResolutionAgentState(
        request="The customer was charged twice.",
        ticket_id="TICKET-1001",
    )


def build_agent() -> ResolutionAgent:
    return ResolutionAgent(FakeModelClient())


def build_valid_resolution_response() -> str:
    """
    Build valid structured output expected from the Resolution Agent.

    The test state contains no evidence, so evidence_ids must be empty.
    """

    return """
    {
        "resolution_type": "billing",
        "recommendation": "Review the billing records and verify the duplicate charge.",
        "rationale": "The customer reports being charged twice. The available state does not contain supporting evidence, so the recommendation is limited to verification.",
        "evidence_ids": [],
        "confidence": 0.80,
        "requires_human_review": true,
        "proposed_actions": [
            "Verify the billing records",
            "Confirm whether the charge is duplicated"
        ]
    }
    """


def build_valid_resolution_message() -> TextMessage:
    return TextMessage(
        content=build_valid_resolution_response(),
        source="assistant",
    )


def test_default_agent_name():
    agent = build_agent()

    assert agent.name == RESOLUTION_AGENT_NAME


def test_custom_agent_name():
    agent = ResolutionAgent(
        FakeModelClient(),
        name="custom_resolution_agent",
    )

    assert agent.name == "custom_resolution_agent"


def test_constructor_rejects_none_model_client():
    with pytest.raises(ValueError, match="model_client"):
        ResolutionAgent(None)


def test_constructor_rejects_empty_name():
    with pytest.raises(ValueError, match="name"):
        ResolutionAgent(
            FakeModelClient(),
            name="   ",
        )


def test_agent_property_returns_underlying_agent():
    agent = build_agent()

    assert agent.agent is not None


def test_build_task_returns_prompt():
    agent = build_agent()
    state = build_state()

    task = agent.build_task(state)

    assert isinstance(task, str)
    assert state.request in task
    assert "resolution" in task.lower()


def test_build_task_rejects_invalid_state():
    agent = build_agent()

    with pytest.raises(TypeError, match="ResolutionAgentState"):
        agent.build_task("invalid")  # type: ignore[arg-type]


def test_extract_final_response_returns_latest_text_message():
    first = TextMessage(
        content="First response.",
        source="assistant",
    )

    second = TextMessage(
        content="Final response.",
        source="assistant",
    )

    result = SimpleNamespace(
        messages=[first, second],
    )

    response = ResolutionAgent._extract_final_response(result)

    assert response == "Final response."


def test_extract_final_response_strips_whitespace():
    message = TextMessage(
        content="   Final response.   ",
        source="assistant",
    )

    result = SimpleNamespace(
        messages=[message],
    )

    response = ResolutionAgent._extract_final_response(result)

    assert response == "Final response."


def test_extract_final_response_returns_none_without_messages():
    result = SimpleNamespace(messages=[])

    response = ResolutionAgent._extract_final_response(result)

    assert response is None


def test_extract_final_response_returns_none_without_text_message():
    result = SimpleNamespace(
        messages=[
            object(),
            SimpleNamespace(content={"not": "text"}),
        ]
    )

    response = ResolutionAgent._extract_final_response(result)

    assert response is None


def test_extract_final_response_handles_missing_messages_attribute():
    result = SimpleNamespace()

    response = ResolutionAgent._extract_final_response(result)

    assert response is None


def test_run_rejects_invalid_state():
    agent = build_agent()

    async def execute() -> None:
        with pytest.raises(TypeError, match="ResolutionAgentState"):
            await agent.run("invalid")  # type: ignore[arg-type]

    asyncio.run(execute())


def test_run_completes_with_structured_response():
    state = build_state()
    agent = build_agent()

    response_message = build_valid_resolution_message()

    fake_result = SimpleNamespace(
        messages=[response_message],
    )

    fake_agent = FakeAutoGenAgent(result=fake_result)
    agent._agent = fake_agent

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    result = asyncio.run(execute())

    assert result.status == ResolutionAgentStatus.COMPLETED
    assert result.iteration_count == 1
    assert result.errors == []

    assert result.recommendation is not None
    assert (
        result.recommendation.resolution_type
        == ResolutionType.BILLING
    )
    assert (
        result.recommendation.recommendation
        == "Review the billing records and verify the duplicate charge."
    )
    assert result.recommendation.evidence_ids == []
    assert result.recommendation.confidence == 0.80
    assert result.recommendation.requires_human_review is True

    assert (
        result.execution_metadata["agent_response"]
        == build_valid_resolution_response().strip()
    )

    assert fake_agent.last_task is not None
    assert state.request in fake_agent.last_task


def test_run_increments_iteration_count():
    state = build_state()
    state.iteration_count = 4

    agent = build_agent()

    fake_result = SimpleNamespace(
        messages=[
            build_valid_resolution_message(),
        ]
    )

    agent._agent = FakeAutoGenAgent(result=fake_result)

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    result = asyncio.run(execute())

    assert result.iteration_count == 5
    assert result.status == ResolutionAgentStatus.COMPLETED
    assert result.recommendation is not None


def test_run_fails_when_response_is_empty():
    state = build_state()
    agent = build_agent()

    fake_result = SimpleNamespace(
        messages=[
            TextMessage(
                content="   ",
                source="assistant",
            )
        ]
    )

    agent._agent = FakeAutoGenAgent(result=fake_result)

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    result = asyncio.run(execute())

    assert result.status == ResolutionAgentStatus.FAILED
    assert result.recommendation is None
    assert result.execution_metadata.get("agent_response") is None
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "empty_agent_response"
    assert result.errors[0].source == RESOLUTION_AGENT_NAME
    assert result.errors[0].recoverable is False


def test_run_fails_when_autogen_raises_exception():
    state = build_state()
    agent = build_agent()

    agent._agent = FakeAutoGenAgent(
        exception=RuntimeError("model execution failed")
    )

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    result = asyncio.run(execute())

    assert result.status == ResolutionAgentStatus.FAILED
    assert result.recommendation is None
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "RuntimeError"
    assert result.errors[0].message == "model execution failed"
    assert result.errors[0].recoverable is False
    assert "agent_response" not in result.execution_metadata


def test_run_builds_task_from_current_state():
    state = build_state()
    agent = build_agent()

    fake_result = SimpleNamespace(
        messages=[
            build_valid_resolution_message(),
        ]
    )

    fake_agent = FakeAutoGenAgent(result=fake_result)
    agent._agent = fake_agent

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    asyncio.run(execute())

    assert fake_agent.last_task is not None
    assert "TICKET-1001" in fake_agent.last_task
    assert "The customer was charged twice." in fake_agent.last_task


def test_run_preserves_existing_errors():
    state = build_state()

    state.errors.append(
        ResolutionAgentError(
            source="retrieval_agent",
            error_type="retrieval_warning",
            message="Evidence set was partially available.",
            recoverable=True,
        )
    )

    agent = build_agent()

    fake_result = SimpleNamespace(
        messages=[
            build_valid_resolution_message(),
        ]
    )

    agent._agent = FakeAutoGenAgent(result=fake_result)

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    result = asyncio.run(execute())

    assert result.status == ResolutionAgentStatus.COMPLETED
    assert result.recommendation is not None

    assert len(result.errors) == 1
    assert result.errors[0].error_type == "retrieval_warning"
    assert result.errors[0].source == "retrieval_agent"


def test_run_does_not_create_unvalidated_recommendation():
    """
    Plain-text LLM output must never become a trusted recommendation.

    The parser rejects non-JSON output, so the agent must fail safely
    and leave state.recommendation as None.

    The raw LLM response is intentionally preserved in execution_metadata
    so validation failures can be diagnosed without rerunning the agent.
    """

    state = build_state()
    agent = build_agent()

    raw_response = "Recommend checking the billing policy first."

    fake_result = SimpleNamespace(
        messages=[
            TextMessage(
                content=raw_response,
                source="assistant",
            )
        ]
    )

    agent._agent = FakeAutoGenAgent(result=fake_result)

    async def execute() -> ResolutionAgentState:
        return await agent.run(state)

    result = asyncio.run(execute())

    assert result.status == ResolutionAgentStatus.FAILED
    assert result.recommendation is None

    # The raw LLM response must be preserved for diagnostics.
    assert result.execution_metadata["agent_response"] == raw_response

    # The failure must identify the output-validation boundary.
    failure = result.execution_metadata["resolution_failure"]

    assert failure["error_type"] == "ValueError"
    assert failure["error_message"] == (
        "Resolution Agent output must be valid JSON."
    )
    assert failure["agent_response"] == raw_response
    assert failure["stage"] == "resolution_output_validation"

    # The structured agent error must also preserve the failure.
    assert len(result.errors) == 1
    assert result.errors[0].source == RESOLUTION_AGENT_NAME
    assert result.errors[0].error_type == "ValueError"
    assert result.errors[0].message == (
        "Resolution Agent output must be valid JSON."
    )
    assert result.errors[0].recoverable is False