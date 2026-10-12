"""
Unit tests for the Coordinator Agent.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.coordinator.agent import (
    COORDINATOR_AGENT_NAME,
    CoordinatorAgent,
)
from app.agents.coordinator.state import (
    CoordinatorState,
    WorkflowError,
    WorkflowStatus,
)


def _text_message(content: str):
    """
    Create a real AutoGen TextMessage for testing response extraction.
    """

    from autogen_agentchat.messages import TextMessage

    return TextMessage(
        content=content,
        source="coordinator_agent",
    )


def test_coordinator_agent_requires_model_client():
    """
    CoordinatorAgent must reject a missing model client.
    """

    with pytest.raises(
        ValueError,
        match="model_client must not be None",
    ):
        CoordinatorAgent(None)


def test_coordinator_agent_initializes_with_model_client():
    """
    CoordinatorAgent should initialize an AutoGen AssistantAgent
    using the supplied model client.
    """

    model_client = MagicMock()

    agent = CoordinatorAgent(model_client)

    assert agent.name == COORDINATOR_AGENT_NAME
    assert agent.agent.name == COORDINATOR_AGENT_NAME


def test_coordinator_agent_accepts_custom_name():
    """
    CoordinatorAgent should support a custom AutoGen agent name.
    """

    model_client = MagicMock()

    agent = CoordinatorAgent(
        model_client,
        name="custom_coordinator",
    )

    assert agent.name == "custom_coordinator"
    assert agent.agent.name == "custom_coordinator"


def test_build_task_requires_coordinator_state():
    """
    build_task() must reject invalid workflow state objects.
    """

    model_client = MagicMock()
    agent = CoordinatorAgent(model_client)

    with pytest.raises(
        TypeError,
        match="state must be an instance of CoordinatorState",
    ):
        agent.build_task("invalid-state")


def test_build_task_contains_current_workflow_state():
    """
    build_task() should construct a prompt containing the current
    Coordinator workflow state.
    """

    model_client = MagicMock()
    agent = CoordinatorAgent(model_client)

    state = CoordinatorState(
        request="Why is my payment delayed?",
    )

    task = agent.build_task(state)

    assert "Why is my payment delayed?" in task
    assert "CURRENT WORKFLOW STATE" in task
    assert "ML / MODEL PREDICTION SIGNALS" in task


def test_extract_final_response_returns_latest_text_message():
    """
    The latest textual AutoGen message should be returned.
    """

    messages = [
        _text_message("Earlier response"),
        _text_message("Final response"),
    ]

    result = SimpleNamespace(messages=messages)

    response = CoordinatorAgent._extract_final_response(result)

    assert response == "Final response"


def test_extract_final_response_strips_whitespace():
    """
    Extracted textual responses should be stripped of surrounding
    whitespace.
    """

    messages = [
        _text_message("   Final response   "),
    ]

    result = SimpleNamespace(messages=messages)

    response = CoordinatorAgent._extract_final_response(result)

    assert response == "Final response"


def test_extract_final_response_returns_none_when_no_messages():
    """
    No messages should result in None.
    """

    result = SimpleNamespace(messages=[])

    response = CoordinatorAgent._extract_final_response(result)

    assert response is None


def test_extract_final_response_returns_none_when_no_text_message():
    """
    Non-text messages should not be treated as final textual responses.
    """

    result = SimpleNamespace(
        messages=[
            object(),
            object(),
        ]
    )

    response = CoordinatorAgent._extract_final_response(result)

    assert response is None


def test_run_updates_state_on_success():
    """
    A successful AutoGen execution should:

    - increment iteration_count
    - store the final response
    - mark the workflow completed
    """

    async def _test():
        model_client = MagicMock()
        agent = CoordinatorAgent(model_client)

        result = SimpleNamespace(
            messages=[
                _text_message(
                    "The ticket requires retrieval."
                ),
            ]
        )

        agent._agent.run = AsyncMock(
            return_value=result
        )

        state = CoordinatorState(
            request="Why is the ticket delayed?",
        )

        updated_state = await agent.run(state)

        assert updated_state is state

        assert (
            updated_state.status
            == WorkflowStatus.COMPLETED
        )

        assert (
            updated_state.final_response
            == "The ticket requires retrieval."
        )

        assert updated_state.iteration_count == 1

        agent._agent.run.assert_awaited_once()

    asyncio.run(_test())


def test_run_marks_state_failed_on_empty_response():
    """
    An empty AutoGen response should fail the workflow and record
    a structured workflow error.
    """

    async def _test():
        model_client = MagicMock()
        agent = CoordinatorAgent(model_client)

        result = SimpleNamespace(
            messages=[]
        )

        agent._agent.run = AsyncMock(
            return_value=result
        )

        state = CoordinatorState(
            request="Investigate the issue.",
        )

        updated_state = await agent.run(state)

        assert (
            updated_state.status
            == WorkflowStatus.FAILED
        )

        assert updated_state.final_response is None

        assert updated_state.iteration_count == 1

        assert len(updated_state.errors) == 1

        error = updated_state.errors[0]

        assert error.source == COORDINATOR_AGENT_NAME
        assert error.error_type == "empty_agent_response"
        assert error.recoverable is False

    asyncio.run(_test())


def test_run_records_agent_exception():
    """
    Exceptions raised by AutoGen should be converted into a structured
    workflow error and should not silently disappear.
    """

    async def _test():
        model_client = MagicMock()
        agent = CoordinatorAgent(model_client)

        agent._agent.run = AsyncMock(
            side_effect=RuntimeError(
                "LLM unavailable"
            )
        )

        state = CoordinatorState(
            request="Investigate the issue.",
        )

        updated_state = await agent.run(state)

        assert (
            updated_state.status
            == WorkflowStatus.FAILED
        )

        assert updated_state.final_response is None

        assert updated_state.iteration_count == 1

        assert len(updated_state.errors) == 1

        error = updated_state.errors[0]

        assert isinstance(error, WorkflowError)

        assert error.source == COORDINATOR_AGENT_NAME
        assert error.error_type == "RuntimeError"
        assert error.message == "LLM unavailable"
        assert error.recoverable is False

    asyncio.run(_test())


def test_run_uses_state_aware_prompt_builder():
    """
    CoordinatorAgent.run() must construct its task from the current
    CoordinatorState rather than using a static task.
    """

    async def _test():
        model_client = MagicMock()
        agent = CoordinatorAgent(model_client)

        result = SimpleNamespace(
            messages=[
                _text_message(
                    "Coordinator response."
                ),
            ]
        )

        agent._agent.run = AsyncMock(
            return_value=result
        )

        state = CoordinatorState(
            request="Analyze this ticket.",
        )

        with patch(
            "app.agents.coordinator.agent.build_coordinator_prompt",
            return_value="STATE-AWARE PROMPT",
        ) as prompt_builder:

            await agent.run(state)

        prompt_builder.assert_called_once_with(state)

        agent._agent.run.assert_awaited_once_with(
            task="STATE-AWARE PROMPT"
        )

    asyncio.run(_test())