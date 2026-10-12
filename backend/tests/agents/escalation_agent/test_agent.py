#test_agent.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from autogen_agentchat.messages import TextMessage

from app.agents.escalation_agent.agent import (
    ESCALATION_AGENT_NAME,
    EscalationAgent,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationAgentStatus,
)


class FakeModelClient:
    """Minimal model client used to construct the AutoGen agent."""

    async def close(self) -> None:
        return None


def build_agent() -> EscalationAgent:
    return EscalationAgent(
        model_client=FakeModelClient(),
    )


def build_state() -> EscalationAgentState:
    return EscalationAgentState(
        request="Customer cannot access the application.",
        ticket_id="TICKET-001",
    )


def test_agent_name_default() -> None:
    agent = build_agent()

    assert agent.name == ESCALATION_AGENT_NAME
    assert agent.name == "escalation_agent"


def test_agent_requires_model_client() -> None:
    with pytest.raises(
        ValueError,
        match="model_client must not be None",
    ):
        EscalationAgent(model_client=None)


def test_agent_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        EscalationAgent(
            model_client=FakeModelClient(),
            name="   ",
        )


def test_agent_property_returns_autogen_agent() -> None:
    agent = build_agent()

    assert agent.agent is not None
    assert agent.agent.name == "escalation_agent"


def test_build_task_requires_valid_state() -> None:
    agent = build_agent()

    with pytest.raises(
        TypeError,
        match="EscalationAgentState",
    ):
        agent.build_task("invalid")  # type: ignore[arg-type]


def test_build_task_returns_prompt() -> None:
    agent = build_agent()
    state = build_state()

    task = agent.build_task(state)

    assert isinstance(task, str)
    assert task.strip()

    assert (
        "Analyze the following customer support case"
        in task
    )
    assert "TICKET-001" in task
    assert (
        "Customer cannot access the application."
        in task
    )


def test_run_rejects_invalid_state() -> None:
    agent = build_agent()

    with pytest.raises(
        TypeError,
        match="EscalationAgentState",
    ):
        import asyncio

        asyncio.run(
            agent.run("invalid")  # type: ignore[arg-type]
        )


def test_extract_final_response_returns_latest_text_message() -> None:
    first = TextMessage(
        content="First response.",
        source="assistant",
    )

    second = TextMessage(
        content="Final escalation analysis.",
        source="assistant",
    )

    result = SimpleNamespace(
        messages=[
            first,
            second,
        ]
    )

    response = EscalationAgent._extract_final_response(
        result
    )

    assert response == "Final escalation analysis."


def test_extract_final_response_skips_empty_messages() -> None:
    empty_message = TextMessage(
        content="   ",
        source="assistant",
    )

    final_message = TextMessage(
        content="Escalation analysis.",
        source="assistant",
    )

    result = SimpleNamespace(
        messages=[
            empty_message,
            final_message,
        ]
    )

    response = EscalationAgent._extract_final_response(
        result
    )

    assert response == "Escalation analysis."


def test_extract_final_response_returns_none_without_messages() -> None:
    result = SimpleNamespace(messages=[])

    response = EscalationAgent._extract_final_response(
        result
    )

    assert response is None


def test_extract_final_response_returns_none_when_no_text_message() -> None:
    result = SimpleNamespace(
        messages=[
            object(),
            SimpleNamespace(content={"not": "text"}),
        ]
    )

    response = EscalationAgent._extract_final_response(
        result
    )

    assert response is None


def test_run_completes_and_stores_agent_response() -> None:
    agent = build_agent()
    state = build_state()

    expected_response = (
        "Escalation is recommended because the supplied "
        "operational signals indicate elevated SLA risk."
    )

    agent.agent.run = AsyncMock(
        return_value=SimpleNamespace(
            messages=[
                TextMessage(
                    content=expected_response,
                    source="assistant",
                )
            ]
        )
    )

    import asyncio

    result = asyncio.run(agent.run(state))

    assert result is state
    assert result.status is EscalationAgentStatus.COMPLETED
    assert result.iteration_count == 1
    assert result.errors == []

    assert (
        result.execution_metadata["agent_response"]
        == expected_response
    )


def test_run_fails_when_agent_returns_no_text() -> None:
    agent = build_agent()
    state = build_state()

    agent.agent.run = AsyncMock(
        return_value=SimpleNamespace(
            messages=[]
        )
    )

    import asyncio

    result = asyncio.run(agent.run(state))

    assert result is state
    assert result.status is EscalationAgentStatus.FAILED
    assert result.iteration_count == 1

    assert len(result.errors) == 1

    error = result.errors[0]

    assert error.source == "escalation_agent"
    assert error.error_type == "empty_agent_response"
    assert error.recoverable is False

    assert "agent_response" not in result.execution_metadata


def test_run_captures_agent_exception() -> None:
    agent = build_agent()
    state = build_state()

    agent.agent.run = AsyncMock(
        side_effect=RuntimeError("model execution failed")
    )

    import asyncio

    result = asyncio.run(agent.run(state))

    assert result is state
    assert result.status is EscalationAgentStatus.FAILED
    assert result.iteration_count == 1

    assert len(result.errors) == 1

    error = result.errors[0]

    assert error.source == "escalation_agent"
    assert error.error_type == "RuntimeError"
    assert error.message == "model execution failed"
    assert error.recoverable is False

    assert "agent_response" not in result.execution_metadata


def test_run_increments_iteration_count_on_each_run() -> None:
    agent = build_agent()
    state = build_state()

    agent.agent.run = AsyncMock(
        return_value=SimpleNamespace(
            messages=[
                TextMessage(
                    content="No escalation recommended.",
                    source="assistant",
                )
            ]
        )
    )

    import asyncio

    asyncio.run(agent.run(state))

    assert state.iteration_count == 1

    asyncio.run(agent.run(state))

    assert state.iteration_count == 2


def test_run_sets_running_before_execution() -> None:
    agent = build_agent()
    state = build_state()

    observed_status: list[EscalationAgentStatus] = []

    async def fake_run(*, task: str):
        observed_status.append(state.status)

        return SimpleNamespace(
            messages=[
                TextMessage(
                    content="Escalation analysis completed.",
                    source="assistant",
                )
            ]
        )

    agent.agent.run = fake_run

    import asyncio

    result = asyncio.run(agent.run(state))

    assert observed_status == [
        EscalationAgentStatus.RUNNING
    ]

    assert result.status is EscalationAgentStatus.COMPLETED


def test_close_handles_model_client_close() -> None:
    agent = build_agent()

    close_mock = AsyncMock()
    agent.agent._model_client.close = close_mock

    import asyncio

    asyncio.run(agent.close())

    close_mock.assert_awaited_once()
