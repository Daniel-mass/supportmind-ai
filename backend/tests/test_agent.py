import pytest
from pydantic import ValidationError

from backend.app.intelligence.schemas.agent import Agent


def make_valid_agent() -> Agent:
    return Agent(
        agent_id="agent-123",
        name="Support Agent 123",
        team="Service Desk",
        active=True,
        source="helpdesk",
        metadata={
            "department": "IT Support",
        },
    )


def test_valid_agent_creation():
    agent = make_valid_agent()

    assert agent.agent_id == "agent-123"
    assert agent.name == "Support Agent 123"
    assert agent.team == "Service Desk"
    assert agent.active is True
    assert agent.source == "helpdesk"


def test_optional_fields_can_be_missing():
    agent = Agent(
        agent_id="agent-456",
    )

    assert agent.name is None
    assert agent.team is None
    assert agent.active is True
    assert agent.source == "helpdesk"
    assert agent.metadata == {}


def test_agent_can_be_inactive():
    agent = Agent(
        agent_id="agent-789",
        active=False,
    )

    assert agent.active is False


def test_empty_agent_id_is_rejected():
    with pytest.raises(ValidationError):
        Agent(
            agent_id="",
        )


def test_empty_name_is_rejected():
    with pytest.raises(ValidationError):
        Agent(
            agent_id="agent-123",
            name="",
        )


def test_empty_team_is_rejected():
    with pytest.raises(ValidationError):
        Agent(
            agent_id="agent-123",
            team="",
        )


def test_invalid_assignment_is_rejected():
    agent = make_valid_agent()

    with pytest.raises(ValidationError):
        agent.agent_id = ""


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        Agent(
            agent_id="agent-123",
            unexpected_field="should_fail",
        )