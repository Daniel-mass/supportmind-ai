from __future__ import annotations

import asyncio

from app.agents.response_composer_agent.agent import (
    ResponseComposerAgent,
)
from app.agents.response_composer_agent.contracts.state import (
    ResponseAudience,
    ResponseComposerAgentState,
    ResponseComposerAgentStatus,
    ResponseEvidence,
    ResponseResolutionSignal,
    ResponseTone,
)


class FakeModelClient:
    pass


def build_state() -> ResponseComposerAgentState:
    return ResponseComposerAgentState(
        request="How can I reset my password?",
        ticket_id="SUP-001",
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.PROFESSIONAL,
        evidence=[
            ResponseEvidence(
                evidence_id="E1",
                source="knowledge_base",
                content="Password reset is available from the account login page.",
            )
        ],
        resolution=ResponseResolutionSignal(
            resolution_type="account_action",
            recommendation=(
                "Use the password reset option on the login page."
            ),
            rationale=(
                "The supplied knowledge-base evidence supports "
                "the password reset procedure."
            ),
            confidence=0.95,
        ),
    )


def test_agent_has_expected_name():
    agent = object.__new__(ResponseComposerAgent)

    assert agent.name == "response_composer_agent"


def test_build_task_contains_request_and_context():
    agent = object.__new__(ResponseComposerAgent)

    state = build_state()

    task = agent.build_task(state)

    assert "How can I reset my password?" in task
    assert "E1" in task
    assert "account_action" in task
    assert "customer" in task
    assert "professional" in task


def test_extract_response_text_from_text_message():
    agent = object.__new__(ResponseComposerAgent)

    from autogen_agentchat.messages import TextMessage

    class FakeResult:
        messages = [
            TextMessage(
                content="You can reset your password from the login page.",
                source="response_composer_agent",
            )
        ]

    response = agent._extract_response_text(
        FakeResult()
    )

    assert response == (
        "You can reset your password from the login page."
    )


def test_extract_response_text_returns_empty_without_messages():
    agent = object.__new__(ResponseComposerAgent)

    class FakeResult:
        messages = []

    assert agent._extract_response_text(
        FakeResult()
    ) == ""


def test_extract_evidence_ids_only_returns_existing_ids():
    agent = object.__new__(ResponseComposerAgent)

    state = build_state()

    response = (
        "According to E1, you can reset your password "
        "from the login page."
    )

    evidence_ids = agent._extract_evidence_ids(
        state=state,
        response_text=response,
    )

    assert evidence_ids == ["E1"]


def test_extract_evidence_ids_does_not_create_unknown_ids():
    agent = object.__new__(ResponseComposerAgent)

    state = build_state()

    response = (
        "According to E1 and E999, you can reset your password."
    )

    evidence_ids = agent._extract_evidence_ids(
        state=state,
        response_text=response,
    )

    assert evidence_ids == ["E1"]


def test_state_defaults_are_preserved():
    state = ResponseComposerAgentState(
        request="Test request"
    )

    assert state.status is ResponseComposerAgentStatus.PENDING
    assert state.iteration_count == 0
    assert state.composed_response is None


def test_success_properties_before_execution():
    state = build_state()

    assert state.successful is False
    assert state.has_evidence is True
    assert state.has_resolution is True
    assert state.has_escalation is False
    assert state.has_response is False