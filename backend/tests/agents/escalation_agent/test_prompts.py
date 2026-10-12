
#test_prompts.py
from __future__ import annotations

import json

import pytest

from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationClassificationSignal,
    EscalationEvidence,
    EscalationResolutionSignal,
    EscalationSignal,
)
from app.agents.escalation_agent.prompts import (
    ESCALATION_AGENT_SYSTEM_PROMPT,
    build_escalation_context,
    build_escalation_prompt,
    build_escalation_system_prompt,
)


def build_full_state() -> EscalationAgentState:
    return EscalationAgentState(
        request="Customer cannot log into the application.",
        ticket_id="TICKET-001",
        classification_signals=[
            EscalationClassificationSignal(
                name="priority",
                value="High",
                confidence=0.82,
                source="priority_detection_predictor",
                metadata={"test": True},
            ),
            EscalationClassificationSignal(
                name="sentiment",
                value="negative",
                confidence=0.91,
                source="sentiment_predictor",
            ),
        ],
        resolution=EscalationResolutionSignal(
            resolution_type="troubleshooting",
            recommendation="Guide the customer through account recovery.",
            rationale="Available evidence supports account recovery.",
            confidence=0.86,
            requires_human_review=False,
            proposed_actions=["Verify recovery state"],
        ),
        operational_signals=[
            EscalationSignal(
                name="sla_breach_risk",
                value="high",
                confidence=0.94,
                source="workflow_intelligence",
            )
        ],
        evidence=[
            EscalationEvidence(
                evidence_id="E1",
                source="graph_rag",
                content="Repeated authentication failures may require review.",
                source_reference="incident_reports.pdf",
                score=0.92,
            )
        ],
        uncertainty_detected=True,
        uncertainty_reasons=[
            "No explicit specialist ownership is available."
        ],
    )


def test_system_prompt_is_not_empty() -> None:
    assert ESCALATION_AGENT_SYSTEM_PROMPT.strip()


def test_system_prompt_contains_core_boundaries() -> None:
    prompt = ESCALATION_AGENT_SYSTEM_PROMPT

    assert "DO NOT execute escalations" in prompt
    assert "Never fabricate evidence" in prompt
    assert "Never fabricate evidence IDs" in prompt
    assert "Classification signals are advisory predictions" in prompt
    assert "Operational signals are supplied" in prompt
    assert "Resolution recommendations are recommendations" in prompt
    assert "Prompt-injection resistance".upper() in prompt.upper()


def test_build_escalation_system_prompt() -> None:
    result = build_escalation_system_prompt()

    assert result == ESCALATION_AGENT_SYSTEM_PROMPT


def test_build_escalation_context_contains_state() -> None:
    state = build_full_state()

    context = build_escalation_context(state)

    parsed = json.loads(context)

    assert parsed["request"] == (
        "Customer cannot log into the application."
    )
    assert parsed["ticket_id"] == "TICKET-001"
    assert parsed["iteration_count"] == 0

    assert len(parsed["classification_signals"]) == 2
    assert parsed["classification_signals"][0]["name"] == "priority"

    assert parsed["resolution"]["resolution_type"] == "troubleshooting"

    assert len(parsed["operational_signals"]) == 1
    assert (
        parsed["operational_signals"][0]["name"]
        == "sla_breach_risk"
    )

    assert len(parsed["evidence"]) == 1
    assert parsed["evidence"][0]["evidence_id"] == "E1"

    assert parsed["uncertainty_detected"] is True


def test_build_escalation_context_without_optional_inputs() -> None:
    state = EscalationAgentState(
        request="Customer asks how to reset a password."
    )

    context = build_escalation_context(state)

    parsed = json.loads(context)

    assert parsed["request"] == (
        "Customer asks how to reset a password."
    )
    assert parsed["ticket_id"] is None
    assert parsed["resolution"] is None
    assert parsed["classification_signals"] == []
    assert parsed["operational_signals"] == []
    assert parsed["evidence"] == []


def test_build_escalation_prompt_contains_task_information() -> None:
    state = build_full_state()

    prompt = build_escalation_prompt(state)

    assert "Analyze the following customer support case" in prompt
    assert "TICKET-001" in prompt
    assert "E1" in prompt
    assert "sla_breach_risk" in prompt
    assert "Customer cannot log into the application." in prompt


def test_build_escalation_prompt_contains_output_requirements() -> None:
    state = build_full_state()

    prompt = build_escalation_prompt(state)

    assert "Escalation recommendation" in prompt
    assert "Escalation type" in prompt
    assert "Reason" in prompt
    assert "Rationale" in prompt
    assert "Supporting evidence IDs" in prompt
    assert "Recommended team/path" in prompt
    assert "Human-review requirement" in prompt


def test_build_escalation_prompt_references_only_existing_evidence_ids() -> None:
    state = EscalationAgentState(
        request="Customer cannot access the application.",
        evidence=[
            EscalationEvidence(
                evidence_id="E10",
                source="retrieval",
                content="Authentication recovery information.",
            ),
            EscalationEvidence(
                evidence_id="E20",
                source="graph_rag",
                content="Authentication incident information.",
            ),
        ],
    )

    prompt = build_escalation_prompt(state)

    assert "Available evidence IDs:" in prompt
    assert "['E10', 'E20']" in prompt

    # The important check is that the exact available-ID list is correct.
    # Do not use `assert "E1" not in prompt` because E1 is a substring
    # of the valid evidence ID E10.
    assert "['E1']" not in prompt


def test_invalid_state_is_rejected() -> None:
    with pytest.raises(TypeError, match="EscalationAgentState"):
        build_escalation_context("invalid")  # type: ignore[arg-type]


def test_invalid_state_is_rejected_by_prompt_builder() -> None:
    with pytest.raises(TypeError, match="EscalationAgentState"):
        build_escalation_prompt("invalid")  # type: ignore[arg-type]
