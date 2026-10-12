from __future__ import annotations

from app.agents.response_composer_agent.contracts.state import (
    ResponseAudience,
    ResponseTone,
)
from app.agents.response_composer_agent.prompts import (
    RESPONSE_COMPOSER_SYSTEM_PROMPT,
    build_composition_instruction,
    build_system_prompt,
)


def test_system_prompt_is_non_empty():
    prompt = build_system_prompt()

    assert isinstance(prompt, str)
    assert prompt.strip()


def test_system_prompt_matches_constant():
    assert (
        build_system_prompt()
        == RESPONSE_COMPOSER_SYSTEM_PROMPT
    )


def test_system_prompt_defines_composer_role():
    prompt = build_system_prompt()

    assert "Response Composer Agent" in prompt
    assert "SupportMind AI" in prompt


def test_system_prompt_prevents_new_decisions():
    prompt = build_system_prompt()

    assert "classifying the ticket" in prompt
    assert "retrieving evidence" in prompt
    assert "creating a new resolution" in prompt
    assert "making a new escalation decision" in prompt
    assert "executing customer-impacting actions" in prompt


def test_system_prompt_contains_grounding_rules():
    prompt = build_system_prompt()

    assert "Do not invent facts" in prompt
    assert "Do not invent evidence IDs" in prompt
    assert "Do not claim that an action was completed" in prompt
    assert "If the available information is insufficient" in prompt
    assert "guessing" in prompt

def test_system_prompt_contains_resolution_rules():
    prompt = build_system_prompt()

    assert "The Resolution Agent has already produced" in prompt
    assert "Do not create a different resolution" in prompt
    assert "Do not silently modify the recommendation" in prompt


def test_system_prompt_contains_escalation_rules():
    prompt = build_system_prompt()

    assert "The Escalation Agent has already determined" in prompt
    assert "Do not independently decide" in prompt
    assert "Do not claim that escalation has already happened" in prompt


def test_system_prompt_contains_evidence_rules():
    prompt = build_system_prompt()

    assert "Use supplied evidence" in prompt
    assert "Only cite evidence IDs that exist" in prompt
    assert "Do not fabricate citations" in prompt


def test_system_prompt_contains_audience_rules():
    prompt = build_system_prompt()

    assert "CUSTOMER:" in prompt
    assert "SUPPORT_AGENT:" in prompt
    assert "INTERNAL:" in prompt


def test_system_prompt_contains_tone_rules():
    prompt = build_system_prompt()

    assert "PROFESSIONAL:" in prompt
    assert "FRIENDLY:" in prompt
    assert "CONCISE:" in prompt
    assert "TECHNICAL:" in prompt


def test_composition_instruction_contains_customer_configuration():
    instruction = build_composition_instruction(
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.PROFESSIONAL,
    )

    assert "Target audience: customer" in instruction
    assert "Requested tone: professional" in instruction
    assert "supplied context" in instruction


def test_composition_instruction_contains_support_agent_configuration():
    instruction = build_composition_instruction(
        audience=ResponseAudience.SUPPORT_AGENT,
        tone=ResponseTone.TECHNICAL,
    )

    assert "Target audience: support_agent" in instruction
    assert "Requested tone: technical" in instruction


def test_composition_instruction_contains_internal_configuration():
    instruction = build_composition_instruction(
        audience=ResponseAudience.INTERNAL,
        tone=ResponseTone.CONCISE,
    )

    assert "Target audience: internal" in instruction
    assert "Requested tone: concise" in instruction


def test_composition_instruction_does_not_make_business_decisions():
    instruction = build_composition_instruction(
        audience=ResponseAudience.CUSTOMER,
        tone=ResponseTone.FRIENDLY,
    )

    assert "should escalate" not in instruction.lower()
    assert "resolve the ticket" not in instruction.lower()
    assert "classify the ticket" not in instruction.lower()
    assert "retrieve evidence" not in instruction.lower()