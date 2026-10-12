#tools.py
from __future__ import annotations

from typing import Any

from app.agents.response_composer_agent.contracts.state import (
    ResponseComposerAgentState,
)


def validate_evidence_references(
    evidence_ids: list[str],
    state: ResponseComposerAgentState,
) -> tuple[bool, list[str]]:
    """
    Validate that every referenced evidence ID exists
    in the Response Composer state.

    This function does not retrieve or create evidence.
    It only validates references against evidence already
    supplied by the upstream retrieval pipeline.
    """
    available_ids = {
        evidence.evidence_id
        for evidence in state.evidence
    }

    missing_ids = [
        evidence_id
        for evidence_id in evidence_ids
        if evidence_id not in available_ids
    ]

    return len(missing_ids) == 0, missing_ids


def summarize_upstream_signals(
    state: ResponseComposerAgentState,
) -> dict[str, Any]:
    """
    Build a deterministic summary of upstream outputs.

    This function does not make any new:
    - classification decisions
    - resolution decisions
    - escalation decisions
    """

    classification = [
        {
            "name": signal.name,
            "value": signal.value,
            "confidence": signal.confidence,
            "source": signal.source,
        }
        for signal in state.classification_signals
    ]

    resolution = None

    if state.resolution is not None:
        resolution = {
            "resolution_type": state.resolution.resolution_type,
            "recommendation": state.resolution.recommendation,
            "rationale": state.resolution.rationale,
            "confidence": state.resolution.confidence,
            "requires_human_review": (
                state.resolution.requires_human_review
            ),
            "proposed_actions": list(
                state.resolution.proposed_actions
            ),
        }

    escalation = None

    if state.escalation is not None:
        escalation = {
            "escalation_type": state.escalation.escalation_type,
            "should_escalate": state.escalation.should_escalate,
            "reason": state.escalation.reason,
            "rationale": state.escalation.rationale,
            "recommended_team": state.escalation.recommended_team,
            "recommended_path": state.escalation.recommended_path,
            "evidence_ids": list(
                state.escalation.evidence_ids
            ),
            "confidence": state.escalation.confidence,
            "requires_human_review": (
                state.escalation.requires_human_review
            ),
            "proposed_actions": list(
                state.escalation.proposed_actions
            ),
        }

    evidence = [
        {
            "evidence_id": item.evidence_id,
            "source": item.source,
            "content": item.content,
            "source_reference": item.source_reference,
            "score": item.score,
        }
        for item in state.evidence
    ]

    return {
        "request": state.request,
        "ticket_id": state.ticket_id,
        "audience": state.audience.value,
        "tone": state.tone.value,
        "classification": classification,
        "evidence": evidence,
        "resolution": resolution,
        "escalation": escalation,
    }


def build_composition_context(
    state: ResponseComposerAgentState,
) -> dict[str, Any]:
    """
    Build deterministic context for the Response Composer LLM.

    Only information already present in the state is included.
    """

    context = summarize_upstream_signals(state)

    context["constraints"] = {
        "do_not_invent_facts": True,
        "do_not_invent_evidence": True,
        "do_not_change_resolution": True,
        "do_not_change_escalation_decision": True,
        "do_not_execute_actions": True,
    }

    return context


def validate_composed_response(
    response: str,
    evidence_ids: list[str],
    state: ResponseComposerAgentState,
) -> tuple[bool, list[str]]:
    """
    Perform structural validation of a composed response.

    Checks:
    1. Response is not empty.
    2. Every cited evidence ID exists in the supplied evidence.

    This function does not determine whether the response is
    factually correct. That belongs to the Evaluation / Trust Layer.
    """

    errors: list[str] = []

    if not response or not response.strip():
        errors.append("Response must not be empty.")

    evidence_valid, missing_ids = validate_evidence_references(
        evidence_ids=evidence_ids,
        state=state,
    )

    if not evidence_valid:
        errors.append(
            "Unknown evidence IDs: "
            + ", ".join(missing_ids)
        )

    return len(errors) == 0, errors