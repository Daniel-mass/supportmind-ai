"""
Real Escalation Agent diagnostic integration test.

Purpose
-------
Run ONLY the real Escalation Agent against Ollama / qwen3:8b.

This test intentionally does NOT require a structured
EscalationRecommendation yet.

The purpose is to expose the exact raw LLM response and
Escalation Agent state so we can determine the correct
output-contract/parser change before implementing it.

Prerequisites
-------------
Ollama:
    http://127.0.0.1:11434

Model:
    qwen3:8b

Run
---
$env:RUN_REAL_ESCALATION_DIAGNOSTIC="1"
$env:PYTHONPATH="$PWD\backend"

pytest `
    backend\tests\agents\integration\test_real_escalation_diagnostic.py `
    -v -s
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from autogen_ext.models.ollama import (
    OllamaChatCompletionClient,
)

from app.agents.escalation_agent.agent import (
    EscalationAgent,
)

from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
)


# ============================================================================
# Environment
# ============================================================================

RUN_REAL_ESCALATION_DIAGNOSTIC = (
    os.getenv(
        "RUN_REAL_ESCALATION_DIAGNOSTIC",
        "0",
    ).strip()
    == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_REAL_ESCALATION_DIAGNOSTIC,
    reason=(
        "Set RUN_REAL_ESCALATION_DIAGNOSTIC=1 "
        "to run the real Escalation Agent diagnostic test."
    ),
)


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_LLM_MODEL",
    "qwen3:8b",
)


# ============================================================================
# Model client
# ============================================================================


def build_model_client() -> OllamaChatCompletionClient:
    """
    Build the same real AutoGen Ollama client used by
    the multi-agent integration workflow.
    """

    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )


async def close_model_client(
    model_client: OllamaChatCompletionClient,
) -> None:
    """
    Close the asynchronous AutoGen model client.
    """

    close_method = getattr(
        model_client,
        "close",
        None,
    )

    if close_method is None:
        return

    result = close_method()

    if asyncio.iscoroutine(result):
        await result


# ============================================================================
# Diagnostic state
# ============================================================================


def build_diagnostic_state() -> EscalationAgentState:
    """
    Build a realistic Escalation Agent input state.

    This is intentionally limited to information that the
    Escalation Agent can already consume.

    We are NOT fabricating an EscalationRecommendation.

    The recommendation must come from the real LLM.
    """

    return EscalationAgentState(
        request=(
            "A customer is asking about a refund for a subscription "
            "and is frustrated because the issue has not been resolved. "
            "The available support evidence indicates that the customer "
            "is asking for a refund and that the issue requires review "
            "of the applicable subscription and refund policies."
        ),
    )


# ============================================================================
# Diagnostic formatting
# ============================================================================


def print_diagnostic_state(
    state: EscalationAgentState,
) -> None:
    """
    Print the complete Escalation Agent state after execution.
    """

    print()
    print("=" * 80)
    print("ESCALATION AGENT DIAGNOSTIC STATE")
    print("=" * 80)

    print()
    print("STATUS:")
    print(
        state.status.value
        if hasattr(state.status, "value")
        else state.status
    )

    print()
    print("ITERATION COUNT:")
    print(state.iteration_count)

    print()
    print("RECOMMENDATION:")
    print(state.recommendation)

    print()
    print("ERRORS:")

    if state.errors:
        for error in state.errors:
            if hasattr(error, "model_dump"):
                print(
                    json.dumps(
                        error.model_dump(),
                        indent=2,
                        default=str,
                    )
                )
            else:
                print(error)
    else:
        print("None")

    print()
    print("UNCERTAINTY DETECTED:")
    print(state.uncertainty_detected)

    print()
    print("UNCERTAINTY REASONS:")
    print(state.uncertainty_reasons)

    print()
    print("EXECUTION METADATA:")

    if state.execution_metadata:
        print(
            json.dumps(
                state.execution_metadata,
                indent=2,
                default=str,
            )
        )
    else:
        print("{}")

    print()
    print("=" * 80)
    print("END ESCALATION DIAGNOSTIC")
    print("=" * 80)


# ============================================================================
# Real Escalation Agent diagnostic
# ============================================================================


def test_real_escalation_agent_diagnostic() -> None:
    """
    Execute the real Escalation Agent and expose its raw response.

    IMPORTANT:
        This test does NOT assert that recommendation exists.

    We are intentionally diagnosing the current contract before
    implementing an output parser.
    """

    model_client = None

    try:
        # ------------------------------------------------------------------
        # 1. Build real model client
        # ------------------------------------------------------------------

        model_client = build_model_client()

        print()
        print("=" * 80)
        print("REAL ESCALATION AGENT DIAGNOSTIC")
        print("=" * 80)

        print()
        print(f"Ollama host: {OLLAMA_HOST}")
        print(f"Ollama model: {OLLAMA_MODEL}")

        # ------------------------------------------------------------------
        # 2. Build real Escalation Agent
        # ------------------------------------------------------------------

        escalation_agent = EscalationAgent(
            model_client=model_client,
        )

        print()
        print("EscalationAgent: ready")

        # ------------------------------------------------------------------
        # 3. Build diagnostic state
        # ------------------------------------------------------------------

        state = build_diagnostic_state()

        print()
        print("Initial state:")
        print(
            json.dumps(
                state.model_dump(),
                indent=2,
                default=str,
            )
        )

        # ------------------------------------------------------------------
        # 4. Execute REAL Escalation Agent
        # ------------------------------------------------------------------

        print()
        print("=" * 80)
        print("STARTING REAL ESCALATION AGENT")
        print("=" * 80)

        result = asyncio.run(
            escalation_agent.run(state)
        )

        # ------------------------------------------------------------------
        # 5. Print complete diagnostic information
        # ------------------------------------------------------------------

        print_diagnostic_state(result)

        # ------------------------------------------------------------------
        # 6. Important diagnostic assertion
        # ------------------------------------------------------------------

        assert result.status.value in {
            "completed",
            "failed",
        }

        print()
        print("RAW AGENT RESPONSE:")
        print("-" * 80)

        raw_response = result.execution_metadata.get(
            "agent_response"
        )

        if raw_response:
            print(raw_response)
        else:
            print(
                "[NO agent_response FOUND IN execution_metadata]"
            )

        print("-" * 80)

        print()
        print(
            "Diagnostic execution completed."
        )

        print(
            "No structured EscalationRecommendation assertion "
            "was intentionally applied."
        )

    finally:
        # ------------------------------------------------------------------
        # 7. Cleanup
        # ------------------------------------------------------------------

        if model_client is not None:
            asyncio.run(
                close_model_client(
                    model_client
                )
            )