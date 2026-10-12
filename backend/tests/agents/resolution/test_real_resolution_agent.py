
from __future__ import annotations

import os

import pytest
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.resolution_agent.agents import ResolutionAgent
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionAgentStatus,
    ResolutionEvidence,
)


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_LLM_MODEL",
    "qwen3:8b",
)


def build_model_client() -> OllamaChatCompletionClient:
    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )


def build_resolution_state() -> ResolutionAgentState:
    return ResolutionAgentState(
        request=(
            "A customer is asking about a refund for a subscription "
            "and is frustrated because the issue has not been resolved."
        ),
        ticket_id="TEST-RESOLUTION-001",
        evidence=[
            ResolutionEvidence(
                evidence_id="E1",
                source="refund_policy.pdf",
                content=(
                    "Customers may request a refund within 14 days of "
                    "the original subscription purchase."
                ),
                source_reference="refund_policy.pdf",
                score=0.95,
            ),
            ResolutionEvidence(
                evidence_id="E2",
                source="subscription_policy.pdf",
                content=(
                    "Refund eligibility depends on the subscription "
                    "plan and the applicable refund period."
                ),
                source_reference="subscription_policy.pdf",
                score=0.91,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_real_resolution_agent_with_qwen3() -> None:
    model_client = build_model_client()
    agent = ResolutionAgent(model_client=model_client)

    state = build_resolution_state()

    result = await agent.run(state)

    print("\n" + "=" * 80)
    print("REAL RESOLUTION AGENT TEST")
    print("=" * 80)

    print(f"Status: {result.status}")
    print(f"Iterations: {result.iteration_count}")

    print("\nErrors:")
    if result.errors:
        for error in result.errors:
            print(f"  [{error.error_type}] {error.message}")
    else:
        print("  None")

    print("\nRecommendation:")
    if result.recommendation is None:
        print("  None")
    else:
        print(
            result.recommendation.model_dump_json(
                indent=2
            )
        )

    print("\nRaw Agent Response:")
    print(
        result.execution_metadata.get(
            "agent_response",
            "Not available",
        )
    )

    print("\nExecution Metadata Keys:")
    print(
        list(
            result.execution_metadata.keys()
        )
    )

    print("=" * 80)

    assert result.status == ResolutionAgentStatus.COMPLETED
    assert result.recommendation is not None
    assert result.recommendation.recommendation
    assert result.recommendation.rationale
