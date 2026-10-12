"""
Real Coordinator integration test for SupportMind AI.

This test validates the complete Coordinator -> AutoGen -> Ollama
execution path using a real Qwen3:8b model.

This test intentionally does NOT connect:
- Retrieval services
- Classification services
- Qdrant
- Neo4j
- Graph-RAG
- Other specialized agents

Those integrations will be tested separately when their tools
are connected to the Coordinator.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.coordinator.agent import CoordinatorAgent
from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
    PredictionSignal,
)


RUN_REAL_COORDINATOR = (
    os.getenv("RUN_REAL_COORDINATOR", "0") == "1"
)


pytestmark = pytest.mark.skipif(
    not RUN_REAL_COORDINATOR,
    reason=(
        "Real Coordinator integration disabled. "
        "Set RUN_REAL_COORDINATOR=1 to run."
    ),
)


def test_real_coordinator_with_real_support_question() -> None:
    """
    Verify that the real Coordinator Agent can process a realistic
    customer-support question using Qwen3:8b.
    """

    model_client = OllamaChatCompletionClient(
        model=os.getenv(
            "SUPPORTMIND_LLM_MODEL",
            "qwen3:8b",
        ),
        host=os.getenv(
            "OLLAMA_HOST",
            "http://127.0.0.1:11434",
        ),
        options={
            "temperature": 0.0,
            "num_predict": 512,
        },
        timeout=120.0,
    )

    state = CoordinatorState(
        request=(
            "A customer says their subscription was charged twice "
            "this month. They are frustrated and want to know what "
            "happened and how the duplicate charge can be resolved."
        ),
        requested_capabilities=[
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
            Capability.RESOLUTION,
        ],
        predictions={
            "priority": PredictionSignal(
                value="High",
                confidence=0.82,
                source="priority_classifier",
            ),
            "sentiment": PredictionSignal(
                value="Negative",
                confidence=0.91,
                source="sentiment_signal",
            ),
        },
    )

    coordinator = CoordinatorAgent(
        model_client=model_client,
    )

    async def run_coordinator() -> CoordinatorState:
        try:
            return await coordinator.run(state)

        finally:
            await model_client.close()

    result = asyncio.run(run_coordinator())

    assert result.status.value == "completed"

    assert result.final_response is not None

    response = result.final_response.strip()

    assert response

    assert len(response) >= 30

    assert result.iteration_count == 1

    assert not result.errors

    assert result.request == state.request

    # The Coordinator should preserve the advisory ML signals.
    assert result.predictions.priority is not None
    assert result.predictions.priority.value == "High"

    assert result.predictions.sentiment is not None
    assert result.predictions.sentiment.value == "Negative"

    print("\n" + "=" * 80)
    print("REAL COORDINATOR RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)