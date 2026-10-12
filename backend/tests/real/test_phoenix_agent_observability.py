from __future__ import annotations

import asyncio
import os

import pytest

from app.agents.coordinator.agent import CoordinatorAgent
from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
)
from app.observability.phoenix import setup_phoenix


RUN_REAL_PHOENIX = os.getenv(
    "RUN_REAL_PHOENIX",
    "0",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_GRAPH_RAG_MODEL",
    "qwen3:8b",
)


@pytest.mark.real
def test_real_coordinator_phoenix_trace() -> None:
    """Run the real Coordinator Agent and send its trace to Phoenix."""

    if RUN_REAL_PHOENIX != "1":
        pytest.skip(
            "Set RUN_REAL_PHOENIX=1 to run real Phoenix observability."
        )

    print("\n")
    print("=" * 90)
    print("REAL SUPPORTMIND AGENT → PHOENIX OBSERVABILITY")
    print("=" * 90)

    print(f"Agent model: {OLLAMA_MODEL}")

    # Configure Phoenix BEFORE creating/running the agent.
    setup_phoenix()

    print("[OK] Phoenix tracing configured.")

    from autogen_ext.models.ollama import OllamaChatCompletionClient

    model_client = OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host="http://127.0.0.1:11434",
    )

    coordinator = CoordinatorAgent(
        model_client=model_client,
    )

    state = CoordinatorState(
        request="What is NovaCare's refund period?",
        requested_capabilities=[
            Capability.RETRIEVAL,
            Capability.GENERATION,
        ],
        completed_capabilities=[
            Capability.RETRIEVAL,
        ],
    )

    print("[OK] Real CoordinatorAgent created.")

    try:
        print("\n[RUNNING] Coordinator Agent...")

        result_state = asyncio.run(
            coordinator.run(state)
        )

    finally:
        close_method = getattr(
            model_client,
            "close",
            None,
        )

        if close_method is not None:
            close_result = close_method()

            if hasattr(close_result, "__await__"):
                asyncio.run(close_result)

    assert result_state.final_response
    assert result_state.status.value == "completed"

    print("\n[OK] Agent execution completed.")

    print("\nFinal response:")
    print(result_state.final_response)

    print("\n" + "=" * 90)
    print("PHOENIX AGENT OBSERVABILITY RUN PASSED")
    print("=" * 90)
    print()
    print("Open Phoenix:")
    print("http://localhost:6006")
    print("=" * 90)