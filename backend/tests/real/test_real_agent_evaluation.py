from __future__ import annotations

import os

import pytest

from deepeval.metrics import AnswerRelevancyMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from app.agents.coordinator.agent import CoordinatorAgent
from app.agents.coordinator.state import (
    AgentResult,
    Capability,
    CoordinatorState,
    EvidenceReference,
    PredictionSignal,
)
from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel


RUN_REAL_AGENT_EVALUATION = os.getenv(
    "RUN_REAL_AGENT_EVALUATION",
    "0",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_GRAPH_RAG_MODEL",
    "qwen3:8b",
)


@pytest.mark.real
def test_real_supportmind_coordinator_agent_evaluation() -> None:
    if RUN_REAL_AGENT_EVALUATION != "1":
        pytest.skip(
            "Set RUN_REAL_AGENT_EVALUATION=1 to run the real Agent Evaluation."
        )

    print("\n")
    print("=" * 90)
    print("REAL SUPPORTMIND AGENT → DEEPEVAL EVALUATION")
    print("=" * 90)
    print(f"Agent model: {OLLAMA_MODEL}")

    # ------------------------------------------------------------------
    # 1. Build realistic workflow state
    # ------------------------------------------------------------------

    state = CoordinatorState(
        request="What is NovaCare's refund period?",
        requested_capabilities=[
            Capability.RETRIEVAL,
            Capability.GENERATION,
        ],
        completed_capabilities=[
            Capability.RETRIEVAL,
        ],
        agent_results={
            "retrieval_agent": AgentResult(
                agent_name="retrieval_agent",
                success=True,
                result={
                    "retrieval_type": "hybrid",
                    "document_count": 2,
                },
                evidence_ids=["E1", "E2"],
            ),
        },
        evidence=[
            EvidenceReference(
                evidence_id="E1",
                source="knowledge_base",
                content=(
                    "NovaCare Technologies allows customers to request "
                    "a full refund within 30 days of purchase."
                ),
                provenance={
                    "document": "refund_policy.pdf",
                },
            ),
            EvidenceReference(
                evidence_id="E2",
                source="knowledge_base",
                content=(
                    "Refund requests must include the original "
                    "order information."
                ),
                provenance={
                    "document": "refund_policy.pdf",
                },
            ),
        ],
        predictions={
            "category": PredictionSignal(
                value="refund_policy",
                confidence=0.96,
                source="supportmind_classifier",
            ),
        },
    )

    print("\n[OK] CoordinatorState created.")

    # ------------------------------------------------------------------
    # 2. Create the REAL AutoGen Coordinator
    #
    # IMPORTANT:
    # Replace this model-client construction with the exact Ollama
    # AutoGen client factory already used by your SupportMind runtime
    # if your project has one.
    # ------------------------------------------------------------------

    from autogen_ext.models.ollama import OllamaChatCompletionClient

    model_client = OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host="http://127.0.0.1:11434",
    )

    coordinator = CoordinatorAgent(
        model_client=model_client,
    )

    print("[OK] Real CoordinatorAgent created.")

    # ------------------------------------------------------------------
    # 3. Execute the REAL Coordinator
    # ------------------------------------------------------------------

    result_state = None

    try:
        print("\n[RUNNING] Coordinator Agent...")

        result_state = __import__("asyncio").run(
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
                __import__("asyncio").run(close_result)

    assert result_state is not None

    print("\n[OK] Coordinator execution completed.")
    print(f"Workflow status: {result_state.status.value}")
    print(f"Iterations: {result_state.iteration_count}")

    assert result_state.final_response
    assert result_state.status.value == "completed"

    print("\nFinal response:")
    print(result_state.final_response)

    # ------------------------------------------------------------------
    # 4. Build the Agent Evaluation Case from REAL execution
    # ------------------------------------------------------------------

    evaluation_case = LLMTestCase(
        input=result_state.request,
        actual_output=result_state.final_response,
        expected_output=(
            "NovaCare Technologies provides a full refund "
            "within 30 days of purchase."
        ),
    )

    print("\n[OK] DeepEval test case created.")

    # ------------------------------------------------------------------
    # 5. DeepEval Task Completion
    # ------------------------------------------------------------------

    deepeval_model = Qwen3DeepEvalModel(
        model=OLLAMA_MODEL,
    )

    task_metric = TaskCompletionMetric(
        threshold=0.7,
        model=deepeval_model,
        include_reason=True,
        async_mode=False,
    )

    print("\n[RUNNING] DeepEval Task Completion...")

    task_metric.measure(evaluation_case)

    print(f"Task Completion score: {task_metric.score}")
    print(f"Task Completion passed: {task_metric.success}")
    print(f"Task Completion reason: {task_metric.reason}")

    # ------------------------------------------------------------------
    # 6. DeepEval Final Response Quality
    # ------------------------------------------------------------------

    response_metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=deepeval_model,
        include_reason=True,
        async_mode=False,
    )

    print("\n[RUNNING] DeepEval Answer Relevancy...")

    response_metric.measure(evaluation_case)

    print(f"Answer Relevancy score: {response_metric.score}")
    print(f"Answer Relevancy passed: {response_metric.success}")
    print(f"Answer Relevancy reason: {response_metric.reason}")

    # ------------------------------------------------------------------
    # 7. Evaluation assertions
    # ------------------------------------------------------------------

    assert task_metric.success
    assert response_metric.success

    print("\n" + "=" * 90)
    print("REAL AGENT EVALUATION PASSED")
    print("=" * 90)