from __future__ import annotations

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel


def test_deepeval_custom_model_faithfulness() -> None:
    """Verify FaithfulnessMetric using our custom Qwen3 adapter."""

    model = Qwen3DeepEvalModel(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    metric = FaithfulnessMetric(
        threshold=0.7,
        model=model,
        include_reason=True,
        async_mode=False,
    )

    test_case = LLMTestCase(
        input="What is NovaCare's refund period?",
        actual_output=(
            "NovaCare Technologies provides customers with "
            "a 30-day full refund."
        ),
        retrieval_context=[
            (
                "NovaCare Technologies allows customers to request "
                "a full refund within 30 days of purchase."
            ),
            (
                "Refund requests must include the original order "
                "information."
            ),
        ],
    )

    metric.measure(test_case)

    print("\nFaithfulness score:")
    print(metric.score)

    print("\nFaithfulness reason:")
    print(metric.reason)

    assert metric.score is not None
    assert metric.reason is not None