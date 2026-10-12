from __future__ import annotations

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase


def test_deepeval_faithfulness_grounded() -> None:
    model = OllamaModel(
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

    print("\nGrounded answer score:")
    print(metric.score)

    print("\nGrounded answer reason:")
    print(metric.reason)

    assert metric.score is not None
    assert metric.score >= 0.7


def test_deepeval_faithfulness_hallucinated() -> None:
    model = OllamaModel(
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
            "a 90-day full refund."
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

    print("\nHallucinated answer score:")
    print(metric.score)

    print("\nHallucinated answer reason:")
    print(metric.reason)

    assert metric.score is not None
    assert metric.score < 0.7