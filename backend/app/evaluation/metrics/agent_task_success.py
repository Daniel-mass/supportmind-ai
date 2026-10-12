from __future__ import annotations

from deepeval.metrics import TaskCompletionMetric
from deepeval.test_case import LLMTestCase

from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel


class AgentTaskSuccessEvaluator:
    """Evaluate whether an agent completed the requested task."""

    def __init__(
        self,
        model: Qwen3DeepEvalModel,
        threshold: float = 0.7,
    ) -> None:
        self.metric = TaskCompletionMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
            async_mode=False,
        )

    def evaluate(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
    ) -> TaskCompletionMetric:
        """Evaluate one agent task."""

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
        )

        self.metric.measure(test_case)

        return self.metric