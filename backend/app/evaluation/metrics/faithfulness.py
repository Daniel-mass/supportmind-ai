from __future__ import annotations

from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel


class FaithfulnessEvaluator:
    """Evaluate whether generated answers are supported by retrieved context."""

    def __init__(
        self,
        model: Qwen3DeepEvalModel,
        threshold: float = 0.7,
    ) -> None:
        self.metric = FaithfulnessMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
            async_mode=False,
        )

    def evaluate(
        self,
        input_text: str,
        actual_output: str,
        retrieval_context: list[str],
    ) -> FaithfulnessMetric:
        """Evaluate one generated answer for faithfulness."""

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )

        self.metric.measure(test_case)

        return self.metric