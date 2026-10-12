from __future__ import annotations

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel


class AgentResponseQualityEvaluator:
    """Evaluate the relevance of the agent's final response."""

    def __init__(
        self,
        model: Qwen3DeepEvalModel,
        threshold: float = 0.7,
    ) -> None:
        self.metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=model,
            include_reason=True,
            async_mode=False,
        )

    def evaluate(
        self,
        input_text: str,
        actual_output: str,
    ) -> AnswerRelevancyMetric:
        """Evaluate one agent response."""

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
        )

        self.metric.measure(test_case)

        return self.metric