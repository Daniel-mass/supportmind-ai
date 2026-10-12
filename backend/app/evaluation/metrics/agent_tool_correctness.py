from __future__ import annotations

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase


class AgentToolCorrectnessEvaluator:
    """Evaluate whether the agent used the expected tools correctly."""

    def __init__(
        self,
        threshold: float = 0.7,
    ) -> None:
        self.metric = ToolCorrectnessMetric(
            threshold=threshold,
            include_reason=True,
            async_mode=False,
        )

    def evaluate(
        self,
        input_text: str,
        actual_output: str,
        tools_called: list,
        expected_tools: list,
    ) -> ToolCorrectnessMetric:
        """Evaluate one agent tool trajectory."""

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            tools_called=tools_called,
            expected_tools=expected_tools,
        )

        self.metric.measure(test_case)

        return self.metric