from __future__ import annotations

from app.evaluation.contracts.agent_evaluation import (
    AgentEvaluationCase,
    AgentMetricResult,
)
from app.evaluation.metrics.agent_response_quality import (
    AgentResponseQualityEvaluator,
)
from app.evaluation.metrics.agent_task_success import (
    AgentTaskSuccessEvaluator,
)
from app.evaluation.metrics.agent_tool_correctness import (
    AgentToolCorrectnessEvaluator,
)


class AgentEvaluator:
    """Run the selected DeepEval metrics for an agent case."""

    def __init__(
        self,
        task_success_evaluator: AgentTaskSuccessEvaluator,
        tool_correctness_evaluator: AgentToolCorrectnessEvaluator,
        response_quality_evaluator: AgentResponseQualityEvaluator,
    ) -> None:
        self.task_success_evaluator = task_success_evaluator
        self.tool_correctness_evaluator = (
            tool_correctness_evaluator
        )
        self.response_quality_evaluator = (
            response_quality_evaluator
        )

    def evaluate_case(
        self,
        case: AgentEvaluationCase,
    ) -> list[AgentMetricResult]:
        """Evaluate one agent case using all three metrics."""

        task_metric = self.task_success_evaluator.evaluate(
            input_text=case.input,
            actual_output=case.actual_output,
            expected_output=case.expected_output,
        )

        tool_metric = self.tool_correctness_evaluator.evaluate(
            input_text=case.input,
            actual_output=case.actual_output,
            tools_called=case.tools_called,
            expected_tools=case.expected_tools,
        )

        response_metric = self.response_quality_evaluator.evaluate(
            input_text=case.input,
            actual_output=case.actual_output,
        )

        return [
            AgentMetricResult(
                metric_name="task_success",
                score=task_metric.score,
                passed=task_metric.success,
                reason=task_metric.reason,
            ),
            AgentMetricResult(
                metric_name="tool_correctness",
                score=tool_metric.score,
                passed=tool_metric.success,
                reason=tool_metric.reason,
            ),
            AgentMetricResult(
                metric_name="response_quality",
                score=response_metric.score,
                passed=response_metric.success,
                reason=response_metric.reason,
            ),
        ]