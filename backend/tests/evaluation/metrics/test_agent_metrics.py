from unittest.mock import MagicMock, patch

from app.evaluation.metrics.agent_response_quality import (
    AgentResponseQualityEvaluator,
)
from app.evaluation.metrics.agent_task_success import (
    AgentTaskSuccessEvaluator,
)
from app.evaluation.metrics.agent_tool_correctness import (
    AgentToolCorrectnessEvaluator,
)


@patch(
    "app.evaluation.metrics.agent_task_success.TaskCompletionMetric"
)
def test_task_success_evaluator_calls_metric(mock_metric_class):
    fake_metric = MagicMock()
    fake_metric.score = 1.0
    fake_metric.success = True
    fake_metric.reason = "Task completed."
    mock_metric_class.return_value = fake_metric

    evaluator = AgentTaskSuccessEvaluator(
        model=MagicMock()
    )

    result = evaluator.evaluate(
        input_text="What is the refund period?",
        actual_output="The refund period is 30 days.",
        expected_output="The refund period is 30 days.",
    )

    fake_metric.measure.assert_called_once()
    assert result.score == 1.0
    assert result.success is True


@patch(
    "app.evaluation.metrics.agent_tool_correctness.ToolCorrectnessMetric"
)
def test_tool_correctness_evaluator_calls_metric(mock_metric_class):
    fake_metric = MagicMock()
    fake_metric.score = 1.0
    fake_metric.success = True
    fake_metric.reason = "Expected tool was used."

    mock_metric_class.return_value = fake_metric

    evaluator = AgentToolCorrectnessEvaluator()

    result = evaluator.evaluate(
        input_text="Find support ticket information.",
        actual_output="The ticket information is available.",
        tools_called=[],
        expected_tools=[],
    )

    fake_metric.measure.assert_called_once()
    assert result.score == 1.0
    assert result.success is True


@patch(
    "app.evaluation.metrics.agent_response_quality.AnswerRelevancyMetric"
)
def test_response_quality_evaluator_calls_metric(mock_metric_class):
    fake_metric = MagicMock()
    fake_metric.score = 1.0
    fake_metric.success = True
    fake_metric.reason = "The response is relevant."
    mock_metric_class.return_value = fake_metric

    evaluator = AgentResponseQualityEvaluator(
        model=MagicMock()
    )

    result = evaluator.evaluate(
        input_text="What is the refund period?",
        actual_output="The refund period is 30 days.",
    )

    fake_metric.measure.assert_called_once()
    assert result.score == 1.0
    assert result.success is True