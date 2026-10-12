from unittest.mock import MagicMock

from app.evaluation.contracts.agent_evaluation import AgentEvaluationCase
from app.evaluation.runners.agent_evaluator import AgentEvaluator


def test_agent_evaluator_runs_all_metrics():
    task_success = MagicMock()
    task_success.evaluate.return_value = MagicMock(
        score=1.0,
        success=True,
        reason="Task completed successfully.",
    )

    tool_correctness = MagicMock()
    tool_correctness.evaluate.return_value = MagicMock(
        score=1.0,
        success=True,
        reason="Expected tools were used correctly.",
    )

    response_quality = MagicMock()
    response_quality.evaluate.return_value = MagicMock(
        score=1.0,
        success=True,
        reason="Final response is relevant.",
    )

    evaluator = AgentEvaluator(
        task_success_evaluator=task_success,
        tool_correctness_evaluator=tool_correctness,
        response_quality_evaluator=response_quality,
    )

    case = AgentEvaluationCase(
        case_id="agent_case_001",
        input="What is the refund period?",
        actual_output="NovaCare Technologies provides a 30-day refund period.",
        expected_output="NovaCare Technologies provides a 30-day refund period.",
        tools_called=["retrieval", "graph_rag"],
        expected_tools=["retrieval", "graph_rag"],
    )

    results = evaluator.evaluate_case(case)

    assert len(results) == 3

    assert results[0].metric_name == "task_success"
    assert results[0].score == 1.0
    assert results[0].passed is True

    assert results[1].metric_name == "tool_correctness"
    assert results[1].score == 1.0
    assert results[1].passed is True

    assert results[2].metric_name == "response_quality"
    assert results[2].score == 1.0
    assert results[2].passed is True

    task_success.evaluate.assert_called_once()
    tool_correctness.evaluate.assert_called_once()
    response_quality.evaluate.assert_called_once()