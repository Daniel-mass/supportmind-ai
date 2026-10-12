from app.evaluation.contracts.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationResult,
    AgentMetricResult,
)


def test_agent_evaluation_case_defaults():
    case = AgentEvaluationCase(
        case_id="agent_case_001",
        input="What are the priority levels in the support data?",
        actual_output="The support data contains multiple priority levels.",
    )

    assert case.case_id == "agent_case_001"
    assert case.tools_called == []
    assert case.expected_tools == []


def test_agent_metric_result_valid_score():
    result = AgentMetricResult(
        metric_name="task_success",
        score=0.9,
        passed=True,
        reason="The task was completed.",
    )

    assert result.score == 0.9
    assert result.passed is True


def test_agent_metric_result_rejects_invalid_score():
    try:
        AgentMetricResult(
            metric_name="task_success",
            score=1.5,
        )
        assert False
    except ValueError:
        assert True


def test_agent_evaluation_result_metric_count():
    result = AgentEvaluationResult(
        case_id="agent_case_001",
        metrics=[
            AgentMetricResult(
                metric_name="task_success",
                score=1.0,
                passed=True,
            ),
            AgentMetricResult(
                metric_name="tool_correctness",
                score=1.0,
                passed=True,
            ),
        ],
    )

    assert result.metric_count == 2
    assert result.evaluated is True


def test_agent_evaluation_result_empty():
    result = AgentEvaluationResult(
        case_id="agent_case_001",
    )

    assert result.metric_count == 0
    assert result.evaluated is False