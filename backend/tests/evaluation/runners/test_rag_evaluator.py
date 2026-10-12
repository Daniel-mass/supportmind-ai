from __future__ import annotations

from unittest.mock import MagicMock

from app.evaluation.contracts.evaluation import EvaluationCase
from app.evaluation.runners.rag_evaluator import RAGEvaluator


def test_rag_evaluator_evaluate_case() -> None:
    """Verify evaluation of a single RAG case."""

    mock_faithfulness_evaluator = MagicMock()

    mock_metric = MagicMock()
    mock_metric.score = 0.95
    mock_metric.success = True
    mock_metric.reason = "The answer is supported by the context."

    mock_faithfulness_evaluator.evaluate.return_value = mock_metric

    evaluator = RAGEvaluator(
        faithfulness_evaluator=mock_faithfulness_evaluator,
    )

    case = EvaluationCase(
        case_id="test_case_001",
        query="What is the refund period?",
        retrieved_context=[
            "Customers can request a full refund within 30 days."
        ],
        generated_answer="Customers have 30 days to request a full refund.",
        metadata={
            "category": "refund_policy",
        },
    )

    result = evaluator.evaluate_case(case)

    mock_faithfulness_evaluator.evaluate.assert_called_once_with(
        input_text=case.query,
        actual_output=case.generated_answer,
        retrieval_context=case.retrieved_context,
    )

    assert result.metric_name == "faithfulness"
    assert result.score == 0.95
    assert result.passed is True
    assert result.reason == "The answer is supported by the context."
    assert result.metadata["case_id"] == "test_case_001"
    assert result.metadata["category"] == "refund_policy"


def test_rag_evaluator_evaluate_cases() -> None:
    """Verify evaluation of multiple RAG cases."""

    mock_faithfulness_evaluator = MagicMock()

    first_metric = MagicMock()
    first_metric.score = 1.0
    first_metric.success = True
    first_metric.reason = "Fully supported."

    second_metric = MagicMock()
    second_metric.score = 0.0
    second_metric.success = False
    second_metric.reason = "The answer contradicts the context."

    mock_faithfulness_evaluator.evaluate.side_effect = [
        first_metric,
        second_metric,
    ]

    evaluator = RAGEvaluator(
        faithfulness_evaluator=mock_faithfulness_evaluator,
    )

    cases = [
        EvaluationCase(
            case_id="case_001",
            query="What is the refund period?",
            retrieved_context=["Refunds are available within 30 days."],
            generated_answer="Refunds are available within 30 days.",
        ),
        EvaluationCase(
            case_id="case_002",
            query="What is the refund period?",
            retrieved_context=["Refunds are available within 30 days."],
            generated_answer="Refunds are available within 90 days.",
        ),
    ]

    results = evaluator.evaluate_cases(cases)

    assert len(results) == 2

    assert results[0].metric_name == "faithfulness"
    assert results[0].score == 1.0
    assert results[0].passed is True

    assert results[1].metric_name == "faithfulness"
    assert results[1].score == 0.0
    assert results[1].passed is False

    assert mock_faithfulness_evaluator.evaluate.call_count == 2