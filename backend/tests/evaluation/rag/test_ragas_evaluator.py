from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.evaluation.contracts.evaluation import EvaluationCase
from app.evaluation.rag.ragas_evaluator import RAGASEvaluator


def _create_mock_metric(score: float, reason: str) -> MagicMock:
    metric = MagicMock()

    metric.ascore = AsyncMock(
        return_value=SimpleNamespace(
            value=score,
            reason=reason,
        )
    )

    return metric


def test_evaluate_case_without_embeddings() -> None:
    llm = MagicMock()

    faithfulness = _create_mock_metric(
        score=0.90,
        reason="The answer is supported by the retrieved context.",
    )

    context_precision = _create_mock_metric(
        score=0.85,
        reason="The retrieved context is relevant.",
    )

    context_recall = _create_mock_metric(
        score=0.80,
        reason="The required information was retrieved.",
    )

    with patch(
        "app.evaluation.rag.ragas_evaluator.Faithfulness",
        return_value=faithfulness,
    ), patch(
        "app.evaluation.rag.ragas_evaluator.ContextPrecisionWithoutReference",
        return_value=context_precision,
    ), patch(
        "app.evaluation.rag.ragas_evaluator.ContextRecall",
        return_value=context_recall,
    ):
        evaluator = RAGASEvaluator(
            llm=llm,
            embeddings=None,
        )

        case = EvaluationCase(
            case_id="rag_test_001",
            query="What is the refund period?",
            retrieved_context=[
                "Customers can request a refund within 30 days."
            ],
            generated_answer=(
                "Customers can request a refund within 30 days."
            ),
            reference_answer=(
                "Customers can request a refund within 30 days."
            ),
        )

        result = asyncio.run(
            evaluator.evaluate_case(case)
        )

    assert result.case_id == "rag_test_001"
    assert result.metric_count == 3
    assert result.evaluated is True
    assert result.overall_passed is True

    metric_names = {
        metric.metric_name
        for metric in result.metrics
    }

    assert metric_names == {
        "faithfulness",
        "context_precision",
        "context_recall",
    }


def test_answer_relevancy_is_disabled_without_embeddings() -> None:
    llm = MagicMock()

    faithfulness = MagicMock()
    context_precision = MagicMock()
    context_recall = MagicMock()

    with patch(
        "app.evaluation.rag.ragas_evaluator.Faithfulness",
        return_value=faithfulness,
    ), patch(
        "app.evaluation.rag.ragas_evaluator.ContextPrecisionWithoutReference",
        return_value=context_precision,
    ), patch(
        "app.evaluation.rag.ragas_evaluator.ContextRecall",
        return_value=context_recall,
    ):
        evaluator = RAGASEvaluator(
            llm=llm,
            embeddings=None,
        )

    assert evaluator.answer_relevancy is None


def test_evaluate_case_without_reference_skips_context_recall() -> None:
    llm = MagicMock()

    faithfulness = _create_mock_metric(
        score=0.90,
        reason="Supported.",
    )

    context_precision = _create_mock_metric(
        score=0.85,
        reason="Relevant.",
    )

    context_recall = _create_mock_metric(
        score=0.75,
        reason="Should not be called.",
    )

    with patch(
        "app.evaluation.rag.ragas_evaluator.Faithfulness",
        return_value=faithfulness,
    ), patch(
        "app.evaluation.rag.ragas_evaluator.ContextPrecisionWithoutReference",
        return_value=context_precision,
    ), patch(
        "app.evaluation.rag.ragas_evaluator.ContextRecall",
        return_value=context_recall,
    ):
        evaluator = RAGASEvaluator(
            llm=llm,
            embeddings=None,
        )

        case = EvaluationCase(
            case_id="rag_test_002",
            query="How do I reset my password?",
            retrieved_context=[
                "Users can reset their password from account settings."
            ],
            generated_answer=(
                "You can reset your password from account settings."
            ),
        )

        result = asyncio.run(
            evaluator.evaluate_case(case)
        )

    assert result.metric_count == 2

    metric_names = {
        metric.metric_name
        for metric in result.metrics
    }

    assert metric_names == {
        "faithfulness",
        "context_precision",
    }

    context_recall.ascore.assert_not_awaited()