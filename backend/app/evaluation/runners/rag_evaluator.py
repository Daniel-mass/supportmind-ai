from __future__ import annotations

from app.evaluation.contracts.evaluation import (
    EvaluationCase,
    EvaluationMetricResult,
)
from app.evaluation.metrics.faithfulness import FaithfulnessEvaluator


class RAGEvaluator:
    """Run RAG evaluation metrics for evaluation cases."""

    def __init__(
        self,
        faithfulness_evaluator: FaithfulnessEvaluator,
    ) -> None:
        self.faithfulness_evaluator = faithfulness_evaluator

    def evaluate_case(
        self,
        case: EvaluationCase,
    ) -> EvaluationMetricResult:
        """Evaluate one RAG case using the faithfulness metric."""

        metric = self.faithfulness_evaluator.evaluate(
            input_text=case.query,
            actual_output=case.generated_answer,
            retrieval_context=case.retrieved_context,
        )

        return EvaluationMetricResult(
            metric_name="faithfulness",
            score=metric.score,
            passed=metric.success,
            reason=metric.reason,
            metadata={
                "case_id": case.case_id,
                **case.metadata,
            },
        )

    def evaluate_cases(
        self,
        cases: list[EvaluationCase],
    ) -> list[EvaluationMetricResult]:
        """Evaluate multiple RAG cases."""

        results: list[EvaluationMetricResult] = []

        for case in cases:
            result = self.evaluate_case(case)
            results.append(result)

        return results