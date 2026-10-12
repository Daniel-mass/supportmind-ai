from __future__ import annotations

from app.evaluation.contracts.evaluation import EvaluationCase


def get_faithfulness_cases() -> list[EvaluationCase]:
    """Return representative RAG cases for faithfulness evaluation."""

    return [
        EvaluationCase(
            case_id="faithfulness_grounded_refund",
            query="What is NovaCare's refund period?",
            retrieved_context=[
                (
                    "NovaCare Technologies allows customers to request "
                    "a full refund within 30 days of purchase."
                ),
                (
                    "Refund requests must include the original order "
                    "information."
                ),
            ],
            generated_answer=(
                "NovaCare Technologies provides customers with "
                "a 30-day full refund."
            ),
            reference_answer=(
                "NovaCare Technologies allows customers to request "
                "a full refund within 30 days of purchase."
            ),
            metadata={
                "category": "refund_policy",
                "expected_faithfulness": "high",
            },
        ),
        EvaluationCase(
            case_id="faithfulness_hallucinated_refund",
            query="What is NovaCare's refund period?",
            retrieved_context=[
                (
                    "NovaCare Technologies allows customers to request "
                    "a full refund within 30 days of purchase."
                ),
                (
                    "Refund requests must include the original order "
                    "information."
                ),
            ],
            generated_answer=(
                "NovaCare Technologies provides customers with "
                "a 90-day full refund."
            ),
            reference_answer=(
                "NovaCare Technologies allows customers to request "
                "a full refund within 30 days of purchase."
            ),
            metadata={
                "category": "refund_policy",
                "expected_faithfulness": "low",
            },
        ),
    ]