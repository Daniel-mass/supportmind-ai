from __future__ import annotations

from app.evaluation.datasets.rag_cases import get_faithfulness_cases
from app.evaluation.metrics.faithfulness import FaithfulnessEvaluator
from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel
from app.evaluation.runners.rag_evaluator import RAGEvaluator


def test_real_rag_faithfulness_evaluation() -> None:
    """Run the real RAG faithfulness evaluation using Qwen3."""

    model = Qwen3DeepEvalModel(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    faithfulness_evaluator = FaithfulnessEvaluator(
        model=model,
        threshold=0.7,
    )

    evaluator = RAGEvaluator(
        faithfulness_evaluator=faithfulness_evaluator,
    )

    cases = get_faithfulness_cases()

    results = evaluator.evaluate_cases(cases)

    print("\n========================================")
    print("RAG FAITHFULNESS EVALUATION")
    print("========================================")

    for result in results:
        print(f"\nCase ID: {result.metadata['case_id']}")
        print(f"Score: {result.score}")
        print(f"Passed: {result.passed}")
        print(f"Reason: {result.reason}")

    assert len(results) == len(cases)

    for result in results:
        assert result.metric_name == "faithfulness"
        assert result.score is not None
        assert result.reason is not None

    grounded_result = results[0]
    hallucinated_result = results[1]

    assert grounded_result.score >= 0.7
    assert hallucinated_result.score < 0.7