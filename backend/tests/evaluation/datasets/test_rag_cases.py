from __future__ import annotations

from app.evaluation.datasets.rag_cases import get_faithfulness_cases


def test_get_faithfulness_cases() -> None:
    """Verify the faithfulness evaluation dataset."""

    cases = get_faithfulness_cases()

    assert len(cases) == 2

    assert cases[0].case_id == "faithfulness_grounded_refund"
    assert cases[1].case_id == "faithfulness_hallucinated_refund"

    for case in cases:
        assert case.query
        assert case.generated_answer
        assert len(case.retrieved_context) > 0
        assert case.metadata["expected_faithfulness"] in {
            "high",
            "low",
        }