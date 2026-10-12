from __future__ import annotations

from deepeval.metrics.faithfulness.schema import Claims

from app.evaluation.models.qwen3_deepeval import Qwen3DeepEvalModel


def test_qwen3_deepeval_plain_generation() -> None:
    """Verify that the adapter can generate normal text."""

    model = Qwen3DeepEvalModel(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    result = model.generate(
        "Reply with exactly: QWEN3 DEEPEVAL ADAPTER PASSED"
    )

    print("\nPlain generation result:")
    print(result)

    assert isinstance(result, str)
    assert result.strip()


def test_qwen3_deepeval_structured_generation() -> None:
    """Verify that the adapter can generate a Pydantic schema."""

    model = Qwen3DeepEvalModel(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    result = model.generate(
        (
            "Extract the factual claims from this sentence. "
            "Return the claims only using the provided schema.\n\n"
            "NovaCare Technologies provides customers with "
            "a 30-day full refund."
        ),
        schema=Claims,
    )

    print("\nStructured generation result:")
    print(result)

    assert isinstance(result, Claims)
    assert len(result.claims) > 0

    combined_claims = " ".join(result.claims)

    assert (
        "30-day" in combined_claims
        or "30 days" in combined_claims
    )