from __future__ import annotations

from deepeval.models import OllamaModel


def test_deepeval_ollama_generation() -> None:
    model = OllamaModel(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.0,
    )

    result, cost = model.generate(
        "Reply with exactly: DEEPEVAL OLLAMA TEST PASSED"
    )

    print("\nDeepEval result:")
    print(result)

    print("\nCost:")
    print(cost)

    assert result