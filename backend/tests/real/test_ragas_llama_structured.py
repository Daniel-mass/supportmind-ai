from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from ragas.llms import llm_factory


class Output(BaseModel):
    statements: list[str]


def test_llama_structured_generation() -> None:
    client = OpenAI(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
    )

    llm = llm_factory(
        "llama3.2:latest",
        provider="openai",
        client=client,
    )

    result = llm.generate(
        "Return three simple factual statements "
        "about a customer support ticket.",
        Output,
    )

    print("\nStructured result:")
    print(result)

    assert result.statements
    assert len(result.statements) == 3