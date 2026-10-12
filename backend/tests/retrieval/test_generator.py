"""
Unit tests for the provider-independent Graph-RAG generator.
"""

from __future__ import annotations

import pytest

from app.retrieval.graph_rag.generator import (
    GraphRAGGenerator,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGContext,
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
)


# ============================================================================
# Fake providers
# ============================================================================


class FakeProvider:
    """
    Deterministic provider used by unit tests.
    """

    def __init__(
        self,
        answer: str = "The ticket was delayed during approval. [E1]",
    ) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

        return self.answer


class FailingProvider:
    """
    Provider that raises a generation error.
    """

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        raise RuntimeError(
            "provider failed"
        )


class NonStringProvider:
    """
    Provider that violates the provider contract.
    """

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ):
        return {
            "answer": "invalid"
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def context() -> GraphRAGContext:
    return GraphRAGContext(
        query="Why was ticket ABC-123 delayed?",
        evidence=[
            GraphRAGEvidence(
                evidence_id="E1",
                source=(
                    GraphRAGEvidenceSource
                    .HELPDESK_GRAPH
                ),
                evidence_type=(
                    GraphRAGEvidenceType
                    .WORKFLOW_EVENT
                ),
                text=(
                    "Ticket ABC-123 spent 18 hours "
                    "waiting for approval."
                ),
                source_reference=(
                    "workflow:ABC-123:1"
                ),
            )
        ],
        selected_count=1,
        character_count=58,
    )


@pytest.fixture
def empty_context() -> GraphRAGContext:
    return GraphRAGContext(
        query="Why was ticket ABC-123 delayed?",
        evidence=[],
        selected_count=0,
        character_count=0,
    )


# ============================================================================
# Constructor
# ============================================================================


def test_constructor_accepts_valid_provider() -> None:
    provider = FakeProvider()

    generator = GraphRAGGenerator(
        provider=provider,
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    assert generator.provider is provider
    assert generator.provider_name == "ollama"
    assert generator.model_name == "qwen2.5:7b"


def test_constructor_rejects_none_provider() -> None:
    with pytest.raises(
        ValueError
    ):
        GraphRAGGenerator(
            provider=None,  # type: ignore[arg-type]
            provider_name="ollama",
            model_name="qwen2.5:7b",
        )


def test_constructor_rejects_invalid_provider() -> None:
    with pytest.raises(
        TypeError
    ):
        GraphRAGGenerator(
            provider=object(),
            provider_name="ollama",
            model_name="qwen2.5:7b",
        )


@pytest.mark.parametrize(
    "provider_name",
    [
        "",
        "   ",
        None,
    ],
)
def test_constructor_rejects_invalid_provider_name(
    provider_name,
) -> None:
    with pytest.raises(
        (TypeError, ValueError)
    ):
        GraphRAGGenerator(
            provider=FakeProvider(),
            provider_name=provider_name,
            model_name="qwen2.5:7b",
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
        None,
    ],
)
def test_constructor_rejects_invalid_model_name(
    model_name,
) -> None:
    with pytest.raises(
        (TypeError, ValueError)
    ):
        GraphRAGGenerator(
            provider=FakeProvider(),
            provider_name="ollama",
            model_name=model_name,
        )


# ============================================================================
# Generation
# ============================================================================


def test_generate_calls_provider(
    context: GraphRAGContext,
) -> None:
    provider = FakeProvider()

    generator = GraphRAGGenerator(
        provider=provider,
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    result = generator.generate(
        context
    )

    assert result == (
        "The ticket was delayed during approval. [E1]"
    )

    assert len(provider.calls) == 1


def test_generate_passes_built_prompts_to_provider(
    context: GraphRAGContext,
) -> None:
    provider = FakeProvider()

    generator = GraphRAGGenerator(
        provider=provider,
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    generator.generate(
        context
    )

    call = provider.calls[0]

    assert (
        "You are SupportMind Graph-RAG"
        in call["system_prompt"]
    )

    assert (
        "Why was ticket ABC-123 delayed?"
        in call["user_prompt"]
    )

    assert "[E1]" in (
        call["user_prompt"]
    )


def test_generate_uses_conservative_prompt_when_evidence_empty(
    empty_context: GraphRAGContext,
) -> None:
    provider = FakeProvider(
        answer="The available evidence is insufficient."
    )

    generator = GraphRAGGenerator(
        provider=provider,
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    result = generator.generate(
        empty_context
    )

    assert result == (
        "The available evidence is insufficient."
    )

    user_prompt = (
        provider.calls[0]["user_prompt"]
    )

    assert "Do not speculate." in (
        user_prompt
    )


# ============================================================================
# Validation
# ============================================================================


def test_generate_rejects_invalid_context() -> None:
    generator = GraphRAGGenerator(
        provider=FakeProvider(),
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    with pytest.raises(
        TypeError
    ):
        generator.generate(
            "invalid"  # type: ignore[arg-type]
        )


def test_generate_rejects_non_string_provider_output(
    context: GraphRAGContext,
) -> None:
    generator = GraphRAGGenerator(
        provider=NonStringProvider(),
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    with pytest.raises(
        TypeError,
        match="must return a string",
    ):
        generator.generate(
            context
        )


def test_generate_rejects_empty_provider_output(
    context: GraphRAGContext,
) -> None:
    generator = GraphRAGGenerator(
        provider=FakeProvider(
            answer="   "
        ),
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    with pytest.raises(
        RuntimeError,
        match="empty answer",
    ):
        generator.generate(
            context
        )


def test_generate_propagates_provider_error(
    context: GraphRAGContext,
) -> None:
    generator = GraphRAGGenerator(
        provider=FailingProvider(),
        provider_name="ollama",
        model_name="qwen2.5:7b",
    )

    with pytest.raises(
        RuntimeError,
        match="provider failed",
    ):
        generator.generate(
            context
        )