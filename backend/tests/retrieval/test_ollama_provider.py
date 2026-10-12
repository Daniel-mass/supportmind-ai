"""
Unit tests for the SupportMind Ollama provider.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.retrieval.graph_rag.providers.ollama import (
    OllamaGenerationConfig,
    OllamaProvider,
)

# ============================================================================
# Fake clients
# ============================================================================


class FakeOllamaClient:
    """
    Deterministic fake Ollama client.
    """

    def __init__(
        self,
        response,
    ) -> None:
        self.response = response
        self.calls: list[dict] = []
        self.closed = False

    def chat(
        self,
        *,
        model,
        messages,
        options=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "options": options,
                "kwargs": kwargs,
            }
        )

        return self.response

    def close(self) -> None:
        self.closed = True


class FailingOllamaClient:
    """
    Fake client used to test provider error handling.
    """

    def chat(
        self,
        *,
        model,
        messages,
        options=None,
        **kwargs,
    ):
        raise RuntimeError(
            "connection failed"
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config() -> OllamaGenerationConfig:
    return OllamaGenerationConfig(
        host="http://localhost:11434",
        model_name="qwen2.5:7b",
        temperature=0.1,
        max_tokens=512,
        timeout=30.0,
    )


# ============================================================================
# Configuration
# ============================================================================


def test_generation_config_defaults() -> None:
    config = OllamaGenerationConfig()

    assert config.host == (
        "http://localhost:11434"
    )
    assert config.model_name == (
        "qwen2.5:7b"
    )
    assert config.temperature == 0.1
    assert config.max_tokens == 1024
    assert config.timeout == 120.0


def test_generation_config_rejects_empty_model() -> None:
    with pytest.raises(
        ValueError
    ):
        OllamaGenerationConfig(
            model_name="   "
        )


def test_generation_config_rejects_negative_temperature() -> None:
    with pytest.raises(
        ValueError
    ):
        OllamaGenerationConfig(
            temperature=-0.1
        )


def test_generation_config_rejects_invalid_max_tokens() -> None:
    with pytest.raises(
        ValueError
    ):
        OllamaGenerationConfig(
            max_tokens=0
        )


def test_generation_config_rejects_invalid_timeout() -> None:
    with pytest.raises(
        ValueError
    ):
        OllamaGenerationConfig(
            timeout=0
        )


# ============================================================================
# Generation
# ============================================================================


def test_generate_with_mapping_response(
    config: OllamaGenerationConfig,
) -> None:
    client = FakeOllamaClient(
        {
            "message": {
                "content": (
                    "The ticket was delayed "
                    "during approval. [E2]"
                )
            }
        }
    )

    provider = OllamaProvider(
        config=config,
        client=client,
    )

    result = provider.generate(
        system_prompt=(
            "Use evidence only."
        ),
        user_prompt=(
            "Why was ABC-123 delayed?"
        ),
    )

    assert result == (
        "The ticket was delayed "
        "during approval. [E2]"
    )

    assert len(client.calls) == 1

    call = client.calls[0]

    assert call["model"] == (
        "qwen2.5:7b"
    )

    assert call["messages"] == [
        {
            "role": "system",
            "content": (
                "Use evidence only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Why was ABC-123 delayed?"
            ),
        },
    ]

    assert call["options"] == {
        "temperature": 0.1,
        "num_predict": 512,
    }


def test_generate_with_object_response(
    config: OllamaGenerationConfig,
) -> None:
    response = SimpleNamespace(
        message=SimpleNamespace(
            content="Grounded answer. [E1]"
        )
    )

    client = FakeOllamaClient(
        response
    )

    provider = OllamaProvider(
        config=config,
        client=client,
    )

    result = provider.generate(
        system_prompt="System",
        user_prompt="Question",
    )

    assert result == (
        "Grounded answer. [E1]"
    )


def test_generate_strips_prompt_whitespace(
    config: OllamaGenerationConfig,
) -> None:
    client = FakeOllamaClient(
        {
            "message": {
                "content": "Answer"
            }
        }
    )

    provider = OllamaProvider(
        config=config,
        client=client,
    )

    provider.generate(
        system_prompt="  System instructions  ",
        user_prompt="  User question  ",
    )

    messages = (
        client.calls[0]["messages"]
    )

    assert messages[0]["content"] == (
        "System instructions"
    )
    assert messages[1]["content"] == (
        "User question"
    )


# ============================================================================
# Validation
# ============================================================================


def test_generate_rejects_empty_system_prompt(
    config: OllamaGenerationConfig,
) -> None:
    provider = OllamaProvider(
        config=config,
        client=FakeOllamaClient(
            {
                "message": {
                    "content": "Answer"
                }
            }
        ),
    )

    with pytest.raises(
        ValueError
    ):
        provider.generate(
            system_prompt="   ",
            user_prompt="Question",
        )


def test_generate_rejects_empty_user_prompt(
    config: OllamaGenerationConfig,
) -> None:
    provider = OllamaProvider(
        config=config,
        client=FakeOllamaClient(
            {
                "message": {
                    "content": "Answer"
                }
            }
        ),
    )

    with pytest.raises(
        ValueError
    ):
        provider.generate(
            system_prompt="System",
            user_prompt="   ",
        )


# ============================================================================
# Error handling
# ============================================================================


def test_generate_wraps_client_failure(
    config: OllamaGenerationConfig,
) -> None:
    provider = OllamaProvider(
        config=config,
        client=FailingOllamaClient(),
    )

    with pytest.raises(
        RuntimeError,
        match="Ollama generation failed",
    ):
        provider.generate(
            system_prompt="System",
            user_prompt="Question",
        )


def test_generate_rejects_empty_response(
    config: OllamaGenerationConfig,
) -> None:
    provider = OllamaProvider(
        config=config,
        client=FakeOllamaClient(
            {
                "message": {
                    "content": "   "
                }
            }
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        provider.generate(
            system_prompt="System",
            user_prompt="Question",
        )


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {"message": {}},
        {"message": {"content": None}},
    ],
)
def test_generate_rejects_unusable_response(
    config: OllamaGenerationConfig,
    response,
) -> None:
    provider = OllamaProvider(
        config=config,
        client=FakeOllamaClient(
            response
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="empty response",
    ):
        provider.generate(
            system_prompt="System",
            user_prompt="Question",
        )


# ============================================================================
# Lifecycle
# ============================================================================


def test_close_calls_client_close(
    config: OllamaGenerationConfig,
) -> None:
    client = FakeOllamaClient(
        {
            "message": {
                "content": "Answer"
            }
        }
    )

    provider = OllamaProvider(
        config=config,
        client=client,
    )

    provider.close()

    assert client.closed is True