"""
Unit tests for the Ollama LLM provider.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.llm.config import LLMSettings
from app.llm.providers.ollama import OllamaProvider


def test_ollama_provider_initializes_with_valid_settings():
    """
    OllamaProvider should initialize with valid Ollama settings.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
    )

    provider = OllamaProvider(settings)

    assert provider is not None


def test_ollama_provider_rejects_invalid_settings_type():
    """
    OllamaProvider should reject objects that are not LLMSettings.
    """

    with pytest.raises(TypeError):
        OllamaProvider(
            settings=MagicMock()
        )


def test_ollama_provider_rejects_non_ollama_provider():
    """
    OllamaProvider must only accept provider='ollama'.
    """

    settings = LLMSettings(
        provider="openai",
    )

    with pytest.raises(ValueError):
        OllamaProvider(settings)


def test_provider_name():
    """
    provider_name should always identify Ollama.
    """

    settings = LLMSettings(
        provider="ollama",
    )

    provider = OllamaProvider(settings)

    assert provider.provider_name == "ollama"


def test_model_name_returns_configured_model():
    """
    model_name should expose the configured Ollama model.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_model="qwen3:8b",
    )

    provider = OllamaProvider(settings)

    assert provider.model_name == "qwen3:8b"


def test_model_name_returns_custom_model():
    """
    OllamaProvider should support models other than the default.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_model="llama3.2:3b",
    )

    provider = OllamaProvider(settings)

    assert provider.model_name == "llama3.2:3b"


@patch(
    "app.llm.providers.ollama.OllamaChatCompletionClient"
)
def test_create_model_client_uses_configured_values(
    mock_client,
):
    """
    create_model_client() should construct the AutoGen Ollama client
    using the configured settings.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_host="http://127.0.0.1:11434",
        ollama_model="qwen3:8b",
        temperature=0.2,
        max_tokens=4096,
        timeout=180.0,
    )

    expected_client = MagicMock()
    mock_client.return_value = expected_client

    provider = OllamaProvider(settings)

    client = provider.create_model_client()

    assert client is expected_client

    mock_client.assert_called_once_with(
        model="qwen3:8b",
        host="http://127.0.0.1:11434",
        options={
            "temperature": 0.2,
            "num_predict": 4096,
        },
        timeout=180.0,
    )


@patch(
    "app.llm.providers.ollama.OllamaChatCompletionClient"
)
def test_create_model_client_uses_default_settings(
    mock_client,
):
    """
    create_model_client() should correctly use LLMSettings defaults.
    """

    settings = LLMSettings(
        provider="ollama",
    )

    provider = OllamaProvider(settings)

    provider.create_model_client()

    mock_client.assert_called_once_with(
        model="qwen3:8b",
        host="http://127.0.0.1:11434",
        options={
            "temperature": 0.0,
            "num_predict": 2048,
        },
        timeout=120.0,
    )


@patch(
    "app.llm.providers.ollama.OllamaChatCompletionClient"
)
def test_create_model_client_does_not_modify_settings(
    mock_client,
):
    """
    Creating the model client must not mutate provider settings.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_model="qwen3:8b",
        temperature=0.5,
        max_tokens=1024,
        timeout=90.0,
    )

    provider = OllamaProvider(settings)

    provider.create_model_client()

    assert settings.ollama_model == "qwen3:8b"
    assert settings.temperature == 0.5
    assert settings.max_tokens == 1024
    assert settings.timeout == 90.0


@patch(
    "app.llm.providers.ollama.OllamaChatCompletionClient"
)
def test_create_model_client_can_be_called_multiple_times(
    mock_client,
):
    """
    Each call should create a model client from the current configuration.

    Provider construction itself remains side-effect free.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_model="qwen3:8b",
    )

    first_client = MagicMock(name="first_client")
    second_client = MagicMock(name="second_client")

    mock_client.side_effect = [
        first_client,
        second_client,
    ]

    provider = OllamaProvider(settings)

    first = provider.create_model_client()
    second = provider.create_model_client()

    assert first is first_client
    assert second is second_client

    assert mock_client.call_count == 2


def test_provider_name_is_read_only():
    """
    provider_name should remain a read-only property.
    """

    settings = LLMSettings(
        provider="ollama",
    )

    provider = OllamaProvider(settings)

    with pytest.raises(AttributeError):
        provider.provider_name = "another-provider"


def test_model_name_is_read_only():
    """
    model_name should remain a read-only property.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_model="qwen3:8b",
    )

    provider = OllamaProvider(settings)

    with pytest.raises(AttributeError):
        provider.model_name = "another-model"