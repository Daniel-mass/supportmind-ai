"""
Unit tests for LLM configuration.
"""

import pytest
from pydantic import ValidationError

from app.llm.config import (
    LLMSettings,
    get_llm_settings,
)


def test_default_settings():
    """
    Default configuration should target the local Ollama setup.
    """

    settings = LLMSettings()

    assert settings.provider == "ollama"
    assert settings.ollama_host == (
        "http://127.0.0.1:11434"
    )
    assert settings.ollama_model == "qwen3:8b"
    assert settings.temperature == 0.0
    assert settings.max_tokens == 2048
    assert settings.timeout == 120.0


def test_custom_settings():
    """
    Explicit configuration values should be accepted.
    """

    settings = LLMSettings(
        provider="ollama",
        ollama_host="http://localhost:11434/",
        ollama_model="qwen3:8b",
        temperature=0.2,
        max_tokens=4096,
        timeout=180,
    )

    assert settings.provider == "ollama"
    assert settings.ollama_host == (
        "http://localhost:11434"
    )
    assert settings.ollama_model == "qwen3:8b"
    assert settings.temperature == 0.2
    assert settings.max_tokens == 4096
    assert settings.timeout == 180.0


def test_provider_name_is_normalized():
    """
    Provider names should be normalized to lowercase.
    """

    settings = LLMSettings(
        provider="  OLLAMA  ",
    )

    assert settings.provider == "ollama"


def test_ollama_host_trailing_slash_is_removed():
    """
    Ollama host normalization should remove trailing slashes.
    """

    settings = LLMSettings(
        ollama_host="http://localhost:11434///",
    )

    assert settings.ollama_host == (
        "http://localhost:11434"
    )


def test_empty_provider_is_rejected():
    """
    An empty provider must not be accepted.
    """

    with pytest.raises(
        ValidationError,
        match="LLM provider must not be empty",
    ):
        LLMSettings(provider="   ")


def test_empty_ollama_host_is_rejected():
    """
    An empty Ollama host must not be accepted.
    """

    with pytest.raises(
        ValidationError,
        match="Ollama host must not be empty",
    ):
        LLMSettings(ollama_host="   ")


def test_empty_ollama_model_is_rejected():
    """
    An empty Ollama model must not be accepted.
    """

    with pytest.raises(
        ValidationError,
        match="Ollama model must not be empty",
    ):
        LLMSettings(ollama_model="   ")


def test_temperature_cannot_be_negative():
    """
    Temperature must be zero or greater.
    """

    with pytest.raises(ValidationError):
        LLMSettings(temperature=-0.1)


def test_temperature_cannot_exceed_two():
    """
    Temperature must not exceed the supported upper boundary.
    """

    with pytest.raises(ValidationError):
        LLMSettings(temperature=2.1)


def test_max_tokens_must_be_positive():
    """
    Maximum output tokens must be greater than zero.
    """

    with pytest.raises(ValidationError):
        LLMSettings(max_tokens=0)


def test_timeout_must_be_positive():
    """
    Timeout must be greater than zero.
    """

    with pytest.raises(ValidationError):
        LLMSettings(timeout=0)


def test_unknown_configuration_fields_are_rejected():
    """
    Unexpected configuration fields should fail fast.
    """

    with pytest.raises(ValidationError):
        LLMSettings(
            unsupported_setting="unexpected"
        )


def test_get_llm_settings_returns_settings():
    """
    get_llm_settings() should return a validated LLMSettings object.
    """

    settings = get_llm_settings()

    assert isinstance(settings, LLMSettings)


def test_get_llm_settings_is_cached():
    """
    The settings factory should return the same cached object.
    """

    first = get_llm_settings()
    second = get_llm_settings()

    assert first is second