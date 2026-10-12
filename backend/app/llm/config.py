"""
LLM configuration for SupportMind AI.

This module contains validated configuration for the LLM layer.

Responsibilities:
- Define LLM configuration fields.
- Load configuration from environment variables.
- Validate configuration values.
- Provide safe defaults for local development.

This module does NOT:
- Create an LLM client.
- Connect to Ollama.
- Call AutoGen.
- Execute model inference.
- Perform network requests.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMSettings(BaseModel):
    """
    Validated configuration for the SupportMind LLM layer.

    Provider-specific implementation is handled by the provider modules.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    provider: str = Field(
        default="ollama",
        min_length=1,
    )

    ollama_host: str = Field(
        default="http://127.0.0.1:11434",
        min_length=1,
    )

    ollama_model: str = Field(
        default="qwen3:8b",
        min_length=1,
    )

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int = Field(
        default=2048,
        gt=0,
    )

    timeout: float = Field(
        default=120.0,
        gt=0.0,
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        """
        Normalize and validate the provider name.
        """

        normalized = value.strip().lower()

        if not normalized:
            raise ValueError(
                "LLM provider must not be empty."
            )

        return normalized

    @field_validator("ollama_host")
    @classmethod
    def validate_ollama_host(cls, value: str) -> str:
        """
        Normalize the Ollama host URL.

        Trailing slashes are removed so provider implementations can
        construct endpoint URLs consistently.
        """

        normalized = value.strip().rstrip("/")

        if not normalized:
            raise ValueError(
                "Ollama host must not be empty."
            )

        return normalized

    @field_validator("ollama_model")
    @classmethod
    def validate_ollama_model(cls, value: str) -> str:
        """
        Normalize the configured Ollama model name.
        """

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "Ollama model must not be empty."
            )

        return normalized


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    """
    Return the cached default LLM settings.

    Environment-variable loading will be added at the application
    configuration boundary once the project's environment-loading
    convention is confirmed.

    Keeping this function separate from provider construction allows
    providers to consume a stable configuration object without
    performing network operations.
    """

    return LLMSettings()