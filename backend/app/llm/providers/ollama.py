"""
Ollama LLM provider for SupportMind AI.

This module adapts the SupportMind LLM provider contract to
AutoGen's Ollama model client.

Responsibilities:
- Validate Ollama-specific configuration.
- Expose provider and model identity.
- Create an AutoGen-compatible Ollama model client.

This module does NOT:
- Create AutoGen agents.
- Execute model inference directly.
- Contain agent orchestration logic.
- Perform prompt construction.
- Contain business logic.
"""

from __future__ import annotations

from typing import Any

from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.llm.config import LLMSettings
from app.llm.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    SupportMind LLM provider implementation for Ollama.

    The provider is responsible only for adapting validated
    SupportMind configuration into an AutoGen Ollama model client.
    """

    def __init__(
        self,
        settings: LLMSettings,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            settings: Validated LLM configuration.

        Raises:
            TypeError: If settings is not an LLMSettings instance.
            ValueError: If the configured provider is not Ollama.
        """

        if not isinstance(settings, LLMSettings):
            raise TypeError(
                "settings must be an instance of LLMSettings."
            )

        if settings.provider != "ollama":
            raise ValueError(
                "OllamaProvider requires provider='ollama'."
            )

        self._settings = settings

    @property
    def provider_name(self) -> str:
        """
        Return the provider identifier.
        """

        return "ollama"

    @property
    def model_name(self) -> str:
        """
        Return the configured Ollama model name.
        """

        return self._settings.ollama_model

    def create_model_client(self) -> Any:
        """
        Create an AutoGen-compatible Ollama model client.

        No model inference is performed here. The returned client is
        consumed later by the Agentic Layer.
        """

        return OllamaChatCompletionClient(
            model=self._settings.ollama_model,
            host=self._settings.ollama_host,
            options={
                "temperature": self._settings.temperature,
                "num_predict": self._settings.max_tokens,
            },
            timeout=self._settings.timeout,
        )