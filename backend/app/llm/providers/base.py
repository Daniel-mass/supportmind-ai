"""
Base LLM provider contract for SupportMind AI.

This module defines the interface that every SupportMind LLM provider
must implement.

Responsibilities:
- Define the provider abstraction.
- Expose provider/model identity.
- Define creation of an AutoGen-compatible model client.

This module does NOT:
- Connect to a model provider.
- Perform network requests.
- Execute inference.
- Know about Ollama.
- Create AutoGen agents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Abstract base class for SupportMind LLM providers.

    Providers are responsible for creating the model client used by
    the Agentic Layer.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """
        Return the provider identifier.

        Example:
            "ollama"
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the configured model identifier.

        Example:
            "qwen3:8b"
        """

        raise NotImplementedError

    @abstractmethod
    def create_model_client(self) -> Any:
        """
        Create and return an AutoGen-compatible model client.

        The concrete provider is responsible for constructing the
        appropriate AutoGen model client.

        Returns:
            AutoGen-compatible model client.
        """

        raise NotImplementedError