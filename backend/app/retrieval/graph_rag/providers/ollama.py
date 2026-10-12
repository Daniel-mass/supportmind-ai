"""
ollama.py

Ollama LLM provider for SupportMind Graph-RAG.

Responsibilities
----------------
- Manage the Ollama client configuration.
- Validate generation parameters.
- Send system/user prompts to Ollama.
- Extract generated text from the response.
- Provide deterministic, testable error handling.

Non-responsibilities
--------------------
- Retrieval
- Evidence fusion
- Prompt construction
- Grounding validation
- Graph traversal
- Business logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


# ============================================================================
# Client protocol
# ============================================================================


class OllamaClientProtocol(Protocol):
    """
    Minimal Ollama client interface required by OllamaProvider.

    Using a Protocol keeps unit tests independent from a live Ollama server.
    """

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Generate a chat response.
        """
        ...


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class OllamaGenerationConfig:
    """
    Generation configuration for the Ollama provider.
    """

    host: str = "http://localhost:11434"
    model_name: str = "qwen3:8b"

    temperature: float = 0.1
    max_tokens: int = 1024

    timeout: float = 120.0

    def __post_init__(self) -> None:
        if not isinstance(
            self.host,
            str,
        ) or not self.host.strip():
            raise ValueError(
                "host must be a non-empty string."
            )

        if not isinstance(
            self.model_name,
            str,
        ) or not self.model_name.strip():
            raise ValueError(
                "model_name must be a non-empty string."
            )

        if isinstance(
            self.temperature,
            bool,
        ) or not isinstance(
            self.temperature,
            (int, float),
        ):
            raise TypeError(
                "temperature must be numeric."
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature cannot be negative."
            )

        if (
            isinstance(
                self.max_tokens,
                bool,
            )
            or not isinstance(
                self.max_tokens,
                int,
            )
        ):
            raise TypeError(
                "max_tokens must be an integer."
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        if isinstance(
            self.timeout,
            bool,
        ) or not isinstance(
            self.timeout,
            (int, float),
        ):
            raise TypeError(
                "timeout must be numeric."
            )

        if self.timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )


# ============================================================================
# Ollama provider
# ============================================================================


class OllamaProvider:
    """
    Provider wrapper around the Ollama Python client.

    The real Ollama client is created lazily when one is not injected.
    """

    def __init__(
        self,
        config: OllamaGenerationConfig | None = None,
        client: OllamaClientProtocol | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else OllamaGenerationConfig()
        )

        self._client = (
            client
            if client is not None
            else self._create_client(
                self.config.host
            )
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """
        Generate a response from Ollama.

        Parameters
        ----------
        system_prompt:
            System-level generation instructions.

        user_prompt:
            User question plus generation context.

        Returns
        -------
        str
            Generated assistant text.
        """

        system_prompt = self._validate_prompt(
            system_prompt,
            "system_prompt",
        )

        user_prompt = self._validate_prompt(
            user_prompt,
            "user_prompt",
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        options = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
        }

        try:
            response = self._client.chat(
                model=self.config.model_name,
                messages=messages,
                options=options,
                think=False,
            )
        except Exception as exc:
            import traceback

            print("\n" + "=" * 80)
            print("OLLAMA REAL ERROR")
            print("=" * 80)
            print(f"Exception type : {type(exc).__name__}")
            print(f"Exception text : {exc}")
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80 + "\n")

            raise RuntimeError(
                f"Ollama generation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        content = self._extract_content(
            response
        )

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return content

    # ========================================================================
    # Client construction
    # ========================================================================

    @staticmethod
    def _create_client(
        host: str,
    ) -> OllamaClientProtocol:
        """
        Create the real Ollama Python client.

        Import is kept inside the factory so unit tests do not require the
        package or a running Ollama service.
        """

        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError(
                "The 'ollama' Python package is not installed. "
                "Install it with: pip install ollama"
            ) from exc

        return Client(
            host=host
        )

    # ========================================================================
    # Response extraction
    # ========================================================================

    @staticmethod
    def _extract_content(
        response: Any,
    ) -> str:
        """
        Extract assistant message content from supported Ollama response forms.

        The provider accepts both mapping-like responses and object-like
        responses so tests can use lightweight fake objects.
        """

        if response is None:
            return ""

        # --------------------------------------------------------------
        # Mapping-style response
        # --------------------------------------------------------------

        if isinstance(
            response,
            dict,
        ):
            message = response.get(
                "message"
            )

            if isinstance(
                message,
                dict,
            ):
                content = message.get(
                    "content"
                )

                if isinstance(
                    content,
                    str,
                ):
                    return content.strip()

        # --------------------------------------------------------------
        # Object-style response
        # --------------------------------------------------------------

        message = getattr(
            response,
            "message",
            None,
        )

        if message is not None:
            content = getattr(
                message,
                "content",
                None,
            )

            if isinstance(
                content,
                str,
            ):
                return content.strip()

        # --------------------------------------------------------------
        # Defensive fallback
        # --------------------------------------------------------------

        content = getattr(
            response,
            "content",
            None,
        )

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        return ""

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_prompt(
        prompt: str,
        field_name: str,
    ) -> str:
        """
        Validate and normalize a prompt.
        """

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string."
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return prompt

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def close(self) -> None:
        """
        Close the underlying client when it exposes a close() method.
        """

        close_method = getattr(
            self._client,
            "close",
            None,
        )

        if callable(
            close_method
        ):
            close_method()

    def __enter__(
        self,
    ) -> "OllamaProvider":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()