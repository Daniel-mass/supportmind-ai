"""
generator.py

Provider-independent Graph-RAG generation layer.

Responsibilities
----------------
- Accept GraphRAGContext.
- Build generation prompts using prompts.py.
- Delegate generation to an injected LLM provider.
- Validate the provider output.
- Expose provider/model metadata.

Non-responsibilities
--------------------
- Retrieval
- Evidence fusion
- Graph traversal
- Groundedness validation
- Citation validation
- Provider-specific client handling
"""

from __future__ import annotations

from typing import Protocol

from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGContext,
)
from app.retrieval.graph_rag.prompts import (
    build_prompt_set,
)


# ============================================================================
# Provider protocol
# ============================================================================


class LLMProviderProtocol(Protocol):
    """
    Minimal provider interface required by GraphRAGGenerator.
    """

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate text from system/user prompts."""
        ...


# ============================================================================
# Graph-RAG Generator
# ============================================================================


class GraphRAGGenerator:
    """
    Provider-independent generator for SupportMind Graph-RAG.

    The generator does not know how retrieval or evidence collection works.
    It receives an already-fused GraphRAGContext and sends the corresponding
    prompts to the configured provider.
    """

    def __init__(
        self,
        *,
        provider: LLMProviderProtocol,
        provider_name: str,
        model_name: str,
    ) -> None:
        if provider is None:
            raise ValueError(
                "provider cannot be None."
            )

        if not hasattr(
            provider,
            "generate",
        ):
            raise TypeError(
                "provider must expose generate()."
            )

        self.provider = provider

        self.provider_name = (
            self._validate_required_string(
                provider_name,
                "provider_name",
            )
        )

        self.model_name = (
            self._validate_required_string(
                model_name,
                "model_name",
            )
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def generate(
        self,
        context: GraphRAGContext,
    ) -> str:
        """
        Generate a final answer from a fused Graph-RAG context.
        """

        if not isinstance(
            context,
            GraphRAGContext,
        ):
            raise TypeError(
                "context must be a GraphRAGContext instance."
            )

        system_prompt, user_prompt = (
            build_prompt_set(context)
        )

        try:
            answer = self.provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception:
            # Preserve the original provider exception. The provider owns
            # provider-specific error translation and diagnostics.
            raise

        if not isinstance(
            answer,
            str,
        ):
            raise TypeError(
                "LLM provider must return a string."
            )

        answer = answer.strip()

        if not answer:
            raise RuntimeError(
                "LLM provider returned an empty answer."
            )

        return answer

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_required_string(
        value: str,
        field_name: str,
    ) -> str:
        """
        Normalize and validate provider metadata.
        """

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return value