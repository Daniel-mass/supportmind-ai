"""
vector_registry.py

Provider registry for SupportMind AI vector stores.

Responsibilities
----------------
1. Register vector store provider instances.
2. Resolve providers by name.
3. Prevent duplicate registrations.
4. Support provider isolation.
5. Keep the application provider-agnostic.

SupportMind AI
"""

from __future__ import annotations

from typing import Any

from app.vector_store.exceptions import (
    UnsupportedVectorOperationError,
)


class VectorStoreRegistry:
    """
    Registry for vector store provider instances.

    The registry stores already-created vector store objects.

    Example
    -------
    registry = VectorStoreRegistry()

    registry.register(qdrant_store)

    store = registry.get("qdrant")
    """

    def __init__(self) -> None:
        """
        Initialize an empty vector store registry.
        """

        self._providers: dict[str, Any] = {}

    # ==========================================================
    # Provider Registration
    # ==========================================================

    def register(
        self,
        provider: Any,
    ) -> None:
        """
        Register a vector store provider instance.

        The provider must expose a non-empty `provider`
        attribute that uniquely identifies it.

        Raises
        ------
        ValueError
            If provider is invalid or provider name is empty.

        UnsupportedVectorOperationError
            If provider is already registered.
        """

        if provider is None:
            raise ValueError(
                "Vector provider cannot be None."
            )

        if not hasattr(provider, "provider"):
            raise ValueError(
                "Vector provider must expose a "
                "'provider' attribute."
            )

        provider_name = provider.provider

        if not isinstance(provider_name, str):
            raise ValueError(
                "Vector provider name must be a string."
            )

        normalized_name = provider_name.strip().lower()

        if not normalized_name:
            raise ValueError(
                "Vector provider name cannot be empty."
            )

        if normalized_name in self._providers:
            raise UnsupportedVectorOperationError(
                f"Vector provider "
                f"'{normalized_name}' "
                f"is already registered."
            )

        self._providers[
            normalized_name
        ] = provider

    # ==========================================================
    # Provider Retrieval
    # ==========================================================

    def get(
        self,
        name: str,
    ) -> Any:
        """
        Retrieve a registered provider instance.

        Raises
        ------
        ValueError
            If provider name is invalid.

        UnsupportedVectorOperationError
            If provider is not registered.
        """

        normalized_name = self._normalize_name(
            name
        )

        if normalized_name not in self._providers:
            raise UnsupportedVectorOperationError(
                f"Vector provider "
                f"'{normalized_name}' "
                f"is not registered."
            )

        return self._providers[
            normalized_name
        ]

    # ==========================================================
    # Provider Existence
    # ==========================================================

    def has_provider(
        self,
        name: str,
    ) -> bool:
        """
        Return True if a provider is registered.

        Invalid names raise ValueError.
        """

        normalized_name = self._normalize_name(
            name
        )

        return (
            normalized_name
            in self._providers
        )

    # ==========================================================
    # Provider Removal
    # ==========================================================

    def unregister(
        self,
        name: str,
    ) -> None:
        """
        Remove a registered provider.

        Raises
        ------
        ValueError
            If provider name is invalid.

        UnsupportedVectorOperationError
            If provider is not registered.
        """

        normalized_name = self._normalize_name(
            name
        )

        if normalized_name not in self._providers:
            raise UnsupportedVectorOperationError(
                f"Vector provider "
                f"'{normalized_name}' "
                f"is not registered."
            )

        del self._providers[
            normalized_name
        ]

    # ==========================================================
    # Provider Listing
    # ==========================================================

    def list_providers(
        self,
    ) -> list[str]:
        """
        Return all registered provider names.

        Results are sorted for deterministic behavior.
        """

        return sorted(
            self._providers.keys()
        )

    # ==========================================================
    # Registry Management
    # ==========================================================

    def clear(
        self,
    ) -> None:
        """
        Remove all registered providers.
        """

        self._providers.clear()

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:
        """
        Normalize and validate provider names.
        """

        if not isinstance(
            name,
            str,
        ):
            raise ValueError(
                "Vector provider name must be a string."
            )

        normalized_name = name.strip().lower()

        if not normalized_name:
            raise ValueError(
                "Vector provider name cannot be empty."
            )

        return normalized_name


# ==============================================================
# Global Registry
# ==============================================================

vector_registry = VectorStoreRegistry()
"""
Application-level default vector store registry.
"""