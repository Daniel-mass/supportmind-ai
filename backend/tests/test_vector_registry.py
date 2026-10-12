"""
Unit tests for VectorRegistry.

SupportMind AI

Test Strategy
-------------
1. Validate provider registration.
2. Validate provider retrieval.
3. Validate duplicate provider protection.
4. Validate provider removal.
5. Validate missing provider handling.
6. Validate registry listing.
7. Validate clear/reset behavior.
8. Validate provider isolation.
9. Validate invalid provider names.
10. Validate registry lifecycle.
"""

from __future__ import annotations

import unittest

from app.vector_store.exceptions import (
    UnsupportedVectorOperationError,
)
from app.vector_store.vector_registry import VectorStoreRegistry


# ==============================================================
# Mock Vector Store
# ==============================================================


class MockVectorStore:
    """
    Minimal fake vector store used for registry testing.

    The registry should only care that the object can be
    registered and retrieved. No actual Qdrant operations
    are required.
    """

    def __init__(
        self,
        provider_name: str = "mock",
    ) -> None:

        self._provider_name = provider_name

    @property
    def provider(self) -> str:

        return self._provider_name


class AnotherMockVectorStore:
    """
    Second fake provider used to verify provider isolation.
    """

    @property
    def provider(self) -> str:

        return "another_mock"


# ==============================================================
# Test Suite
# ==============================================================


class TestVectorRegistry(unittest.TestCase):
    """
    Unit tests for VectorRegistry.
    """

    def setUp(self) -> None:
        """
        Create a fresh registry before every test.

        This prevents tests from sharing global registry state.
        """

        self.registry = VectorStoreRegistry()

        self.mock_store = MockVectorStore(
            provider_name="mock",
        )

        self.another_store = AnotherMockVectorStore()

    # ==========================================================
    # Registration
    # ==========================================================

    def test_register_provider(self) -> None:
        """
        Provider can be successfully registered.
        """

        self.registry.register(
            self.mock_store,
        )

        result = self.registry.get(
            "mock",
        )

        self.assertIs(
            result,
            self.mock_store,
        )

    def test_register_multiple_providers(self) -> None:
        """
        Multiple different providers can coexist.
        """

        self.registry.register(
            self.mock_store,
        )

        self.registry.register(
            self.another_store,
        )

        self.assertIs(
            self.registry.get("mock"),
            self.mock_store,
        )

        self.assertIs(
            self.registry.get("another_mock"),
            self.another_store,
        )

    # ==========================================================
    # Duplicate Registration
    # ==========================================================

    def test_duplicate_provider_registration_fails(self) -> None:
        """
        Registering the same provider twice must fail.

        Silent replacement is dangerous because it can
        accidentally overwrite a production vector backend.
        """

        self.registry.register(
            self.mock_store,
        )

        with self.assertRaises(
            UnsupportedVectorOperationError,
        ):

            self.registry.register(
                MockVectorStore(
                    provider_name="mock",
                ),
            )

    # ==========================================================
    # Retrieval
    # ==========================================================

    def test_get_registered_provider(self) -> None:
        """
        Registered provider can be retrieved by name.
        """

        self.registry.register(
            self.mock_store,
        )

        provider = self.registry.get(
            "mock",
        )

        self.assertEqual(
            provider.provider,
            "mock",
        )

    def test_get_unregistered_provider_fails(self) -> None:
        """
        Retrieving an unknown provider must fail explicitly.
        """

        with self.assertRaises(
            UnsupportedVectorOperationError,
        ):

            self.registry.get(
                "qdrant",
            )

    # ==========================================================
    # Provider Listing
    # ==========================================================

    def test_list_providers_empty_registry(self) -> None:
        """
        Empty registry returns an empty provider list.
        """

        providers = self.registry.list_providers()

        self.assertEqual(
            providers,
            [],
        )

    def test_list_providers(self) -> None:
        """
        Registry returns all registered provider names.
        """

        self.registry.register(
            self.mock_store,
        )

        self.registry.register(
            self.another_store,
        )

        providers = self.registry.list_providers()

        self.assertEqual(
            set(providers),
            {
                "mock",
                "another_mock",
            },
        )

    # ==========================================================
    # Provider Existence
    # ==========================================================

    def test_has_provider_returns_true(self) -> None:
        """
        has_provider returns True for registered providers.
        """

        self.registry.register(
            self.mock_store,
        )

        self.assertTrue(
            self.registry.has_provider(
                "mock",
            ),
        )

    def test_has_provider_returns_false(self) -> None:
        """
        has_provider returns False for unknown providers.
        """

        self.assertFalse(
            self.registry.has_provider(
                "qdrant",
            ),
        )

    # ==========================================================
    # Provider Removal
    # ==========================================================

    def test_unregister_provider(self) -> None:
        """
        Registered provider can be removed.
        """

        self.registry.register(
            self.mock_store,
        )

        self.registry.unregister(
            "mock",
        )

        self.assertFalse(
            self.registry.has_provider(
                "mock",
            ),
        )

    def test_unregister_missing_provider_fails(self) -> None:
        """
        Removing an unknown provider must fail explicitly.
        """

        with self.assertRaises(
            UnsupportedVectorOperationError,
        ):

            self.registry.unregister(
                "qdrant",
            )

    def test_unregister_does_not_affect_other_providers(
        self,
    ) -> None:
        """
        Removing one provider must not remove others.
        """

        self.registry.register(
            self.mock_store,
        )

        self.registry.register(
            self.another_store,
        )

        self.registry.unregister(
            "mock",
        )

        self.assertFalse(
            self.registry.has_provider(
                "mock",
            ),
        )

        self.assertTrue(
            self.registry.has_provider(
                "another_mock",
            ),
        )

    # ==========================================================
    # Clear Registry
    # ==========================================================

    def test_clear_registry(self) -> None:
        """
        clear removes all registered providers.
        """

        self.registry.register(
            self.mock_store,
        )

        self.registry.register(
            self.another_store,
        )

        self.registry.clear()

        self.assertEqual(
            self.registry.list_providers(),
            [],
        )

    def test_clear_empty_registry(self) -> None:
        """
        Clearing an already empty registry is safe.
        """

        self.registry.clear()

        self.assertEqual(
            self.registry.list_providers(),
            [],
        )

    # ==========================================================
    # Provider Name Validation
    # ==========================================================

    def test_register_empty_provider_name_fails(self) -> None:
        """
        Empty provider names must not be accepted.
        """

        invalid_store = MockVectorStore(
            provider_name="",
        )

        with self.assertRaises(
            ValueError,
        ):

            self.registry.register(
                invalid_store,
            )

    def test_register_whitespace_provider_name_fails(
        self,
    ) -> None:
        """
        Whitespace-only provider names must not be accepted.
        """

        invalid_store = MockVectorStore(
            provider_name="   ",
        )

        with self.assertRaises(
            ValueError,
        ):

            self.registry.register(
                invalid_store,
            )

    def test_get_empty_provider_name_fails(self) -> None:
        """
        Empty provider lookup must fail.
        """

        with self.assertRaises(
            ValueError,
        ):

            self.registry.get(
                "",
            )

    def test_unregister_empty_provider_name_fails(
        self,
    ) -> None:
        """
        Empty provider removal must fail.
        """

        with self.assertRaises(
            ValueError,
        ):

            self.registry.unregister(
                "",
            )

    # ==========================================================
    # Provider Isolation
    # ==========================================================

    def test_provider_instances_are_isolated(self) -> None:
        """
        Registry must preserve the exact provider instance
        registered under each provider name.
        """

        first_store = MockVectorStore(
            provider_name="provider_a",
        )

        second_store = MockVectorStore(
            provider_name="provider_b",
        )

        self.registry.register(
            first_store,
        )

        self.registry.register(
            second_store,
        )

        self.assertIs(
            self.registry.get("provider_a"),
            first_store,
        )

        self.assertIs(
            self.registry.get("provider_b"),
            second_store,
        )

        self.assertIsNot(
            self.registry.get("provider_a"),
            second_store,
        )

    # ==========================================================
    # Registry Lifecycle
    # ==========================================================

    def test_registry_can_be_reused_after_clear(
        self,
    ) -> None:
        """
        Registry can be cleared and reused safely.
        """

        self.registry.register(
            self.mock_store,
        )

        self.registry.clear()

        new_store = MockVectorStore(
            provider_name="qdrant",
        )

        self.registry.register(
            new_store,
        )

        self.assertEqual(
            self.registry.list_providers(),
            [
                "qdrant",
            ],
        )

        self.assertIs(
            self.registry.get(
                "qdrant",
            ),
            new_store,
        )


# ==============================================================
# Test Runner
# ==============================================================

if __name__ == "__main__":

    unittest.main()