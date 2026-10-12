"""
Production-grade tests for VectorPipeline.

Test Strategy
--------------
1. Unit tests use mocked BaseVectorStore implementations.
2. VectorStoreRegistry is used to resolve providers.
3. Tests validate orchestration and delegation only.
4. Provider-specific Qdrant behavior is tested separately
   in test_qdrant_store.py.

Coverage
--------
- Provider resolution
- Collection creation
- Collection deletion
- Collection existence
- Single upsert
- Batch upsert
- Retrieval
- Single deletion
- Batch deletion
- Similarity search
- Metadata filtering
- Vector count
- Provider health
- Input validation
- Error propagation
- Empty batch handling
- Provider isolation

SupportMind AI
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from app.embeddings.embedding_schema import EmbeddedNode
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.exceptions import (
    InvalidVectorDimensionError,
    InvalidVectorFilterError,
    UnsupportedVectorOperationError,
    VectorCollectionNotFoundError,
    VectorDeletionError,
    VectorInsertionError,
    VectorSearchError,
    VectorStoreConnectionError,
)
from app.vector_store.vector_filters import (
    FilterCondition,
    FilterOperator,
    VectorFilter,
)
from app.vector_store.vector_pipeline import VectorPipeline
from app.vector_store.vector_registry import (
    VectorStoreRegistry,
)
from app.vector_store.vector_search_result import (
    VectorSearchResult,
)


# ==============================================================
# Test Configuration
# ==============================================================

QDRANT_PROVIDER = "qdrant"

MILVUS_PROVIDER = "milvus"

TEST_COLLECTION = "supportmind_test_vectors"

VECTOR_DIMENSION = 3


# ==============================================================
# Test Data Helpers
# ==============================================================


def make_node(
    node_id: str = "node_1",
    document_id: str = "doc_1",
    text: str = "Refund policy information.",
    embedding: list[float] | None = None,
    metadata: dict | None = None,
) -> EmbeddedNode:
    """
    Create a deterministic EmbeddedNode for testing.
    """

    vector = (
        embedding
        if embedding is not None
        else [
            1.0,
            0.0,
            0.0,
        ]
    )

    return EmbeddedNode(
        node_id=node_id,
        document_id=document_id,
        text=text,
        embedding=vector,
        embedding_model="test-model",
        embedding_dimension=len(vector),
        metadata=metadata or {},
    )


def make_result(
    node_id: str = "node_1",
    document_id: str = "doc_1",
    text: str = "Refund policy information.",
    score: float = 0.95,
    metadata: dict | None = None,
    provider: str = QDRANT_PROVIDER,
) -> VectorSearchResult:
    """
    Create a deterministic VectorSearchResult.
    """

    return VectorSearchResult(
        id=node_id,
        node_id=node_id,
        document_id=document_id,
        text=text,
        score=score,
        metadata=metadata or {},
        embedding_model="test-model",
        provider=provider,
        collection_name=TEST_COLLECTION,
    )


# ==============================================================
# Mock Provider Factory
# ==============================================================


def make_mock_provider(
    provider_name: str = QDRANT_PROVIDER,
) -> MagicMock:
    """
    Create a mocked vector store provider.

    The mock follows the BaseVectorStore contract.
    """

    provider = MagicMock(
        spec=BaseVectorStore,
    )

    type(provider).provider = property(
        lambda _: provider_name
    )

    provider.collection_exists.return_value = True

    provider.health_check.return_value = True

    provider.count.return_value = 0

    provider.get.return_value = []

    provider.search.return_value = []

    return provider


# ==============================================================
# Unit Tests
# ==============================================================


class TestVectorPipeline(unittest.TestCase):
    """
    Unit tests for VectorPipeline.

    Provider implementations are mocked.

    These tests verify that VectorPipeline correctly
    orchestrates operations through VectorStoreRegistry.
    """

    def setUp(self):

        self.registry = VectorStoreRegistry()

        self.qdrant_provider = make_mock_provider(
            QDRANT_PROVIDER
        )

        self.registry.register(
            self.qdrant_provider
        )

        self.pipeline = VectorPipeline(
            registry=self.registry,
        )

    # ==========================================================
    # Provider Resolution
    # ==========================================================

    def test_provider_resolution(self):

        provider = self.pipeline.get_provider(
            QDRANT_PROVIDER
        )

        self.assertIs(
            provider,
            self.qdrant_provider,
        )

    def test_provider_resolution_is_case_insensitive(self):

        provider = self.pipeline.get_provider(
            "QDRANT"
        )

        self.assertIs(
            provider,
            self.qdrant_provider,
        )

    def test_unregistered_provider_raises_error(self):

        with self.assertRaises(
            UnsupportedVectorOperationError
        ):

            self.pipeline.get_provider(
                "pinecone"
            )

    # ==========================================================
    # Collection Management
    # ==========================================================

    def test_create_collection_delegates_to_provider(self):

        self.pipeline.create_collection(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            dimension=VECTOR_DIMENSION,
        )

        self.qdrant_provider.create_collection.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            dimension=VECTOR_DIMENSION,
        )

    def test_delete_collection_delegates_to_provider(self):

        self.pipeline.delete_collection(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
        )

        self.qdrant_provider.delete_collection.assert_called_once_with(
            collection_name=TEST_COLLECTION,
        )

    def test_collection_exists_delegates_to_provider(self):

        self.qdrant_provider.collection_exists.return_value = (
            True
        )

        result = self.pipeline.collection_exists(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
        )

        self.assertTrue(result)

        self.qdrant_provider.collection_exists.assert_called_once_with(
            TEST_COLLECTION,
        )

    # ==========================================================
    # Upsert
    # ==========================================================

    def test_upsert_delegates_to_provider(self):

        node = make_node()

        self.pipeline.upsert(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            node=node,
        )

        self.qdrant_provider.upsert.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            node=node,
        )

    def test_upsert_batch_delegates_to_provider(self):

        nodes = [
            make_node(
                node_id="node_1",
            ),
            make_node(
                node_id="node_2",
                embedding=[
                    0.0,
                    1.0,
                    0.0,
                ],
            ),
        ]

        self.pipeline.upsert_batch(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

        self.qdrant_provider.upsert_batch.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

    def test_upsert_batch_empty_list(self):

        self.pipeline.upsert_batch(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            nodes=[],
        )

        self.qdrant_provider.upsert_batch.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            nodes=[],
        )

    # ==========================================================
    # Retrieval
    # ==========================================================

    def test_get_delegates_to_provider(self):

        expected_results = [
            make_result(
                node_id="node_1",
            )
        ]

        self.qdrant_provider.get.return_value = (
            expected_results
        )

        results = self.pipeline.get(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            ids=[
                "node_1",
            ],
        )

        self.assertEqual(
            results,
            expected_results,
        )

        self.qdrant_provider.get.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            ids=[
                "node_1",
            ],
        )

    def test_get_empty_ids(self):

        self.qdrant_provider.get.return_value = []

        results = self.pipeline.get(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            ids=[],
        )

        self.assertEqual(
            results,
            [],
        )

        self.qdrant_provider.get.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            ids=[],
        )

    # ==========================================================
    # Deletion
    # ==========================================================

    def test_delete_delegates_to_provider(self):

        self.pipeline.delete(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            node_id="node_1",
        )

        self.qdrant_provider.delete.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            node_id="node_1",
        )

    def test_delete_batch_delegates_to_provider(self):

        ids = [
            "node_1",
            "node_2",
            "node_3",
        ]

        self.pipeline.delete_batch(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            ids=ids,
        )

        self.qdrant_provider.delete_batch.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            ids=ids,
        )

    def test_delete_batch_empty_ids(self):

        self.pipeline.delete_batch(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            ids=[],
        )

        self.qdrant_provider.delete_batch.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            ids=[],
        )

    # ==========================================================
    # Similarity Search
    # ==========================================================

    def test_search_delegates_to_provider(self):

        expected_results = [
            make_result(
                node_id="node_1",
                score=0.98,
            ),
            make_result(
                node_id="node_2",
                score=0.87,
            ),
        ]

        self.qdrant_provider.search.return_value = (
            expected_results
        )

        query_vector = [
            1.0,
            0.0,
            0.0,
        ]

        results = self.pipeline.search(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            query_vector=query_vector,
            top_k=2,
        )

        self.assertEqual(
            results,
            expected_results,
        )

        self.qdrant_provider.search.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            query_vector=query_vector,
            top_k=2,
            filters=None,
        )

    def test_search_with_filter(self):

        vector_filter = VectorFilter(
            conditions=[
                FilterCondition(
                    field="category",
                    operator=FilterOperator.EQUAL,
                    value="billing",
                )
            ]
        )

        self.qdrant_provider.search.return_value = []

        self.pipeline.search(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=5,
            filters=vector_filter,
        )

        self.qdrant_provider.search.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=5,
            filters=vector_filter,
        )

    # ==========================================================
    # Count
    # ==========================================================

    def test_count_delegates_to_provider(self):

        self.qdrant_provider.count.return_value = 42

        result = self.pipeline.count(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
        )

        self.assertEqual(
            result,
            42,
        )

        self.qdrant_provider.count.assert_called_once_with(
            TEST_COLLECTION,
        )

    # ==========================================================
    # Health
    # ==========================================================

    def test_health_check_delegates_to_provider(self):

        self.qdrant_provider.health_check.return_value = (
            True
        )

        result = self.pipeline.health_check(
            provider_name=QDRANT_PROVIDER,
        )

        self.assertTrue(result)

        self.qdrant_provider.health_check.assert_called_once()

    def test_health_check_failure(self):

        self.qdrant_provider.health_check.return_value = (
            False
        )

        result = self.pipeline.health_check(
            provider_name=QDRANT_PROVIDER,
        )

        self.assertFalse(result)

    # ==========================================================
    # Provider Isolation
    # ==========================================================

    def test_provider_isolation(self):

        milvus_provider = make_mock_provider(
            MILVUS_PROVIDER
        )

        self.registry.register(
            milvus_provider
        )

        qdrant_node = make_node()

        self.pipeline.upsert(
            provider_name=QDRANT_PROVIDER,
            collection_name=TEST_COLLECTION,
            node=qdrant_node,
        )

        self.assertTrue(
            self.qdrant_provider.upsert.called
        )

        self.assertFalse(
            milvus_provider.upsert.called
        )

    # ==========================================================
    # Provider Errors
    # ==========================================================

    def test_collection_not_found_error_propagates(self):

        self.qdrant_provider.create_collection.side_effect = (
            VectorCollectionNotFoundError(
                "Collection not found"
            )
        )

        with self.assertRaises(
            VectorCollectionNotFoundError
        ):

            self.pipeline.create_collection(
                provider_name=QDRANT_PROVIDER,
                collection_name=TEST_COLLECTION,
                dimension=VECTOR_DIMENSION,
            )

    def test_insertion_error_propagates(self):

        self.qdrant_provider.upsert.side_effect = (
            VectorInsertionError(
                "Insertion failed"
            )
        )

        with self.assertRaises(
            VectorInsertionError
        ):

            self.pipeline.upsert(
                provider_name=QDRANT_PROVIDER,
                collection_name=TEST_COLLECTION,
                node=make_node(),
            )

    def test_dimension_error_propagates(self):

        self.qdrant_provider.upsert.side_effect = (
            InvalidVectorDimensionError(
                "Invalid vector dimension"
            )
        )

        with self.assertRaises(
            InvalidVectorDimensionError
        ):

            self.pipeline.upsert(
                provider_name=QDRANT_PROVIDER,
                collection_name=TEST_COLLECTION,
                node=make_node(),
            )

    def test_search_error_propagates(self):

        self.qdrant_provider.search.side_effect = (
            VectorSearchError(
                "Search failed"
            )
        )

        with self.assertRaises(
            VectorSearchError
        ):

            self.pipeline.search(
                provider_name=QDRANT_PROVIDER,
                collection_name=TEST_COLLECTION,
                query_vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
            )

    def test_deletion_error_propagates(self):

        self.qdrant_provider.delete.side_effect = (
            VectorDeletionError(
                "Deletion failed"
            )
        )

        with self.assertRaises(
            VectorDeletionError
        ):

            self.pipeline.delete(
                provider_name=QDRANT_PROVIDER,
                collection_name=TEST_COLLECTION,
                node_id="node_1",
            )

    def test_connection_error_propagates(self):

        self.qdrant_provider.health_check.side_effect = (
            VectorStoreConnectionError(
                "Connection failed"
            )
        )

        with self.assertRaises(
            VectorStoreConnectionError
        ):

            self.pipeline.health_check(
                provider_name=QDRANT_PROVIDER,
            )

    # ==========================================================
    # Filter Error
    # ==========================================================

    def test_invalid_filter_error_propagates(self):

        vector_filter = VectorFilter(
            conditions=[
                FilterCondition(
                    field="category",
                    operator=FilterOperator.EQUAL,
                    value="billing",
                )
            ]
        )

        self.qdrant_provider.search.side_effect = (
            InvalidVectorFilterError(
                "Invalid filter"
            )
        )

        with self.assertRaises(
            InvalidVectorFilterError
        ):

            self.pipeline.search(
                provider_name=QDRANT_PROVIDER,
                collection_name=TEST_COLLECTION,
                query_vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
                filters=vector_filter,
            )


# ==============================================================
# Registry Integration Tests
# ==============================================================


class TestVectorPipelineRegistryIntegration(
    unittest.TestCase
):
    """
    Validate that VectorPipeline correctly works with
    multiple registered providers.
    """

    def setUp(self):

        self.registry = VectorStoreRegistry()

        self.qdrant_provider = make_mock_provider(
            QDRANT_PROVIDER
        )

        self.milvus_provider = make_mock_provider(
            MILVUS_PROVIDER
        )

        self.registry.register(
            self.qdrant_provider
        )

        self.registry.register(
            self.milvus_provider
        )

        self.pipeline = VectorPipeline(
            registry=self.registry,
        )

    def test_qdrant_provider_selected(self):

        self.pipeline.count(
            provider_name="qdrant",
            collection_name=TEST_COLLECTION,
        )

        self.qdrant_provider.count.assert_called_once_with(
            TEST_COLLECTION,
        )

        self.milvus_provider.count.assert_not_called()

    def test_milvus_provider_selected(self):

        self.pipeline.count(
            provider_name="milvus",
            collection_name=TEST_COLLECTION,
        )

        self.milvus_provider.count.assert_called_once_with(
            TEST_COLLECTION,
        )

        self.qdrant_provider.count.assert_not_called()

    def test_provider_names_are_isolated(self):

        self.qdrant_provider.count.return_value = 10

        self.milvus_provider.count.return_value = 20

        qdrant_count = self.pipeline.count(
            provider_name="qdrant",
            collection_name=TEST_COLLECTION,
        )

        milvus_count = self.pipeline.count(
            provider_name="milvus",
            collection_name=TEST_COLLECTION,
        )

        self.assertEqual(
            qdrant_count,
            10,
        )

        self.assertEqual(
            milvus_count,
            20,
        )


# ==============================================================
# Test Runner
# ==============================================================

if __name__ == "__main__":

    unittest.main()
