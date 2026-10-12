# test_qdrant_store.py
"""
Production-grade tests for QdrantStore.

Test Strategy
--------------
1. Unit Tests
   - Mock QdrantClient.
   - Fast and deterministic.
   - Validate application-level behavior.
   - Validate Qdrant API calls.
   - Validate exception translation.
   - Validate vector dimension checking.

2. Integration Tests
   - Execute against a real Qdrant instance.
   - Automatically skipped when disabled.
   - Validate real collection creation.
   - Validate vector dimensions.
   - Validate upsert and retrieval.
   - Validate filtering.
   - Validate similarity search.
   - Validate count.
   - Validate deletion.

SupportMind AI
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.embeddings.embedding_schema import EmbeddedNode
from app.vector_store.exceptions import (
    InvalidVectorDimensionError,
    UnsupportedVectorOperationError,
    VectorCollectionAlreadyExistsError,
    VectorCollectionNotFoundError,
    VectorDeletionError,
    VectorInsertionError,
    VectorSearchError,
    VectorStoreConnectionError,
)
from app.vector_store.qdrant_store import QdrantStore
from app.vector_store.vector_filters import (
    FilterCondition,
    FilterOperator,
    VectorFilter,
)


# ==============================================================
# Test Configuration
# ==============================================================

TEST_COLLECTION = "supportmind_test_vectors"

VECTOR_DIMENSION = 3

QDRANT_HOST = os.getenv(
    "QDRANT_TEST_HOST",
    "localhost",
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_TEST_PORT",
        "6333",
    )
)


# ==============================================================
# Test Data Helpers
# ==============================================================


def make_embedding(
    node_id: str,
    document_id: str,
    text: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> EmbeddedNode:
    """
    Create a deterministic EmbeddedNode for testing.
    """

    return EmbeddedNode(
        node_id=node_id,
        document_id=document_id,
        text=text,
        embedding=embedding,
        embedding_model="test-model",
        embedding_dimension=len(embedding),
        metadata=metadata or {},
    )


def make_node(
    node_id: str = "node_1",
    document_id: str = "doc_1",
    text: str = "Refund policy information.",
    embedding: list[float] | None = None,
    metadata: dict | None = None,
) -> EmbeddedNode:
    """
    Create a standard test node.
    """

    return make_embedding(
        node_id=node_id,
        document_id=document_id,
        text=text,
        embedding=embedding
        or [1.0, 0.0, 0.0],
        metadata=metadata,
    )


def make_collection_info(
    dimension: int = VECTOR_DIMENSION,
) -> MagicMock:
    """
    Create a mocked Qdrant collection response
    containing a standard VectorParams configuration.
    """

    collection_info = MagicMock()

    collection_info.config.params.vectors = (
        models.VectorParams(
            size=dimension,
            distance=models.Distance.COSINE,
        )
    )

    return collection_info


# ==============================================================
# Unit Tests
# ==============================================================


class TestQdrantStoreUnit(unittest.TestCase):
    """
    Unit tests for QdrantStore.

    QdrantClient is fully mocked.
    No real network connection is required.
    """

    def setUp(self):

        self.client = MagicMock()

        # Collection exists by default.
        self.client.collection_exists.return_value = True

        # Collection is configured for 3-dimensional vectors.
        self.client.get_collection.return_value = (
            make_collection_info(
                VECTOR_DIMENSION
            )
        )

        self.store = QdrantStore(
            client=self.client,
        )

    # ==========================================================
    # Provider
    # ==========================================================

    def test_provider(self):

        self.assertEqual(
            self.store.provider,
            "qdrant",
        )

    # ==========================================================
    # Collection Management
    # ==========================================================

    def test_create_collection(self):

        self.client.collection_exists.return_value = False

        self.store.create_collection(
            collection_name=TEST_COLLECTION,
            dimension=VECTOR_DIMENSION,
        )

        self.client.create_collection.assert_called_once()

        kwargs = (
            self.client
            .create_collection
            .call_args
            .kwargs
        )

        self.assertEqual(
            kwargs["collection_name"],
            TEST_COLLECTION,
        )

        self.assertEqual(
            kwargs["vectors_config"].size,
            VECTOR_DIMENSION,
        )

        self.assertEqual(
            kwargs["vectors_config"].distance,
            models.Distance.COSINE,
        )

    def test_create_collection_existing_collection(self):

        self.client.collection_exists.return_value = True

        with self.assertRaises(
            VectorCollectionAlreadyExistsError
        ):

            self.store.create_collection(
                collection_name=TEST_COLLECTION,
                dimension=VECTOR_DIMENSION,
            )

        self.client.create_collection.assert_not_called()

    def test_create_collection_invalid_dimension(self):

        with self.assertRaises(ValueError):

            self.store.create_collection(
                collection_name=TEST_COLLECTION,
                dimension=0,
            )

        self.client.create_collection.assert_not_called()

    def test_create_collection_empty_name(self):

        with self.assertRaises(ValueError):

            self.store.create_collection(
                collection_name="",
                dimension=VECTOR_DIMENSION,
            )

        self.client.create_collection.assert_not_called()

    def test_create_collection_qdrant_error(self):

        self.client.collection_exists.return_value = False

        self.client.create_collection.side_effect = Exception(
            "Qdrant unavailable"
        )

        with self.assertRaises(
            UnsupportedVectorOperationError
        ):

            self.store.create_collection(
                collection_name=TEST_COLLECTION,
                dimension=VECTOR_DIMENSION,
            )

    def test_collection_exists(self):

        self.client.collection_exists.return_value = True

        result = self.store.collection_exists(
            TEST_COLLECTION
        )

        self.assertTrue(result)

        self.client.collection_exists.assert_called_once_with(
            TEST_COLLECTION
        )

    def test_collection_exists_returns_false(self):

        self.client.collection_exists.return_value = False

        result = self.store.collection_exists(
            TEST_COLLECTION
        )

        self.assertFalse(result)

    def test_collection_exists_qdrant_error(self):

        self.client.collection_exists.side_effect = Exception(
            "Connection refused"
        )

        with self.assertRaises(
            VectorStoreConnectionError
        ):

            self.store.collection_exists(
                TEST_COLLECTION
            )

    def test_delete_collection(self):

        self.client.collection_exists.return_value = True

        self.store.delete_collection(
            TEST_COLLECTION
        )

        self.client.delete_collection.assert_called_once_with(
            collection_name=TEST_COLLECTION
        )

    def test_delete_nonexistent_collection(self):

        self.client.collection_exists.return_value = False

        with self.assertRaises(
            VectorCollectionNotFoundError
        ):

            self.store.delete_collection(
                TEST_COLLECTION
            )

        self.client.delete_collection.assert_not_called()

    # ==========================================================
    # Upsert
    # ==========================================================

    def test_upsert(self):

        self.client.collection_exists.return_value = True

        self.client.get_collection.return_value = (
            make_collection_info(
                VECTOR_DIMENSION
            )
        )

        node = make_node()

        self.store.upsert(
            collection_name=TEST_COLLECTION,
            node=node,
        )

        self.client.upsert.assert_called_once()

        kwargs = (
            self.client
            .upsert
            .call_args
            .kwargs
        )

        self.assertEqual(
            kwargs["collection_name"],
            TEST_COLLECTION,
        )

        points = kwargs["points"]

        self.assertEqual(
            len(points),
            1,
        )

        self.assertEqual(
            str(points[0].id),
            node.node_id,
        )

        self.assertEqual(
            points[0].vector,
            node.embedding,
        )

    def test_upsert_empty_collection_name(self):

        node = make_node()

        with self.assertRaises(ValueError):

            self.store.upsert(
                collection_name="",
                node=node,
            )

        self.client.upsert.assert_not_called()

    def test_upsert_collection_not_found(self):

        self.client.collection_exists.return_value = False

        node = make_node()

        with self.assertRaises(
            VectorCollectionNotFoundError
        ):

            self.store.upsert(
                collection_name=TEST_COLLECTION,
                node=node,
            )

        self.client.upsert.assert_not_called()

    def test_upsert_dimension_mismatch(self):

        self.client.collection_exists.return_value = True

        self.client.get_collection.return_value = (
            make_collection_info(
                VECTOR_DIMENSION
            )
        )

        node = make_node(
            embedding=[
                1.0,
                0.0,
            ],
        )

        with self.assertRaises(
            InvalidVectorDimensionError
        ):

            self.store.upsert(
                collection_name=TEST_COLLECTION,
                node=node,
            )

        self.client.upsert.assert_not_called()

    def test_upsert_qdrant_error(self):

        self.client.collection_exists.return_value = True

        self.client.get_collection.return_value = (
            make_collection_info(
                VECTOR_DIMENSION
            )
        )

        self.client.upsert.side_effect = Exception(
            "Qdrant write failure"
        )

        node = make_node()

        with self.assertRaises(
            VectorInsertionError
        ):

            self.store.upsert(
                collection_name=TEST_COLLECTION,
                node=node,
            )

    def test_upsert_batch(self):

        self.client.collection_exists.return_value = True

        self.client.get_collection.return_value = (
            make_collection_info(
                VECTOR_DIMENSION
            )
        )

        nodes = [
            make_node(
                node_id="node_1",
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="node_2",
                embedding=[
                    0.0,
                    1.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="node_3",
                embedding=[
                    0.0,
                    0.0,
                    1.0,
                ],
            ),
        ]

        self.store.upsert_batch(
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

        self.client.upsert.assert_called_once()

        kwargs = (
            self.client
            .upsert
            .call_args
            .kwargs
        )

        self.assertEqual(
            len(kwargs["points"]),
            3,
        )

    def test_upsert_batch_empty_list(self):

        self.store.upsert_batch(
            collection_name=TEST_COLLECTION,
            nodes=[],
        )

        self.client.upsert.assert_not_called()

    def test_upsert_batch_dimension_mismatch(self):

        self.client.collection_exists.return_value = True

        self.client.get_collection.return_value = (
            make_collection_info(
                VECTOR_DIMENSION
            )
        )

        nodes = [
            make_node(
                node_id="valid_node",
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="invalid_node",
                embedding=[
                    1.0,
                    0.0,
                ],
            ),
        ]

        with self.assertRaises(
            InvalidVectorDimensionError
        ):

            self.store.upsert_batch(
                collection_name=TEST_COLLECTION,
                nodes=nodes,
            )

        # No partial batch write.
        self.client.upsert.assert_not_called()

    # ==========================================================
    # Retrieval
    # ==========================================================

    def test_get(self):

        self.client.collection_exists.return_value = True

        self.client.retrieve.return_value = [
            models.Record(
                id="node_1",
                payload={
                    "node_id": "node_1",
                    "document_id": "doc_1",
                    "text": "Refund policy",
                    "embedding_model": "test-model",
                    "metadata": {},
                },
            )
        ]

        results = self.store.get(
            collection_name=TEST_COLLECTION,
            ids=[
                "node_1"
            ],
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].node_id,
            "node_1",
        )

        self.assertEqual(
            results[0].provider,
            "qdrant",
        )

        self.assertEqual(
            results[0].score,
            0.0,
        )

        self.client.retrieve.assert_called_once()

    def test_get_empty_ids(self):

        result = self.store.get(
            collection_name=TEST_COLLECTION,
            ids=[],
        )

        self.assertEqual(
            result,
            [],
        )

        self.client.retrieve.assert_not_called()

    def test_get_collection_not_found(self):

        self.client.collection_exists.return_value = False

        with self.assertRaises(
            VectorCollectionNotFoundError
        ):

            self.store.get(
                collection_name=TEST_COLLECTION,
                ids=[
                    "node_1"
                ],
            )

        self.client.retrieve.assert_not_called()

    # ==========================================================
    # Deletion
    # ==========================================================

    def test_delete(self):

        self.client.collection_exists.return_value = True

        self.store.delete(
            collection_name=TEST_COLLECTION,
            node_id="node_1",
        )

        self.client.delete.assert_called_once()

        kwargs = (
            self.client
            .delete
            .call_args
            .kwargs
        )

        self.assertEqual(
            kwargs["collection_name"],
            TEST_COLLECTION,
        )

        self.assertEqual(
            kwargs["points_selector"].points,
            [
                "node_1"
            ],
        )

    def test_delete_batch(self):

        self.client.collection_exists.return_value = True

        self.store.delete_batch(
            collection_name=TEST_COLLECTION,
            ids=[
                "node_1",
                "node_2",
                "node_3",
            ],
        )

        self.client.delete.assert_called_once()

        kwargs = (
            self.client
            .delete
            .call_args
            .kwargs
        )

        self.assertEqual(
            kwargs["collection_name"],
            TEST_COLLECTION,
        )

        self.assertEqual(
            kwargs["points_selector"].points,
            [
                "node_1",
                "node_2",
                "node_3",
            ],
        )

    def test_delete_batch_empty_ids(self):

        self.store.delete_batch(
            collection_name=TEST_COLLECTION,
            ids=[],
        )

        self.client.delete.assert_not_called()

    def test_delete_qdrant_error(self):

        self.client.collection_exists.return_value = True

        self.client.delete.side_effect = Exception(
            "Qdrant deletion failure"
        )

        with self.assertRaises(
            VectorDeletionError
        ):

            self.store.delete(
                collection_name=TEST_COLLECTION,
                node_id="node_1",
            )

    # ==========================================================
    # Search
    # ==========================================================

    def test_search(self):

        self.client.collection_exists.return_value = True

        point = MagicMock()

        point.id = "node_1"

        point.score = 0.98

        point.payload = {
            "node_id": "node_1",
            "document_id": "doc_1",
            "text": "Refund policy",
            "embedding_model": "test-model",
            "metadata": {
                "category": "billing",
            },
        }

        response = MagicMock()

        response.points = [
            point
        ]

        self.client.query_points.return_value = response

        results = self.store.search(
            collection_name=TEST_COLLECTION,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=5,
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].node_id,
            "node_1",
        )

        self.assertAlmostEqual(
            results[0].score,
            0.98,
        )

        self.client.query_points.assert_called_once()

        kwargs = (
            self.client
            .query_points
            .call_args
            .kwargs
        )

        self.assertEqual(
            kwargs["collection_name"],
            TEST_COLLECTION,
        )

        self.assertEqual(
            kwargs["query"],
            [
                1.0,
                0.0,
                0.0,
            ],
        )

        self.assertEqual(
            kwargs["limit"],
            5,
        )

        self.assertIsNone(
            kwargs["query_filter"]
        )

    def test_search_with_metadata_filter(self):

        self.client.collection_exists.return_value = True

        response = MagicMock()

        response.points = []

        self.client.query_points.return_value = response

        vector_filter = VectorFilter(
            conditions=[
                FilterCondition(
                    field="category",
                    operator=FilterOperator.EQUAL,
                    value="billing",
                )
            ]
        )

        self.store.search(
            collection_name=TEST_COLLECTION,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=5,
            filters=vector_filter,
        )

        self.client.query_points.assert_called_once()

        kwargs = (
            self.client
            .query_points
            .call_args
            .kwargs
        )

        query_filter = kwargs["query_filter"]

        self.assertIsNotNone(
            query_filter
        )

        self.assertEqual(
            len(query_filter.must),
            1,
        )

        self.assertEqual(
            query_filter.must[0].key,
            "metadata.category",
        )

        self.assertEqual(
            query_filter.must[0].match.value,
            "billing",
        )

    def test_search_invalid_top_k(self):

        with self.assertRaises(ValueError):

            self.store.search(
                collection_name=TEST_COLLECTION,
                query_vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
                top_k=0,
            )

        self.client.query_points.assert_not_called()

    def test_search_empty_vector(self):

        with self.assertRaises(ValueError):

            self.store.search(
                collection_name=TEST_COLLECTION,
                query_vector=[],
            )

        self.client.query_points.assert_not_called()

    def test_search_non_numeric_vector(self):

        with self.assertRaises(ValueError):

            self.store.search(
                collection_name=TEST_COLLECTION,
                query_vector=[
                    1.0,
                    "invalid",
                    0.0,
                ],
            )

        self.client.query_points.assert_not_called()

    def test_search_collection_not_found(self):

        self.client.collection_exists.return_value = False

        with self.assertRaises(
            VectorCollectionNotFoundError
        ):

            self.store.search(
                collection_name=TEST_COLLECTION,
                query_vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
            )

        self.client.query_points.assert_not_called()

    def test_search_qdrant_error(self):

        self.client.collection_exists.return_value = True

        self.client.query_points.side_effect = Exception(
            "Qdrant search failure"
        )

        with self.assertRaises(
            VectorSearchError
        ):

            self.store.search(
                collection_name=TEST_COLLECTION,
                query_vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
            )

    # ==========================================================
    # Count
    # ==========================================================

    def test_count(self):

        self.client.collection_exists.return_value = True

        response = MagicMock()

        response.count = 42

        self.client.count.return_value = response

        result = self.store.count(
            TEST_COLLECTION
        )

        self.assertEqual(
            result,
            42,
        )

        self.client.count.assert_called_once_with(
            collection_name=TEST_COLLECTION,
            exact=True,
        )

    def test_count_collection_not_found(self):

        self.client.collection_exists.return_value = False

        with self.assertRaises(
            VectorCollectionNotFoundError
        ):

            self.store.count(
                TEST_COLLECTION
            )

    # ==========================================================
    # Health
    # ==========================================================

    def test_health_check_success(self):

        self.client.get_collections.return_value = (
            MagicMock()
        )

        result = self.store.health_check()

        self.assertTrue(result)

        self.client.get_collections.assert_called_once()

    def test_health_check_failure(self):

        self.client.get_collections.side_effect = Exception(
            "Connection refused"
        )

        result = self.store.health_check()

        self.assertFalse(result)


# ==============================================================
# Integration Tests
# ==============================================================


@unittest.skipUnless(
    os.getenv(
        "RUN_QDRANT_INTEGRATION_TESTS",
        "false",
    ).lower()
    == "true",
    (
        "Qdrant integration tests disabled. "
        "Set RUN_QDRANT_INTEGRATION_TESTS=true."
    ),
)
class TestQdrantStoreIntegration(unittest.TestCase):
    """
    Integration tests against a real Qdrant instance.

    Expected Qdrant endpoint:

        http://localhost:6333

    PowerShell:

        $env:RUN_QDRANT_INTEGRATION_TESTS="true"

        python -m unittest tests.test_qdrant_store -v
    """

    @classmethod
    def setUpClass(cls):

        cls.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )

        cls.store = QdrantStore(
            client=cls.client,
        )

        try:

            cls.client.get_collections()

        except Exception as exc:

            raise unittest.SkipTest(
                f"Qdrant is unavailable: {exc}"
            )

    def setUp(self):

        if self.store.collection_exists(
            TEST_COLLECTION
        ):

            self.store.delete_collection(
                TEST_COLLECTION
            )

        self.store.create_collection(
            collection_name=TEST_COLLECTION,
            dimension=VECTOR_DIMENSION,
        )

    def tearDown(self):

        if self.store.collection_exists(
            TEST_COLLECTION
        ):

            self.store.delete_collection(
                TEST_COLLECTION
            )

    # ==========================================================
    # Collection
    # ==========================================================

    def test_collection_creation(self):

        self.assertTrue(
            self.store.collection_exists(
                TEST_COLLECTION
            )
        )

    def test_collection_vector_dimension(self):

        info = self.client.get_collection(
            TEST_COLLECTION
        )

        vectors_config = (
            info.config.params.vectors
        )

        self.assertEqual(
            vectors_config.size,
            VECTOR_DIMENSION,
        )

    # ==========================================================
    # Upsert
    # ==========================================================

    def test_upsert_and_retrieve(self):

        node = make_node(
            node_id="integration_node_1",
            document_id="integration_doc_1",
            text="Refund policy information.",
            embedding=[
                1.0,
                0.0,
                0.0,
            ],
            metadata={
                "category": "billing",
                "source": "integration-test",
            },
        )

        self.store.upsert(
            collection_name=TEST_COLLECTION,
            node=node,
        )

        self.assertEqual(
            self.store.count(
                TEST_COLLECTION
            ),
            1,
        )

        results = self.store.get(
            collection_name=TEST_COLLECTION,
            ids=[
                node.node_id
            ],
        )

        self.assertEqual(
            len(results),
            1,
        )

        self.assertEqual(
            results[0].node_id,
            node.node_id,
        )

        self.assertEqual(
            results[0].document_id,
            node.document_id,
        )

        self.assertEqual(
            results[0].text,
            node.text,
        )

        self.assertEqual(
            results[0].metadata,
            node.metadata,
        )

    def test_batch_upsert(self):

        nodes = [
            make_node(
                node_id="batch_1",
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="batch_2",
                embedding=[
                    0.0,
                    1.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="batch_3",
                embedding=[
                    0.0,
                    0.0,
                    1.0,
                ],
            ),
        ]

        self.store.upsert_batch(
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

        self.assertEqual(
            self.store.count(
                TEST_COLLECTION
            ),
            3,
        )

    # ==========================================================
    # Similarity Search
    # ==========================================================

    def test_similarity_search(self):

        nodes = [
            make_node(
                node_id="search_refund",
                text="Refund policy and refund processing.",
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="search_login",
                text="Login troubleshooting guide.",
                embedding=[
                    0.0,
                    1.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="search_payment",
                text="Payment failure troubleshooting.",
                embedding=[
                    0.0,
                    0.0,
                    1.0,
                ],
            ),
        ]

        self.store.upsert_batch(
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

        results = self.store.search(
            collection_name=TEST_COLLECTION,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=2,
        )

        self.assertEqual(
            len(results),
            2,
        )

        self.assertEqual(
            results[0].node_id,
            "search_refund",
        )

        self.assertGreaterEqual(
            results[0].score,
            results[1].score,
        )

    # ==========================================================
    # Filtering
    # ==========================================================

    def test_filtered_search(self):

        nodes = [
            make_node(
                node_id="billing_1",
                text="Billing refund policy.",
                embedding=[
                    1.0,
                    0.0,
                    0.0,
                ],
                metadata={
                    "category": "billing",
                },
            ),
            make_node(
                node_id="technical_1",
                text="Technical login issue.",
                embedding=[
                    0.99,
                    0.01,
                    0.0,
                ],
                metadata={
                    "category": "technical",
                },
            ),
        ]

        self.store.upsert_batch(
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

        vector_filter = VectorFilter(
            conditions=[
                FilterCondition(
                    field="category",
                    operator=FilterOperator.EQUAL,
                    value="billing",
                )
            ]
        )

        results = self.store.search(
            collection_name=TEST_COLLECTION,
            query_vector=[
                1.0,
                0.0,
                0.0,
            ],
            top_k=5,
            filters=vector_filter,
        )

        self.assertGreaterEqual(
            len(results),
            1,
        )

        for result in results:

            self.assertEqual(
                result.metadata["category"],
                "billing",
            )

    # ==========================================================
    # Deletion
    # ==========================================================

    def test_delete(self):

        node = make_node(
            node_id="delete_node",
        )

        self.store.upsert(
            collection_name=TEST_COLLECTION,
            node=node,
        )

        self.assertEqual(
            self.store.count(
                TEST_COLLECTION
            ),
            1,
        )

        self.store.delete(
            collection_name=TEST_COLLECTION,
            node_id=node.node_id,
        )

        self.assertEqual(
            self.store.count(
                TEST_COLLECTION
            ),
            0,
        )

    def test_delete_batch(self):

        nodes = [
            make_node(
                node_id="delete_1",
            ),
            make_node(
                node_id="delete_2",
                embedding=[
                    0.0,
                    1.0,
                    0.0,
                ],
            ),
            make_node(
                node_id="delete_3",
                embedding=[
                    0.0,
                    0.0,
                    1.0,
                ],
            ),
        ]

        self.store.upsert_batch(
            collection_name=TEST_COLLECTION,
            nodes=nodes,
        )

        self.assertEqual(
            self.store.count(
                TEST_COLLECTION
            ),
            3,
        )

        self.store.delete_batch(
            collection_name=TEST_COLLECTION,
            ids=[
                "delete_1",
                "delete_2",
            ],
        )

        self.assertEqual(
            self.store.count(
                TEST_COLLECTION
            ),
            1,
        )

        remaining = self.store.get(
            collection_name=TEST_COLLECTION,
            ids=[
                "delete_3",
            ],
        )

        self.assertEqual(
            len(remaining),
            1,
        )

        self.assertEqual(
            remaining[0].node_id,
            "delete_3",
        )

    # ==========================================================
    # Health
    # ==========================================================

    def test_health_check(self):

        self.assertTrue(
            self.store.health_check()
        )


# ==============================================================
# Test Runner
# ==============================================================

if __name__ == "__main__":

    unittest.main()