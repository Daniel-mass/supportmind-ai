"""
Unit tests for BaseVectorStore.

Uses a minimal in-memory implementation to verify the
provider-agnostic interface.

SupportMind AI
"""

from __future__ import annotations

import unittest

from app.embeddings.embedding_schema import EmbeddedNode
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.vector_filters import VectorFilter
from app.vector_store.vector_search_result import VectorSearchResult


class DummyVectorStore(BaseVectorStore):
    """
    Minimal in-memory implementation for testing the
    BaseVectorStore contract.
    """

    def __init__(self):

        self.storage: dict[str, EmbeddedNode] = {}

    # ---------------------------------------------------------
    # Provider
    # ---------------------------------------------------------

    @property
    def provider(self) -> str:
        return "dummy"

    # ---------------------------------------------------------
    # Collections
    # ---------------------------------------------------------

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:
        return None

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        self.storage.clear()

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        return True

    # ---------------------------------------------------------
    # Upsert
    # ---------------------------------------------------------

    def upsert(
        self,
        collection_name: str,
        node: EmbeddedNode,
    ) -> None:

        self.storage[node.node_id] = node

    def upsert_batch(
        self,
        collection_name: str,
        nodes: list[EmbeddedNode],
    ) -> None:

        for node in nodes:
            self.storage[node.node_id] = node

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def get(
        self,
        collection_name: str,
        ids: list[str],
    ) -> list[VectorSearchResult]:

        results = []

        for node_id in ids:

            node = self.storage.get(node_id)

            if node is None:
                continue

            results.append(

                VectorSearchResult(

                    id=node.node_id,

                    node_id=node.node_id,

                    document_id=node.document_id,

                    text=node.text,

                    score=1.0,

                    embedding_model=node.embedding_model,

                    provider=self.provider,

                    collection_name=collection_name,

                    metadata=node.metadata,

                )

            )

        return results

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete(
        self,
        collection_name: str,
        node_id: str,
    ) -> None:

        self.storage.pop(node_id, None)

    def delete_batch(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:

        for node_id in ids:
            self.storage.pop(node_id, None)

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: VectorFilter | None = None,
    ) -> list[VectorSearchResult]:

        results = []

        for node in self.storage.values():

            results.append(

                VectorSearchResult(

                    id=node.node_id,

                    node_id=node.node_id,

                    document_id=node.document_id,

                    text=node.text,

                    score=1.0,

                    embedding_model=node.embedding_model,

                    provider=self.provider,

                    collection_name=collection_name,

                    metadata=node.metadata,

                )

            )

        return results[:top_k]

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def count(
        self,
        collection_name: str,
    ) -> int:

        return len(self.storage)

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def health_check(self) -> bool:

        return True


class TestBaseVectorStore(unittest.TestCase):

    def setUp(self):

        self.store = DummyVectorStore()

        self.node = EmbeddedNode(

            node_id="node_1",

            document_id="doc_1",

            text="Refund policy",

            embedding=[0.1, 0.2, 0.3],

            embedding_model="bge-m3",

            embedding_dimension=3,

            metadata={

                "source": "pdf",

            },

        )

    # ---------------------------------------------------------
    # Provider
    # ---------------------------------------------------------

    def test_provider(self):

        self.assertEqual(

            self.store.provider,

            "dummy",

        )

    # ---------------------------------------------------------
    # Upsert
    # ---------------------------------------------------------

    def test_upsert_single(self):

        self.store.upsert(

            "documents",

            self.node,

        )

        self.assertEqual(

            self.store.count("documents"),

            1,

        )

    def test_upsert_batch(self):

        node2 = self.node.model_copy(

            update={

                "node_id": "node_2",

            }

        )

        self.store.upsert_batch(

            "documents",

            [

                self.node,

                node2,

            ],

        )

        self.assertEqual(

            self.store.count("documents"),

            2,

        )

    # ---------------------------------------------------------
    # Get
    # ---------------------------------------------------------

    def test_get(self):

        self.store.upsert(

            "documents",

            self.node,

        )

        results = self.store.get(

            "documents",

            ["node_1"],

        )

        self.assertEqual(

            len(results),

            1,

        )

        self.assertEqual(

            results[0].node_id,

            "node_1",

        )

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def test_search(self):

        self.store.upsert(

            "documents",

            self.node,

        )

        results = self.store.search(

            collection_name="documents",

            query_vector=[0.1, 0.2, 0.3],

        )

        self.assertEqual(

            len(results),

            1,

        )

        self.assertEqual(

            results[0].provider,

            "dummy",

        )

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def test_delete(self):

        self.store.upsert(

            "documents",

            self.node,

        )

        self.store.delete(

            "documents",

            "node_1",

        )

        self.assertEqual(

            self.store.count("documents"),

            0,

        )

    def test_delete_batch(self):

        node2 = self.node.model_copy(

            update={

                "node_id": "node_2",

            }

        )

        self.store.upsert_batch(

            "documents",

            [

                self.node,

                node2,

            ],

        )

        self.store.delete_batch(

            "documents",

            [

                "node_1",

                "node_2",

            ],

        )

        self.assertEqual(

            self.store.count("documents"),

            0,

        )

    # ---------------------------------------------------------
    # Collections
    # ---------------------------------------------------------

    def test_collection_exists(self):

        self.assertTrue(

            self.store.collection_exists(

                "documents",

            )

        )

    def test_delete_collection(self):

        self.store.upsert(

            "documents",

            self.node,

        )

        self.store.delete_collection(

            "documents",

        )

        self.assertEqual(

            self.store.count("documents"),

            0,

        )

    # ---------------------------------------------------------
    # Health
    # ---------------------------------------------------------

    def test_health_check(self):

        self.assertTrue(

            self.store.health_check()

        )


if __name__ == "__main__":

    unittest.main()