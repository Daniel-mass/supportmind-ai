"""
Unit tests for VectorSearchResult.

SupportMind AI
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from app.vector_store.vector_search_result import VectorSearchResult


class TestVectorSearchResult(unittest.TestCase):

    def make_result(self, **kwargs):

        data = {

            "id": "point_1",

            "node_id": "node_1",

            "document_id": "doc_1",

            "text": "Refund policy",

            "score": 0.98,

            "embedding_model": "bge-m3",

            "provider": "ollama",

            "collection_name": "supportmind_documents",

            "metadata": {},

        }

        data.update(kwargs)

        return VectorSearchResult(**data)

    def test_create_result(self):

        result = self.make_result()

        self.assertEqual(result.id, "point_1")
        self.assertEqual(result.node_id, "node_1")
        self.assertEqual(result.document_id, "doc_1")
        self.assertEqual(result.text, "Refund policy")
        self.assertEqual(result.score, 0.98)
        self.assertEqual(result.embedding_model, "bge-m3")
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.collection_name, "supportmind_documents")

    def test_metadata_default(self):

        result = self.make_result()

        self.assertEqual(result.metadata, {})

    def test_metadata_custom(self):

        result = self.make_result(

            metadata={
                "source": "pdf",
                "page": 8,
            }

        )

        self.assertEqual(result.metadata["source"], "pdf")
        self.assertEqual(result.metadata["page"], 8)

    def test_validation(self):

        with self.assertRaises(ValidationError):

            self.make_result(score="high")

    def test_sort_by_score(self):

        results = [

            self.make_result(
                id="1",
                node_id="1",
                score=0.40,
                text="A",
            ),

            self.make_result(
                id="2",
                node_id="2",
                score=0.95,
                text="B",
            ),

            self.make_result(
                id="3",
                node_id="3",
                score=0.70,
                text="C",
            ),

        ]

        results.sort(key=lambda x: x.score, reverse=True)

        self.assertEqual(results[0].score, 0.95)
        self.assertEqual(results[1].score, 0.70)
        self.assertEqual(results[2].score, 0.40)

    def test_model_dump(self):

        result = self.make_result(score=0.77)

        dumped = result.model_dump()

        self.assertEqual(dumped["id"], "point_1")
        self.assertEqual(dumped["node_id"], "node_1")
        self.assertEqual(dumped["provider"], "ollama")
        self.assertEqual(dumped["embedding_model"], "bge-m3")
        self.assertEqual(dumped["collection_name"], "supportmind_documents")
        self.assertEqual(dumped["score"], 0.77)

    def test_json_serialization(self):

        result = self.make_result(score=0.55)

        json_data = result.model_dump_json()

        self.assertIn("id", json_data)
        self.assertIn("node_id", json_data)
        self.assertIn("provider", json_data)
        self.assertIn("embedding_model", json_data)
        self.assertIn("collection_name", json_data)
        self.assertIn("score", json_data)

    def test_equality(self):

        a = self.make_result()

        b = self.make_result()

        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()