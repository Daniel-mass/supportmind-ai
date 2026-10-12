"""
Test Embedding Schema
"""

import unittest

from app.embeddings.embedding_schema import EmbeddedNode


class TestEmbeddingSchema(unittest.TestCase):

    def test_create_schema(self):

        node = EmbeddedNode(
            node_id="node-001",
            document_id="DOC-001",
            text="SupportMind AI processes customer refunds.",
            embedding=[0.1, 0.2, 0.3],
            embedding_model="bge-large-en-v1.5",
            embedding_dimension=3,
            metadata={
                "source": "supportmind.txt",
                "chunk_index": 0
            }
        )

        self.assertEqual(node.node_id, "node-001")
        self.assertEqual(node.document_id, "DOC-001")
        self.assertEqual(len(node.embedding), 3)
        self.assertEqual(node.embedding_dimension, 3)
        self.assertEqual(node.embedding_model, "bge-large-en-v1.5")


if __name__ == "__main__":
    unittest.main()