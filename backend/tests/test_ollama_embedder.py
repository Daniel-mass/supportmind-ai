"""
Integration tests for OllamaEmbedder.

Requirements
------------
1. Ollama server running
2. bge-m3:latest installed

Run

    ollama serve

before executing the tests.

Author: SupportMind AI
"""

import unittest

from llama_index.core.schema import TextNode

from app.embeddings.ollama_embedder import OllamaEmbedder
from app.embeddings.embedding_schema import EmbeddedNode


class TestOllamaEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.embedder = OllamaEmbedder()

    def make_node(
        self,
        text: str,
        index: int = 0,
    ):

        return TextNode(

            id_=f"node-{index}",

            text=text,

            metadata={

                "document_id": "DOC-001",

                "source": "supportmind.txt",

                "chunk_index": index,

            },

        )

    # --------------------------------------------------

    def test_provider_name(self):

        self.assertEqual(

            self.embedder.provider,

            "ollama",

        )

    # --------------------------------------------------

    def test_model_name(self):

        self.assertEqual(

            self.embedder.model,

            "bge-m3:latest",

        )

    # --------------------------------------------------

    def test_embed_node(self):

        node = self.make_node(

            "SupportMind AI processes refunds.",

        )

        embedded = self.embedder.embed(node)

        self.assertIsInstance(

            embedded,

            EmbeddedNode,

        )

        self.assertGreater(

            len(embedded.embedding),

            0,

        )

        self.assertEqual(

            embedded.node_id,

            "node-0",

        )

        self.assertEqual(

            embedded.document_id,

            "DOC-001",

        )

        self.assertEqual(

            embedded.embedding_model,

            "bge-m3:latest",

        )

    # --------------------------------------------------

    def test_embed_batch(self):

        nodes = [

            self.make_node(

                "Refund policy",

                0,

            ),

            self.make_node(

                "Payment failed",

                1,

            ),

            self.make_node(

                "Customer support",

                2,

            ),

        ]

        embedded = self.embedder.embed_batch(nodes)

        self.assertEqual(

            len(embedded),

            3,

        )

        for node in embedded:

            self.assertIsInstance(

                node,

                EmbeddedNode,

            )

            self.assertGreater(

                len(node.embedding),

                0,

            )

    # --------------------------------------------------

    def test_consistent_dimension(self):

        a = self.embedder.embed(

            self.make_node(

                "Artificial Intelligence",

                0,

            )

        )

        b = self.embedder.embed(

            self.make_node(

                "Machine Learning",

                1,

            )

        )

        self.assertEqual(

            len(a.embedding),

            len(b.embedding),

        )

        self.assertEqual(

            a.embedding_dimension,

            b.embedding_dimension,

        )

    # --------------------------------------------------

    def test_dimension_property(self):

        node = self.make_node(

            "Dimension test",

        )

        embedded = self.embedder.embed(node)

        self.assertEqual(

            self.embedder.dimension,

            embedded.embedding_dimension,

        )

    # --------------------------------------------------

    def test_metadata_preserved(self):

        node = self.make_node(

            "Metadata test",

            5,

        )

        embedded = self.embedder.embed(node)

        self.assertEqual(

            embedded.metadata["document_id"],

            "DOC-001",

        )

        self.assertEqual(

            embedded.metadata["chunk_index"],

            5,

        )

        self.assertEqual(

            embedded.metadata["source"],

            "supportmind.txt",

        )

    # --------------------------------------------------

    def test_embedding_is_float_vector(self):

        node = self.make_node(

            "Embedding vector",

        )

        embedded = self.embedder.embed(node)

        self.assertTrue(

            all(

                isinstance(x, float)

                for x in embedded.embedding

            )

        )

    # --------------------------------------------------

    def test_empty_text_validation(self):

        node = self.make_node("")

        with self.assertRaises(ValueError):

            self.embedder.embed(node)


if __name__ == "__main__":

    unittest.main()