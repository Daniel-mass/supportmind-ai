import unittest

from llama_index.core.schema import TextNode

from app.embeddings.base_embedder import BaseEmbedder
from app.embeddings.embedding_schema import EmbeddedNode


class DummyEmbedder(BaseEmbedder):
    """
    Minimal implementation for testing BaseEmbedder.
    """

    def _embed(
        self,
        node: TextNode,
    ) -> EmbeddedNode:

        return EmbeddedNode(

            node_id=node.node_id,

            document_id=node.metadata["document_id"],

            text=node.text,

            embedding=[0.1, 0.2, 0.3],

            embedding_model="dummy",

            embedding_dimension=3,

            metadata=node.metadata,

        )

    def _embed_batch(
        self,
        nodes,
    ):

        return [self._embed(node) for node in nodes]


class TestBaseEmbedder(unittest.TestCase):

    def setUp(self):

        self.embedder = DummyEmbedder()

    def test_single_embedding(self):

        node = TextNode(

            text="SupportMind AI",

            metadata={

                "document_id": "DOC001"

            }

        )

        result = self.embedder.embed(node)

        self.assertEqual(result.document_id, "DOC001")

        self.assertEqual(result.embedding_dimension, 3)

        self.assertEqual(result.embedding_model, "dummy")

    def test_batch_embedding(self):

        nodes = [

            TextNode(

                text=f"Sentence {i}",

                metadata={

                    "document_id": "DOC001"

                }

            )

            for i in range(5)

        ]

        results = self.embedder.embed_batch(nodes)

        self.assertEqual(len(results), 5)

    def test_empty_batch(self):

        result = self.embedder.embed_batch([])

        self.assertEqual(result, [])

    def test_empty_text(self):

        node = TextNode(

            text="",

            metadata={

                "document_id": "DOC001"

            }

        )

        with self.assertRaises(ValueError):

            self.embedder.embed(node)

    def test_invalid_type(self):

        with self.assertRaises(TypeError):

            self.embedder.embed("hello")


if __name__ == "__main__":

    unittest.main()