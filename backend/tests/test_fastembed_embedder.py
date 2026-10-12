import unittest

from llama_index.core.schema import TextNode

from app.embeddings.fastembed_embedder import FastEmbedder


class TestFastEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.embedder = FastEmbedder()

    def test_dimension(self):

        self.assertGreater(
            self.embedder.embedding_dimension,
            0,
        )

    def test_embed_node(self):

        node = TextNode(
            text="SupportMind AI"
        )

        embedded = self.embedder.embed(node)

        self.assertEqual(
            len(embedded.embedding),
            self.embedder.embedding_dimension,
        )

    def test_embed_batch(self):

        nodes = [

            TextNode(text="refund policy"),

            TextNode(text="technical support"),

            TextNode(text="GraphRAG"),

        ]

        embedded = self.embedder.embed_batch(nodes)

        self.assertEqual(
            len(embedded),
            3,
        )

        for node in embedded:

            self.assertEqual(
                len(node.embedding),
                self.embedder.embedding_dimension,
            )

    def test_model_name(self):

        self.assertEqual(
            self.embedder.model_name,
            "BAAI/bge-base-en-v1.5",
        )

    def test_consistent_dimension(self):

        a = self.embedder.embed(
            TextNode(text="hello")
        )

        b = self.embedder.embed(
            TextNode(text="world")
        )

        self.assertEqual(
            len(a.embedding),
            len(b.embedding),
        )


if __name__ == "__main__":
    unittest.main()