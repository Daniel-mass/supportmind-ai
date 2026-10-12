import unittest

from llama_index.core.schema import TextNode

from app.embeddings.huggingface_embedder import HuggingFaceEmbedder


class TestHuggingFaceEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.embedder = HuggingFaceEmbedder()

    def test_single_embedding(self):

        node = TextNode(
            text="SupportMind AI automates customer support."
        )

        node.metadata = {
            "document_id": "DOC001"
        }

        embedded = self.embedder.embed(node)

        self.assertEqual(
            embedded.embedding_dimension,
            self.embedder.dimension,
        )

        self.assertEqual(
            len(embedded.embedding),
            self.embedder.dimension,
        )

        self.assertEqual(
            embedded.document_id,
            "DOC001",
        )

    def test_batch_embedding(self):

        nodes = [

            TextNode(text="First document"),

            TextNode(text="Second document"),

            TextNode(text="Third document"),

        ]

        for i, n in enumerate(nodes):
            n.metadata = {
                "document_id": f"DOC{i}"
            }

        results = self.embedder.embed_batch(nodes)

        self.assertEqual(len(results), 3)

        for result in results:

            self.assertEqual(

                len(result.embedding),

                self.embedder.dimension,

            )

    def test_empty_batch(self):

        result = self.embedder.embed_batch([])

        self.assertEqual(result, [])

    def test_dimension(self):

        self.assertGreater(
            self.embedder.dimension,
            0,
        )

    def test_metadata_preserved(self):

        node = TextNode(text="Testing metadata")

        node.metadata = {

            "document_id": "DOC123",

            "department": "Support",

            "priority": "High",

        }

        embedded = self.embedder.embed(node)

        self.assertEqual(

            embedded.metadata["department"],

            "Support",

        )

        self.assertEqual(

            embedded.metadata["priority"],

            "High",

        )


if __name__ == "__main__":
    unittest.main()