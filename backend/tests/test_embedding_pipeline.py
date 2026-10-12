"""
Tests for EmbeddingPipeline.

Author: SupportMind AI
"""

import unittest

from llama_index.core.schema import TextNode

from app.embeddings.embedding_pipeline import EmbeddingPipeline
from app.embeddings.embedding_schema import EmbeddedNode


class TestEmbeddingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.pipeline = EmbeddingPipeline()

    def _node(self, text: str):

        return TextNode(

            id_="node-1",

            text=text,

            metadata={

                "document_id": "DOC-001",

                "source": "supportmind.txt",

            },

        )

    # -----------------------------------------------------

    def test_embed_single_node(self):

        node = self._node(

            "SupportMind AI uses semantic retrieval."

        )

        embedded = self.pipeline.embed(

            node,

            provider="fastembed",

        )

        self.assertIsInstance(

            embedded,

            EmbeddedNode,

        )

        self.assertGreater(

            len(embedded.embedding),

            0,

        )

    # -----------------------------------------------------

    def test_embed_batch(self):

        nodes = [

            self._node("Sentence one."),

            self._node("Sentence two."),

            self._node("Sentence three."),

        ]

        embedded = self.pipeline.embed(

            nodes,

            provider="fastembed",

        )

        self.assertEqual(

            len(embedded),

            3,

        )

        for node in embedded:

            self.assertIsInstance(

                node,

                EmbeddedNode,

            )

    # -----------------------------------------------------

    def test_default_provider(self):

        node = self._node(

            "Default provider should work."

        )

        embedded = self.pipeline.embed(

            node,

        )

        self.assertIsInstance(

            embedded,

            EmbeddedNode,

        )

    # -----------------------------------------------------

    def test_invalid_provider(self):

        node = self._node(

            "Invalid provider."

        )

        with self.assertRaises(ValueError):

            self.pipeline.embed(

                node,

                provider="unknown_provider",

            )

    # -----------------------------------------------------

    def test_embedding_dimension_consistent(self):

        nodes = [

            self._node("One"),

            self._node("Two"),

            self._node("Three"),

        ]

        embedded = self.pipeline.embed(

            nodes,

            provider="fastembed",

        )

        dimensions = {

            len(node.embedding)

            for node in embedded

        }

        self.assertEqual(

            len(dimensions),

            1,

        )

    # -----------------------------------------------------

    def test_metadata_preserved(self):

        node = self._node(

            "Metadata test."

        )

        embedded = self.pipeline.embed(

            node,

            provider="fastembed",

        )

        self.assertEqual(

            embedded.document_id,

            "DOC-001",

        )

        self.assertEqual(

            embedded.metadata["source"],

            "supportmind.txt",

        )


if __name__ == "__main__":

    unittest.main()