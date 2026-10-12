"""
Tests for EmbeddingRegistry.

Author: SupportMind AI
"""

import unittest

from app.embeddings.embedding_registry import EmbeddingRegistry
from app.embeddings.fastembed_embedder import FastEmbedder
from app.embeddings.huggingface_embedder import HuggingFaceEmbedder
from app.embeddings.ollama_embedder import OllamaEmbedder


class TestEmbeddingRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.registry = EmbeddingRegistry()

    # --------------------------------------------------

    def test_get_huggingface_embedder(self):

        embedder = self.registry.get_embedder(
            "huggingface"
        )

        self.assertIsInstance(
            embedder,
            HuggingFaceEmbedder,
        )

    # --------------------------------------------------

    def test_get_fastembed_embedder(self):

        embedder = self.registry.get_embedder(
            "fastembed"
        )

        self.assertIsInstance(
            embedder,
            FastEmbedder,
        )

    # --------------------------------------------------

    def test_get_ollama_embedder(self):

        embedder = self.registry.get_embedder(
            "ollama"
        )

        self.assertIsInstance(
            embedder,
            OllamaEmbedder,
        )

    # --------------------------------------------------

    def test_case_insensitive_provider(self):

        embedder = self.registry.get_embedder(
            "OLLAMA"
        )

        self.assertIsInstance(
            embedder,
            OllamaEmbedder,
        )

    # --------------------------------------------------

    def test_supported_embedders(self):

        providers = self.registry.supported_embedders()

        self.assertEqual(

            providers,

            [
                "fastembed",
                "huggingface",
                "ollama",
            ],

        )

    # --------------------------------------------------

    def test_is_supported(self):

        self.assertTrue(

            self.registry.is_supported(
                "huggingface"
            )

        )

        self.assertTrue(

            self.registry.is_supported(
                "fastembed"
            )

        )

        self.assertTrue(

            self.registry.is_supported(
                "ollama"
            )

        )

        self.assertFalse(

            self.registry.is_supported(
                "random_provider"
            )

        )

    # --------------------------------------------------

    def test_unknown_provider(self):

        with self.assertRaises(ValueError) as context:

            self.registry.get_embedder(
                "random_provider"
            )

        self.assertIn(

            "Unknown embedding provider",

            str(context.exception),

        )

    # --------------------------------------------------

    def test_registry_returns_new_instance(self):
        """
        Registry should construct a new embedder
        each time instead of reusing instances.
        """

        first = self.registry.get_embedder(
            "ollama"
        )

        second = self.registry.get_embedder(
            "ollama"
        )

        self.assertIsNot(
            first,
            second,
        )


if __name__ == "__main__":

    unittest.main()