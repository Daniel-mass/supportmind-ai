"""
embedding_registry.py

Central registry responsible for managing all embedding
providers available inside SupportMind AI.

Providers are registered as classes instead of instantiated
objects. This avoids loading heavyweight embedding models
during application startup.

Responsibilities
----------------
1. Register embedding providers.
2. Create embedders on demand.
3. Validate provider names.
4. List supported providers.

Author: SupportMind AI
"""

from app.embeddings.fastembed_embedder import FastEmbedder
from app.embeddings.huggingface_embedder import HuggingFaceEmbedder
from app.embeddings.ollama_embedder import OllamaEmbedder


class EmbeddingRegistry:
    """
    Factory responsible for creating embedders.
    """

    def __init__(self):

        self._registry = {

            "huggingface": HuggingFaceEmbedder,

            "fastembed": FastEmbedder,

            "ollama": OllamaEmbedder,

        }

    def get_embedder(self, provider: str):

        provider = provider.lower()

        if provider not in self._registry:

            available = ", ".join(

                sorted(self._registry.keys())

            )

            raise ValueError(

                f"Unknown embedding provider "

                f"'{provider}'. "

                f"Available providers: {available}"

            )

        return self._registry[provider]()

    def supported_embedders(self):

        return sorted(

            self._registry.keys()

        )

    def is_supported(self, provider: str):

        return provider.lower() in self._registry