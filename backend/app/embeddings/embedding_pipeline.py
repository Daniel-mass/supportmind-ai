"""
#embedding_pipeline.py

Production-grade embedding pipeline.

Responsibilities
----------------
1. Select embedding provider.
2. Execute embedding.
3. Return EmbeddedNode objects.

The pipeline is intentionally provider-agnostic.
It delegates all provider-specific logic to the
EmbeddingRegistry and BaseEmbedder implementations.

Author: SupportMind AI
"""

from __future__ import annotations

from llama_index.core.schema import TextNode

from app.embeddings.embedding_registry import EmbeddingRegistry
from app.embeddings.embedding_schema import EmbeddedNode


class EmbeddingPipeline:
    """
    Executes one selected embedding provider.
    """

    def __init__(
        self,
        default_provider: str = "fastembed",
        registry: EmbeddingRegistry | None = None,
    ):

        self.registry = registry or EmbeddingRegistry()

        self.default_provider = default_provider

    def embed(
        self,
        nodes: TextNode | list[TextNode],
        provider: str | None = None,
    ) -> EmbeddedNode | list[EmbeddedNode]:
        """
        Embed one or many TextNodes.

        Parameters
        ----------
        nodes
            Single TextNode or list of TextNodes.

        provider
            Embedding provider.
            If omitted, default_provider is used.
        """

        provider = provider or self.default_provider

        embedder = self.registry.get_embedder(provider)

        if isinstance(nodes, TextNode):

            return embedder.embed(nodes)

        return embedder.embed_batch(nodes)