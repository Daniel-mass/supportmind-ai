# backend/app/embeddings/ollama_embedder.py

"""
Production-grade Ollama Embedder.

Uses a locally running Ollama server to generate
dense embeddings for SupportMind AI.

Responsibilities
----------------
1. Embed a single node.
2. Embed batches efficiently.
3. Convert vectors into EmbeddedNode objects.
4. Remain provider-agnostic.

Author: SupportMind AI
"""

from __future__ import annotations

from ollama import Client

from app.embeddings.base_embedder import BaseEmbedder
from app.embeddings.embedding_schema import EmbeddedNode


class OllamaEmbedder(BaseEmbedder):
    """
    Local embedding provider using Ollama.

    Example models
    --------------
    bge-m3
    nomic-embed-text
    mxbai-embed-large
    """

    def __init__(
        self,
        model_name: str = "bge-m3:latest",
        host: str = "http://localhost:11434",
    ) -> None:

        self.model_name = model_name
        self.client = Client(host=host)

        self._dimension: int | None = None

    @property
    def model(self) -> str:
        return self.model_name
    @property
    def provider(self):
       return "ollama"

    @property
    def dimension(self) -> int:
        """
        Lazily determine embedding dimension.
        """

        if self._dimension is None:

            response = self.client.embed(
                model=self.model_name,
                input="dimension check",
            )

            self._dimension = len(response.embeddings[0])

        return self._dimension

    def _embed(self, node) -> EmbeddedNode:
        """
        Embed a single node.
        """

        response = self.client.embed(
            model=self.model_name,
            input=str(node.text),
        )

        vector = list(response.embeddings[0])

        return EmbeddedNode(

            node_id=node.node_id,

            document_id=node.metadata["document_id"],

            text=node.text,

            embedding=vector,

            embedding_model=self.model_name,

            embedding_dimension=len(vector),

            metadata=node.metadata.copy(),

        )

    def _embed_batch(
        self,
        nodes,
    ) -> list[EmbeddedNode]:
        """
        Batch embedding using Ollama.
        """

        texts = [
            str(node.text)
            for node in nodes
        ]

        response = self.client.embed(
            model=self.model_name,
            input=texts,
        )

        embedded_nodes = []

        for node, vector in zip(nodes, response.embeddings):

            embedded_nodes.append(

                EmbeddedNode(

                    node_id=node.node_id,

                    document_id=node.metadata["document_id"],

                    text=node.text,

                    embedding=list(vector),

                    embedding_model=self.model_name,

                    embedding_dimension=len(vector),

                    metadata=node.metadata.copy(),

                )

            )

        return embedded_nodes