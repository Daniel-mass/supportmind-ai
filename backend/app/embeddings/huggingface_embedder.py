"""
huggingface_embedder.py

Production-grade HuggingFace embedding provider.

Responsibilities
----------------
1. Load SentenceTransformer once.
2. Generate dense embeddings.
3. Preserve node metadata.
4. Return EmbeddedNode objects.

SupportMind AI
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from llama_index.core.schema import TextNode

from app.embeddings.base_embedder import BaseEmbedder
from app.embeddings.embedding_schema import EmbeddedNode


class HuggingFaceEmbedder(BaseEmbedder):
    """
    HuggingFace SentenceTransformer embedding provider.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        normalize_embeddings: bool = True,
    ) -> None:

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        self.model = SentenceTransformer(model_name)
    @property
    def provider(self):
       return "huggingface"

    @property
    def dimension(self) -> int:
        """
        Embedding dimension.
        """

        return self.model.get_sentence_embedding_dimension()

    def _embed(
        self,
        node: TextNode,
    ) -> EmbeddedNode:
        """
        Embed a single node.

        Reuses the optimized batch implementation.
        """

        return self._embed_batch([node])[0]

    def _embed_batch(
        self,
        nodes: list[TextNode],
    ) -> list[EmbeddedNode]:
        """
        Embed multiple nodes.
        """

        texts = [node.text for node in nodes]

        vectors = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embedded_nodes: list[EmbeddedNode] = []

        for node, vector in zip(nodes, vectors):

            embedded_nodes.append(

                EmbeddedNode(

                    node_id=node.node_id,

                    document_id=node.metadata.get(
                        "document_id",
                        "",
                    ),

                    text=node.text,

                    embedding=vector.tolist(),

                    embedding_model=self.model_name,

                    embedding_dimension=len(vector),

                    metadata=node.metadata.copy(),

                )

            )

        return embedded_nodes