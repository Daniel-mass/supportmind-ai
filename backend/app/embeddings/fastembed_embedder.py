"""
fastembed_embedder.py

Production-grade FastEmbed implementation.

Uses FastEmbed + ONNX Runtime for
high-speed CPU embedding generation.

SupportMind AI
"""

from __future__ import annotations

from typing import List

from fastembed import TextEmbedding
from llama_index.core.schema import TextNode

from app.embeddings.base_embedder import BaseEmbedder
from app.embeddings.embedding_schema import EmbeddedNode
import os

class FastEmbedder(BaseEmbedder):
    """
    FastEmbed provider.

    Optimized for CPU inference.

    Suitable for:

    • Semantic Search
    • RAG
    • Large-scale ingestion
    • Production CPU deployments
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):

        self.model_name = model_name

        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=os.getenv("FASTEMBED_CACHE_DIR"),
        )

        self._dimension = None
    @property
    def provider(self):
       return "fastembed"

    @property
    def embedding_dimension(self) -> int:

        if self._dimension is None:

            vector = list(
                self.model.embed(
                    ["dimension test"]
                )
            )[0]

            self._dimension = len(vector)

        return self._dimension

    def _embed(
        self,
        node: TextNode,
    ) -> EmbeddedNode:

        text = str(node.text)

        vector = list(
            self.model.embed(
                [text]
            )
        )[0]

        return EmbeddedNode(

            node_id=node.node_id,

            document_id=node.metadata.get(
                "document_id",
                "",
            ),

            text=text,

            embedding=vector.tolist(),

            embedding_model=self.model_name,

            embedding_dimension=len(vector),

            metadata=node.metadata.copy(),

        )

    def _embed_batch(
        self,
        nodes: List[TextNode],
    ) -> List[EmbeddedNode]:

        texts = [

            str(node.text)

            for node in nodes

        ]

        vectors = list(

            self.model.embed(
                texts
            )

        )

        embedded_nodes = []

        for node, vector in zip(
            nodes,
            vectors,
        ):

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