"""
base_embedder.py

Abstract base class for all embedding providers.

Responsibilities
----------------
1. Provide a unified public embedding API.
2. Validate input nodes.
3. Delegate embedding generation to provider implementations.
4. Keep downstream systems provider-agnostic.

Concrete implementations:
- HuggingFaceEmbedder
- FastEmbedEmbedder
- OllamaEmbedder
- OpenAIEmbedder

SupportMind AI
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from llama_index.core.schema import TextNode

from app.embeddings.embedding_schema import EmbeddedNode


class BaseEmbedder(ABC):
    """
    Base class for embedding providers.

    Subclasses only implement:
        _embed()
        _embed_batch()
        provider

    The public methods perform validation and expose
    a consistent interface.
    """

    def embed(
        self,
        node: TextNode,
    ) -> EmbeddedNode:
        """
        Embed a single TextNode.
        """
        self._validate_node(node)
        return self._embed(node)

    def embed_batch(
        self,
        nodes: List[TextNode],
    ) -> List[EmbeddedNode]:
        """
        Embed multiple TextNodes.
        """
        if not nodes:
            return []

        for node in nodes:
            self._validate_node(node)

        return self._embed_batch(nodes)

    @property
    @abstractmethod
    def provider(self) -> str:
        """
        Name of the embedding provider.
        """
        ...

    @abstractmethod
    def _embed(
        self,
        node: TextNode,
    ) -> EmbeddedNode:
        """
        Provider-specific single embedding.
        """
        ...

    @abstractmethod
    def _embed_batch(
        self,
        nodes: List[TextNode],
    ) -> List[EmbeddedNode]:
        """
        Provider-specific batch embedding.
        """
        ...

    @staticmethod
    def _validate_node(
        node: TextNode,
    ) -> None:
        """
        Validate input node.
        """
        if node is None:
            raise ValueError("TextNode cannot be None.")

        if not isinstance(node, TextNode):
            raise TypeError(
                f"Expected TextNode, got {type(node).__name__}."
            )

        if not node.text:
            raise ValueError("TextNode contains empty text.")
