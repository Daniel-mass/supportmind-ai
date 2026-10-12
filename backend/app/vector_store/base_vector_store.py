"""
base_vector_store.py

Abstract interface for every vector database supported by SupportMind AI.

This contract ensures the application remains provider-agnostic.
Concrete implementations (Qdrant, Milvus, Pinecone, Weaviate, etc.)
should implement only provider-specific logic.

Responsibilities
----------------
1. Collection management
2. Insert / Update vectors
3. Retrieve vectors
4. Delete vectors
5. Similarity search
6. Collection statistics
7. Health checks

SupportMind AI
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.embeddings.embedding_schema import EmbeddedNode
from app.vector_store.vector_filters import VectorFilter
from app.vector_store.vector_search_result import VectorSearchResult


class BaseVectorStore(ABC):
    """
    Provider-agnostic interface for vector databases.
    """

    # ==========================================================
    # Provider
    # ==========================================================

    @property
    @abstractmethod
    def provider(self) -> str:
        """
        Name of the vector database provider.

        Examples
        --------
        qdrant
        milvus
        pinecone
        weaviate
        """
        ...

    # ==========================================================
    # Collection Management
    # ==========================================================

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:
        """
        Create a collection.
        """
        ...

    @abstractmethod
    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        Delete a collection.
        """
        ...

    @abstractmethod
    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """
        Returns True if collection exists.
        """
        ...

    # ==========================================================
    # Insert / Update
    # ==========================================================

    @abstractmethod
    def upsert(
        self,
        collection_name: str,
        node: EmbeddedNode,
    ) -> None:
        """
        Insert or update a single vector.
        """
        ...

    @abstractmethod
    def upsert_batch(
        self,
        collection_name: str,
        nodes: list[EmbeddedNode],
    ) -> None:
        """
        Insert or update multiple vectors.
        """
        ...

    # ==========================================================
    # Retrieval
    # ==========================================================

    @abstractmethod
    def get(
        self,
        collection_name: str,
        ids: list[str],
    ) -> list[VectorSearchResult]:
        """
        Retrieve vectors by IDs.
        """
        ...

    # ==========================================================
    # Deletion
    # ==========================================================

    @abstractmethod
    def delete(
        self,
        collection_name: str,
        node_id: str,
    ) -> None:
        """
        Delete a single vector.
        """
        ...

    @abstractmethod
    def delete_batch(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """
        Delete multiple vectors.
        """
        ...

    # ==========================================================
    # Search
    # ==========================================================

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: VectorFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Perform vector similarity search.
        """
        ...

    # ==========================================================
    # Statistics
    # ==========================================================

    @abstractmethod
    def count(
        self,
        collection_name: str,
    ) -> int:
        """
        Return total number of vectors.
        """
        ...

    # ==========================================================
    # Health
    # ==========================================================

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify connectivity with the vector database.
        """
        ...