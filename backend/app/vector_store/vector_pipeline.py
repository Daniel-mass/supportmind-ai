"""
vector_pipeline.py

Provider-agnostic orchestration layer for SupportMind AI vector storage.

Responsibilities
----------------
1. Resolve vector store providers through VectorStoreRegistry.
2. Orchestrate collection lifecycle operations.
3. Orchestrate vector insertion and batch insertion.
4. Orchestrate vector retrieval.
5. Orchestrate vector deletion.
6. Orchestrate similarity search.
7. Support provider-agnostic metadata filtering.
8. Expose vector collection statistics.
9. Expose provider health checks.
10. Keep higher application layers independent from Qdrant,
    Pinecone, Milvus, Weaviate, or any other provider.

Architecture
------------
Application / Retrieval Layer
            |
            v
      VectorPipeline
            |
            v
    VectorStoreRegistry
            |
            v
      BaseVectorStore
            |
            v
    Concrete Provider
    (QdrantStore, etc.)

The pipeline does NOT implement provider-specific logic.

SupportMind AI
"""

from __future__ import annotations

from typing import Any

from app.embeddings.embedding_schema import EmbeddedNode
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.exceptions import (
    VectorCollectionAlreadyExistsError,
    VectorCollectionNotFoundError,
    VectorConfigurationError,
    VectorStoreConnectionError,
)
from app.vector_store.vector_filters import VectorFilter
from app.vector_store.vector_registry import (
    VectorStoreRegistry,
    vector_registry,
)
from app.vector_store.vector_search_result import VectorSearchResult


class VectorPipeline:
    """
    Provider-agnostic orchestration layer for vector storage.

    The pipeline delegates all actual vector database operations
    to a registered BaseVectorStore implementation.

    Example
    -------
    pipeline = VectorPipeline()

    pipeline.register_provider(qdrant_store)

    pipeline.create_collection(
        collection_name="supportmind_documents",
        dimension=384,
    )

    pipeline.upsert(
        collection_name="supportmind_documents",
        node=embedded_node,
    )

    results = pipeline.search(
        collection_name="supportmind_documents",
        query_vector=query_embedding,
        top_k=5,
    )
    """

    def __init__(
        self,
        registry: VectorStoreRegistry | None = None,
        default_provider: str | None = None,
    ) -> None:
        """
        Initialize the vector pipeline.

        Parameters
        ----------
        registry:
            VectorStoreRegistry instance.

            If omitted, the application-level global
            vector_registry is used.

        default_provider:
            Optional provider name used when an operation
            does not explicitly specify a provider.

        Raises
        ------
        ValueError
            If default_provider is invalid.
        """

        self.registry = (
            registry
            if registry is not None
            else vector_registry
        )

        if default_provider is not None:

            self._validate_provider_name(
                default_provider
            )

        self._default_provider = (
            default_provider.strip().lower()
            if default_provider is not None
            else None
        )

    # ==========================================================
    # Provider Configuration
    # ==========================================================

    @property
    def default_provider(self) -> str | None:
        """
        Return the configured default provider name.

        Returns
        -------
        str | None
            Default provider name or None when no default
            provider has been configured.
        """

        return self._default_provider

    def set_default_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Set the default vector store provider.

        The provider must already be registered.

        Parameters
        ----------
        provider_name:
            Registered provider name.

        Raises
        ------
        ValueError
            If provider name is invalid.

        VectorConfigurationError
            If provider is not registered.
        """

        normalized_name = (
            self._validate_provider_name(
                provider_name
            )
        )

        if not self.registry.has_provider(
            normalized_name
        ):

            raise VectorConfigurationError(
                f"Cannot set default vector provider "
                f"'{normalized_name}'. "
                f"The provider is not registered."
            )

        self._default_provider = normalized_name

    def register_provider(
        self,
        provider: BaseVectorStore,
        make_default: bool = False,
    ) -> None:
        """
        Register a vector store provider with the pipeline.

        Parameters
        ----------
        provider:
            Concrete BaseVectorStore implementation.

        make_default:
            If True, make this provider the default provider.

        Raises
        ------
        ValueError
            If provider is invalid.

        UnsupportedVectorOperationError
            If provider is already registered.
        """

        if provider is None:

            raise ValueError(
                "Vector provider cannot be None."
            )

        if not isinstance(
            provider,
            BaseVectorStore,
        ):

            raise ValueError(
                "Vector provider must be an instance "
                "of BaseVectorStore."
            )

        self.registry.register(
            provider
        )

        if make_default:

            self._default_provider = (
                provider.provider
                .strip()
                .lower()
            )

    def unregister_provider(
        self,
        provider_name: str,
    ) -> None:
        """
        Unregister a vector store provider.

        If the provider is currently configured as
        the default provider, the default provider
        is cleared.

        Parameters
        ----------
        provider_name:
            Provider name to remove.
        """

        normalized_name = (
            self._validate_provider_name(
                provider_name
            )
        )

        self.registry.unregister(
            normalized_name
        )

        if (
            self._default_provider
            == normalized_name
        ):

            self._default_provider = None

    def list_providers(
        self,
    ) -> list[str]:
        """
        Return all registered vector store providers.
        """

        return self.registry.list_providers()

    # ==========================================================
    # Provider Resolution
    # ==========================================================

    def get_provider(
        self,
        provider_name: str | None = None,
    ) -> BaseVectorStore:
        """
        Resolve a vector store provider.

        Resolution order
        ----------------
        1. Explicit provider_name.
        2. Configured default provider.

        Parameters
        ----------
        provider_name:
            Optional explicit provider name.

        Returns
        -------
        BaseVectorStore
            Registered provider instance.

        Raises
        ------
        VectorConfigurationError
            If no provider is available.
        """

        resolved_name = self._resolve_provider_name(
            provider_name
        )

        provider = self.registry.get(
            resolved_name
        )

        if not isinstance(
            provider,
            BaseVectorStore,
        ):

            raise VectorConfigurationError(
                f"Registered provider "
                f"'{resolved_name}' "
                f"is not a valid BaseVectorStore implementation."
            )

        return provider

    # ==========================================================
    # Collection Management
    # ==========================================================

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        provider_name: str | None = None,
    ) -> None:
        """
        Create a vector collection.

        Delegates collection creation to the selected
        vector store provider.
        """

        self._validate_collection_name(
            collection_name
        )

        self._validate_dimension(
            dimension
        )

        provider = self.get_provider(
            provider_name
        )

        provider.create_collection(
            collection_name=collection_name.strip(),
            dimension=dimension,
        )

    def delete_collection(
        self,
        collection_name: str,
        provider_name: str | None = None,
    ) -> None:
        """
        Delete a vector collection.
        """

        self._validate_collection_name(
            collection_name
        )

        provider = self.get_provider(
            provider_name
        )

        provider.delete_collection(
            collection_name=collection_name.strip()
        )

    def collection_exists(
        self,
        collection_name: str,
        provider_name: str | None = None,
    ) -> bool:
        """
        Check whether a collection exists.
        """

        self._validate_collection_name(
            collection_name
        )

        provider = self.get_provider(
            provider_name
        )

        return provider.collection_exists(
            collection_name.strip()
        )

    # ==========================================================
    # Insert / Update
    # ==========================================================

    def upsert(
        self,
        collection_name: str,
        node: EmbeddedNode,
        provider_name: str | None = None,
    ) -> None:
        """
        Insert or update a single embedded node.
        """

        self._validate_collection_name(
            collection_name
        )

        if node is None:

            raise ValueError(
                "EmbeddedNode cannot be None."
            )

        provider = self.get_provider(
            provider_name
        )

        provider.upsert(
            collection_name=collection_name.strip(),
            node=node,
        )

    def upsert_batch(
        self,
        collection_name: str,
        nodes: list[EmbeddedNode],
        provider_name: str | None = None,
    ) -> None:
        """
        Insert or update multiple embedded nodes.

        Empty node lists are delegated to the provider,
        allowing provider-specific no-op behavior.
        """

        self._validate_collection_name(
            collection_name
        )

        if nodes is None:

            raise ValueError(
                "Nodes cannot be None."
            )

        if not isinstance(
            nodes,
            list,
        ):

            raise ValueError(
                "Nodes must be provided as a list."
            )

        if any(
            node is None
            for node in nodes
        ):

            raise ValueError(
                "upsert_batch cannot contain None nodes."
            )

        provider = self.get_provider(
            provider_name
        )

        provider.upsert_batch(
            collection_name=collection_name.strip(),
            nodes=nodes,
        )

    # ==========================================================
    # Retrieval
    # ==========================================================

    def get(
        self,
        collection_name: str,
        ids: list[str],
        provider_name: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        Retrieve vectors by their IDs.
        """

        self._validate_collection_name(
            collection_name
        )

        if ids is None:

            raise ValueError(
                "Vector IDs cannot be None."
            )

        if not isinstance(
            ids,
            list,
        ):

            raise ValueError(
                "Vector IDs must be provided as a list."
            )

        provider = self.get_provider(
            provider_name
        )

        return provider.get(
            collection_name=collection_name.strip(),
            ids=ids,
        )

    # ==========================================================
    # Deletion
    # ==========================================================

    def delete(
        self,
        collection_name: str,
        node_id: str,
        provider_name: str | None = None,
    ) -> None:
        """
        Delete a single vector.
        """

        self._validate_collection_name(
            collection_name
        )

        self._validate_node_id(
            node_id
        )

        provider = self.get_provider(
            provider_name
        )

        provider.delete(
            collection_name=collection_name.strip(),
            node_id=node_id.strip(),
        )

    def delete_batch(
        self,
        collection_name: str,
        ids: list[str],
        provider_name: str | None = None,
    ) -> None:
        """
        Delete multiple vectors.
        """

        self._validate_collection_name(
            collection_name
        )

        if ids is None:

            raise ValueError(
                "Vector IDs cannot be None."
            )

        if not isinstance(
            ids,
            list,
        ):

            raise ValueError(
                "Vector IDs must be provided as a list."
            )

        provider = self.get_provider(
            provider_name
        )

        provider.delete_batch(
            collection_name=collection_name.strip(),
            ids=ids,
        )

    # ==========================================================
    # Similarity Search
    # ==========================================================

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: VectorFilter | None = None,
        provider_name: str | None = None,
    ) -> list[VectorSearchResult]:
        """
        Perform provider-agnostic vector similarity search.

        Parameters
        ----------
        collection_name:
            Vector collection to search.

        query_vector:
            Query embedding.

        top_k:
            Maximum number of results.

        filters:
            Optional provider-agnostic metadata filters.

        provider_name:
            Optional provider override.
        """

        self._validate_collection_name(
            collection_name
        )

        self._validate_query_vector(
            query_vector
        )

        if not isinstance(
            top_k,
            int,
        ):

            raise ValueError(
                "top_k must be an integer."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        if filters is not None and not isinstance(
            filters,
            VectorFilter,
        ):

            raise ValueError(
                "filters must be a VectorFilter "
                "instance or None."
            )

        provider = self.get_provider(
            provider_name
        )

        return provider.search(
            collection_name=collection_name.strip(),
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

    # ==========================================================
    # Statistics
    # ==========================================================

    def count(
        self,
        collection_name: str,
        provider_name: str | None = None,
    ) -> int:
        """
        Return the number of vectors in a collection.
        """

        self._validate_collection_name(
            collection_name
        )

        provider = self.get_provider(
            provider_name
        )

        return provider.count(
            collection_name.strip()
        )

    # ==========================================================
    # Health
    # ==========================================================

    def health_check(
        self,
        provider_name: str | None = None,
    ) -> bool:
        """
        Check whether the selected vector store provider
        is reachable.

        Provider health_check implementations are expected
        to return False on connectivity failure rather
        than raising exceptions.
        """

        provider = self.get_provider(
            provider_name
        )

        return provider.health_check()

    # ==========================================================
    # Provider Health
    # ==========================================================

    def health_check_all(
        self,
    ) -> dict[str, bool]:
        """
        Run health checks for every registered provider.

        Returns
        -------
        dict[str, bool]
            Mapping of provider name to health status.

        A failure in one provider does not prevent
        health checks for other providers.
        """

        results: dict[str, bool] = {}

        for provider_name in self.registry.list_providers():

            try:

                provider = self.registry.get(
                    provider_name
                )

                results[
                    provider_name
                ] = bool(
                    provider.health_check()
                )

            except Exception:

                results[
                    provider_name
                ] = False

        return results

    # ==========================================================
    # Internal Provider Resolution
    # ==========================================================

    def _resolve_provider_name(
        self,
        provider_name: str | None,
    ) -> str:
        """
        Resolve the effective provider name.

        Explicit provider name takes priority over
        the configured default provider.
        """

        if provider_name is not None:

            return self._validate_provider_name(
                provider_name
            )

        if self._default_provider is None:

            raise VectorConfigurationError(
                "No vector store provider was specified "
                "and no default provider is configured."
            )

        return self._default_provider

    # ==========================================================
    # Validation Helpers
    # ==========================================================

    @staticmethod
    def _validate_provider_name(
        provider_name: str,
    ) -> str:
        """
        Validate and normalize provider names.
        """

        if not isinstance(
            provider_name,
            str,
        ):

            raise ValueError(
                "Provider name must be a string."
            )

        normalized_name = (
            provider_name.strip().lower()
        )

        if not normalized_name:

            raise ValueError(
                "Provider name cannot be empty."
            )

        return normalized_name

    @staticmethod
    def _validate_collection_name(
        collection_name: str,
    ) -> None:
        """
        Validate collection names.
        """

        if not isinstance(
            collection_name,
            str,
        ):

            raise ValueError(
                "Collection name must be a string."
            )

        if not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

    @staticmethod
    def _validate_dimension(
        dimension: int,
    ) -> None:
        """
        Validate vector dimensions.
        """

        if not isinstance(
            dimension,
            int,
        ):

            raise ValueError(
                "Vector dimension must be an integer."
            )

        if dimension <= 0:

            raise ValueError(
                "Vector dimension must be greater than zero."
            )

    @staticmethod
    def _validate_node_id(
        node_id: str,
    ) -> None:
        """
        Validate a vector node ID.
        """

        if not isinstance(
            node_id,
            str,
        ):

            raise ValueError(
                "Node ID must be a string."
            )

        if not node_id.strip():

            raise ValueError(
                "Node ID cannot be empty."
            )

    @staticmethod
    def _validate_query_vector(
        query_vector: list[float],
    ) -> None:
        """
        Validate a query embedding.
        """

        if not isinstance(
            query_vector,
            list,
        ):

            raise ValueError(
                "Query vector must be provided as a list."
            )

        if not query_vector:

            raise ValueError(
                "Query vector cannot be empty."
            )

        if not all(
            isinstance(
                value,
                (int, float),
            )
            for value in query_vector
        ):

            raise ValueError(
                "Query vector must contain only numeric values."
            )


# ==============================================================
# Global Pipeline
# ==============================================================

vector_pipeline = VectorPipeline(
    registry=vector_registry,
)
"""
Application-level default vector pipeline.

The application can register providers during startup:

    vector_pipeline.register_provider(
        qdrant_store,
        make_default=True,
    )

After that, higher layers can use:

    vector_pipeline.search(...)
    vector_pipeline.upsert(...)
    vector_pipeline.get(...)

without knowing which vector database is being used.
"""