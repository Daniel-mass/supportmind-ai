"""
qdrant_store.py

Production-grade Qdrant implementation of BaseVectorStore.

Responsibilities
----------------
1. Manage Qdrant collections
2. Insert / update vectors
3. Retrieve vectors
4. Delete vectors
5. Similarity search
6. Metadata filtering
7. Vector dimension validation
8. Health checks

SupportMind AI
"""

from __future__ import annotations

import json
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.embeddings.embedding_schema import EmbeddedNode
from app.vector_store.base_vector_store import BaseVectorStore
from app.vector_store.exceptions import (
    InvalidVectorDimensionError,
    InvalidVectorFilterError,
    UnsupportedVectorOperationError,
    VectorCollectionAlreadyExistsError,
    VectorCollectionNotFoundError,
    VectorDeletionError,
    VectorInsertionError,
    VectorSearchError,
    VectorStoreConnectionError,
)
from app.vector_store.vector_filters import VectorFilter
from app.vector_store.vector_search_result import VectorSearchResult


class QdrantStore(BaseVectorStore):
    """
    Production implementation of BaseVectorStore
    backed by Qdrant.

    Supports dependency injection of an existing QdrantClient
    instance for unit testing and integration testing.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        api_key: str | None = None,
        https: bool = False,
        timeout: int = 30,
        client: QdrantClient | None = None,
    ) -> None:
        """
        Initialize QdrantStore.

        Parameters
        ----------
        host:
            Qdrant host.

        port:
            Qdrant HTTP port.

        api_key:
            Optional Qdrant API key.

        https:
            Whether HTTPS should be used.

        timeout:
            Qdrant request timeout.

        client:
            Optional preconfigured QdrantClient.

            If provided, the injected client is used directly.
            This is useful for testing and dependency injection.
        """

        # Dependency injection path.
        # Used by unit tests and integration tests.
        if client is not None:

            self.client = client

            return

        # Production initialization path.
        try:

            self.client = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                https=https,
                timeout=timeout,
            )

        except Exception as exc:

            raise VectorStoreConnectionError(
                f"Unable to initialize Qdrant client: {exc}"
            ) from exc

    # ==========================================================
    # Provider
    # ==========================================================

    @property
    def provider(self) -> str:
        """
        Return the vector store provider name.
        """

        return "qdrant"

    # ==========================================================
    # Private Helpers
    # ==========================================================

    @staticmethod
    def _build_payload(
        node: EmbeddedNode,
    ) -> dict[str, Any]:
        """
        Convert EmbeddedNode into a Qdrant payload.
        """

        return {
            "node_id": node.node_id,
            "document_id": node.document_id,
            "text": node.text,
            "embedding_model": node.embedding_model,
            "embedding_dimension": node.embedding_dimension,
            "metadata": node.metadata,
            "created_at": node.created_at.isoformat(),
        }

    @staticmethod
    def _build_point(
        node: EmbeddedNode,
    ) -> models.PointStruct:
        """
        Convert EmbeddedNode into a Qdrant PointStruct.
        """

        return models.PointStruct(
            id=node.node_id,
            vector=node.embedding,
            payload=QdrantStore._build_payload(node),
        )

    @staticmethod
    def _convert_result(
        point: models.ScoredPoint,
        collection_name: str,
    ) -> VectorSearchResult:
        """
        Convert a Qdrant search result into the
        provider-agnostic VectorSearchResult schema.
        """

        payload = point.payload or {}

        return VectorSearchResult(
            id=str(point.id),
            node_id=payload.get(
                "node_id",
                str(point.id),
            ),
            document_id=payload.get(
                "document_id",
                "",
            ),
            text=payload.get(
                "text",
                "",
            ),
            score=float(point.score),
            embedding_model=payload.get(
                "embedding_model",
                "",
            ),
            provider="qdrant",
            collection_name=collection_name,
            metadata=payload.get(
                "metadata",
                {},
            ),
        )

    @staticmethod
    def _build_filter(
        filters: VectorFilter | None,
    ) -> models.Filter | None:
        """
        Convert provider-agnostic VectorFilter
        into a Qdrant Filter.

        All filter conditions are combined using AND logic.
        """

        if filters is None or filters.is_empty():
            return None

        conditions: list[models.FieldCondition] = []

        for condition in filters.conditions:

            if condition.operator.value == "eq":

                conditions.append(
                    models.FieldCondition(
                        key=f"metadata.{condition.field}",
                        match=models.MatchValue(
                            value=condition.value,
                        ),
                    )
                )

            else:

                raise InvalidVectorFilterError(
                    f"Unsupported filter operator for Qdrant: "
                    f"{condition.operator}"
                )

        if not conditions:
            return None

        return models.Filter(
            must=conditions,
        )

    # ==========================================================
    # Private Vector Validation Helpers
    # ==========================================================

    def _get_collection_dimension(
        self,
        collection_name: str,
    ) -> int:
        """
        Retrieve the configured vector dimension for a
        Qdrant collection.

        This performs exactly one collection lookup and is
        intended to be reused when validating multiple nodes.
        """

        try:

            collection_info = self.client.get_collection(
                collection_name=collection_name,
            )

            vectors_config = (
                collection_info
                .config
                .params
                .vectors
            )

            # Standard single-vector collection.
            if isinstance(
                vectors_config,
                models.VectorParams,
            ):

                return int(
                    vectors_config.size
                )

            raise UnsupportedVectorOperationError(
                "Named or multi-vector Qdrant collections "
                "are not supported by QdrantStore."
            )

        except UnsupportedVectorOperationError:

            raise

        except Exception as exc:

            raise VectorStoreConnectionError(
                f"Failed to retrieve vector configuration "
                f"for Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    def _validate_vector_dimension(
        self,
        collection_name: str,
        node: EmbeddedNode,
    ) -> None:
        """
        Validate a single EmbeddedNode against the
        Qdrant collection dimension.

        This method performs a collection lookup.

        For batch operations, use _validate_nodes_dimension()
        instead so the collection configuration is retrieved
        only once.
        """

        expected_dimension = (
            self._get_collection_dimension(
                collection_name
            )
        )

        actual_dimension = len(
            node.embedding
        )

        if (
            actual_dimension
            != expected_dimension
        ):

            raise InvalidVectorDimensionError(
                f"Embedding dimension mismatch for "
                f"collection '{collection_name}'. "
                f"Expected {expected_dimension}, "
                f"received {actual_dimension}."
            )

        if (
            node.embedding_dimension
            != actual_dimension
        ):

            raise InvalidVectorDimensionError(
                f"EmbeddedNode dimension metadata mismatch. "
                f"embedding_dimension is "
                f"{node.embedding_dimension}, "
                f"but actual embedding length is "
                f"{actual_dimension}."
            )

    def _validate_nodes_dimension(
        self,
        collection_name: str,
        nodes: list[EmbeddedNode],
    ) -> None:
        """
        Validate all nodes in a batch against the Qdrant
        collection dimension.

        The collection configuration is retrieved exactly
        once instead of once per node.
        """

        if not nodes:

            return

        expected_dimension = (
            self._get_collection_dimension(
                collection_name
            )
        )

        for node in nodes:

            actual_dimension = len(
                node.embedding
            )

            if (
                actual_dimension
                != expected_dimension
            ):

                raise InvalidVectorDimensionError(
                    f"Embedding dimension mismatch for "
                    f"collection '{collection_name}'. "
                    f"Expected {expected_dimension}, "
                    f"received {actual_dimension} "
                    f"for node '{node.node_id}'."
                )

            if (
                node.embedding_dimension
                != actual_dimension
            ):

                raise InvalidVectorDimensionError(
                    f"EmbeddedNode dimension metadata mismatch "
                    f"for node '{node.node_id}'. "
                    f"embedding_dimension is "
                    f"{node.embedding_dimension}, "
                    f"but actual embedding length is "
                    f"{actual_dimension}."
                )

    # ==========================================================
    # Qdrant Batch Size Helpers
    # ==========================================================

    # Qdrant's server-side JSON request limit is 32 MiB.
    #
    # We intentionally stay well below that limit to leave room
    # for HTTP/JSON overhead and future payload changes.
    MAX_BATCH_PAYLOAD_BYTES = 16 * 1024 * 1024

    # Safety limit on the number of points in a single request.
    MAX_BATCH_POINTS = 64

    @staticmethod
    def _estimate_point_size(
        point: models.PointStruct,
    ) -> int:
        """
        Estimate the JSON size of one Qdrant point.

        This is intentionally conservative and is used only
        to prevent oversized HTTP requests.
        """

        point_data = {
            "id": point.id,
            "vector": point.vector,
            "payload": point.payload or {},
        }

        encoded = json.dumps(
            point_data,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode(
            "utf-8"
        )

        return len(encoded)

    def _split_points_into_batches(
        self,
        points: list[models.PointStruct],
    ) -> list[list[models.PointStruct]]:
        """
        Split Qdrant points into safe request batches.

        A batch is limited by both:

        1. Maximum estimated JSON payload size.
        2. Maximum number of points.

        This prevents large documents or large metadata payloads
        from exceeding Qdrant's HTTP request limit.
        """

        if not points:

            return []

        batches: list[list[models.PointStruct]] = []

        current_batch: list[models.PointStruct] = []
        current_size = 2  # JSON array brackets

        for point in points:

            point_size = (
                self._estimate_point_size(
                    point
                )
            )

            # --------------------------------------------------
            # A single point larger than the configured safe
            # batch size cannot be solved by splitting the
            # batch. It needs payload reduction.
            # --------------------------------------------------

            if point_size > self.MAX_BATCH_PAYLOAD_BYTES:

                raise VectorInsertionError(
                    f"Single Qdrant point "
                    f"'{point.id}' is approximately "
                    f"{point_size / (1024 * 1024):.2f} MB, "
                    f"which exceeds the configured maximum "
                    f"batch payload of "
                    f"{self.MAX_BATCH_PAYLOAD_BYTES / (1024 * 1024):.0f} MB. "
                    f"Reduce the payload stored with the point."
                )

            # --------------------------------------------------
            # Start a new batch when either limit is reached.
            # --------------------------------------------------

            exceeds_size = (
                current_batch
                and (
                    current_size
                    + point_size
                    + 1
                    > self.MAX_BATCH_PAYLOAD_BYTES
                )
            )

            exceeds_count = (
                current_batch
                and len(current_batch)
                >= self.MAX_BATCH_POINTS
            )

            if (
                exceeds_size
                or exceeds_count
            ):

                batches.append(
                    current_batch
                )

                current_batch = []

                current_size = 2

            current_batch.append(
                point
            )

            current_size += (
                point_size
                + 1
            )

        if current_batch:

            batches.append(
                current_batch
            )

        return batches

    # ==========================================================
    # Collection Management
    # ==========================================================

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
    ) -> None:
        """
        Create a Qdrant collection using cosine similarity.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        if dimension <= 0:

            raise ValueError(
                "Embedding dimension must be greater than zero."
            )

        collection_name = collection_name.strip()

        if self.collection_exists(
            collection_name
        ):

            raise VectorCollectionAlreadyExistsError(
                f"Qdrant collection "
                f"'{collection_name}' already exists."
            )

        try:

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance.COSINE,
                ),
            )

        except VectorCollectionAlreadyExistsError:

            raise

        except Exception as exc:

            raise UnsupportedVectorOperationError(
                f"Failed to create Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        """
        Delete a Qdrant collection.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        collection_name = collection_name.strip()

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        try:

            self.client.delete_collection(
                collection_name=collection_name,
            )

        except Exception as exc:

            raise UnsupportedVectorOperationError(
                f"Failed to delete Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """
        Check whether a Qdrant collection exists.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        collection_name = collection_name.strip()

        try:

            return bool(
                self.client.collection_exists(
                    collection_name,
                )
            )

        except Exception as exc:

            raise VectorStoreConnectionError(
                f"Failed to check existence of Qdrant "
                f"collection '{collection_name}': {exc}"
            ) from exc

    # ==========================================================
    # Insert / Update
    # ==========================================================

    def upsert(
        self,
        collection_name: str,
        node: EmbeddedNode,
    ) -> None:
        """
        Insert or update a single embedded node.

        Qdrant uses node_id as the point ID.

        If the point already exists, Qdrant replaces
        the existing point.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        if node is None:

            raise ValueError(
                "EmbeddedNode cannot be None."
            )

        collection_name = collection_name.strip()

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        self._validate_vector_dimension(
            collection_name=collection_name,
            node=node,
        )

        try:

            point = self._build_point(
                node
            )

            self.client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True,
            )

        except (
            VectorCollectionNotFoundError,
            InvalidVectorDimensionError,
        ):

            raise

        except Exception as exc:

            raise VectorInsertionError(
                f"Failed to upsert node "
                f"'{node.node_id}' into Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    def upsert_batch(
        self,
        collection_name: str,
        nodes: list[EmbeddedNode],
    ) -> None:
        """
        Insert or update multiple embedded nodes.

        Large batches are automatically split into safe
        Qdrant requests based on estimated JSON payload size
        and point count.

        The collection vector dimension is retrieved once
        for the entire batch.

        An empty node list is treated as a no-op.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        collection_name = collection_name.strip()

        if not nodes:

            return

        if any(
            node is None
            for node in nodes
        ):

            raise ValueError(
                "upsert_batch cannot contain None nodes."
            )

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        # ------------------------------------------------------
        # Validate dimensions.
        #
        # IMPORTANT:
        # This performs ONE get_collection() call instead of
        # one call for every node.
        # ------------------------------------------------------

        self._validate_nodes_dimension(
            collection_name=collection_name,
            nodes=nodes,
        )

        try:

            # --------------------------------------------------
            # Build all points.
            # --------------------------------------------------

            points = [
                self._build_point(node)
                for node in nodes
            ]

            # --------------------------------------------------
            # Split into safe HTTP payloads.
            # --------------------------------------------------

            batches = (
                self._split_points_into_batches(
                    points
                )
            )

            print(
                f"[QDRANT] Upserting "
                f"{len(points)} points in "
                f"{len(batches)} batches."
            )

            # --------------------------------------------------
            # Send each batch separately.
            # --------------------------------------------------

            for index, batch in enumerate(
                batches,
                start=1,
            ):

                batch_size_mb = (
                    sum(
                        self._estimate_point_size(
                            point
                        )
                        for point in batch
                    )
                    / (1024 * 1024)
                )

                print(
                    f"[QDRANT] Batch "
                    f"{index}/{len(batches)}: "
                    f"{len(batch)} points, "
                    f"~{batch_size_mb:.2f} MB"
                )

                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True,
                )

            print(
                f"[QDRANT] Successfully upserted "
                f"{len(points)} points."
            )

        except InvalidVectorDimensionError:

            raise

        except VectorInsertionError:

            raise

        except Exception as exc:

            raise VectorInsertionError(
                f"Failed to batch upsert "
                f"{len(nodes)} nodes into Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    # ==========================================================
    # Retrieval
    # ==========================================================

    def get(
        self,
        collection_name: str,
        ids: list[str],
    ) -> list[VectorSearchResult]:
        """
        Retrieve vectors by their point IDs.

        Qdrant retrieve does not return similarity scores,
        so score is represented as 0.0.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        collection_name = collection_name.strip()

        if not ids:

            return []

        if any(
            not isinstance(point_id, str)
            or not point_id.strip()
            for point_id in ids
        ):

            raise ValueError(
                "All vector IDs must be "
                "non-empty strings."
            )

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        try:

            points = self.client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False,
            )

            results: list[VectorSearchResult] = []

            for point in points:

                payload = point.payload or {}

                results.append(
                    VectorSearchResult(
                        id=str(point.id),
                        node_id=payload.get(
                            "node_id",
                            str(point.id),
                        ),
                        document_id=payload.get(
                            "document_id",
                            "",
                        ),
                        text=payload.get(
                            "text",
                            "",
                        ),
                        score=0.0,
                        embedding_model=payload.get(
                            "embedding_model",
                            "",
                        ),
                        provider="qdrant",
                        collection_name=collection_name,
                        metadata=payload.get(
                            "metadata",
                            {},
                        ),
                    )
                )

            return results

        except Exception as exc:

            raise VectorSearchError(
                f"Failed to retrieve vectors from "
                f"Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    # ==========================================================
    # Deletion
    # ==========================================================

    def delete(
        self,
        collection_name: str,
        node_id: str,
    ) -> None:
        """
        Delete a single vector using its node ID.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        if not node_id or not node_id.strip():

            raise ValueError(
                "Node ID cannot be empty."
            )

        collection_name = collection_name.strip()

        node_id = node_id.strip()

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        try:

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[node_id],
                ),
                wait=True,
            )

        except Exception as exc:

            raise VectorDeletionError(
                f"Failed to delete node "
                f"'{node_id}' from Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    def delete_batch(
        self,
        collection_name: str,
        ids: list[str],
    ) -> None:
        """
        Delete multiple vectors using their point IDs.

        An empty ID list is treated as a no-op.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        collection_name = collection_name.strip()

        if not ids:

            return

        if any(
            not isinstance(point_id, str)
            or not point_id.strip()
            for point_id in ids
        ):

            raise ValueError(
                "All vector IDs must be "
                "non-empty strings."
            )

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        try:

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=ids,
                ),
                wait=True,
            )

        except Exception as exc:

            raise VectorDeletionError(
                f"Failed to batch delete "
                f"{len(ids)} vectors from Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    # ==========================================================
    # Similarity Search
    # ==========================================================

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filters: VectorFilter | None = None,
    ) -> list[VectorSearchResult]:
        """
        Perform vector similarity search using Qdrant.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        if not query_vector:

            raise ValueError(
                "Query vector cannot be empty."
            )

        if not all(
            isinstance(value, (int, float))
            for value in query_vector
        ):

            raise ValueError(
                "Query vector must contain only numeric values."
            )

        if top_k <= 0:

            raise ValueError(
                "top_k must be greater than zero."
            )

        collection_name = collection_name.strip()

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        try:

            query_filter = self._build_filter(
                filters
            )

            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )

            return [
                self._convert_result(
                    point=point,
                    collection_name=collection_name,
                )
                for point in response.points
            ]

        except (
            InvalidVectorFilterError,
        ):

            raise

        except Exception as exc:

            raise VectorSearchError(
                f"Failed to search Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    # ==========================================================
    # Statistics
    # ==========================================================

    def count(
        self,
        collection_name: str,
    ) -> int:
        """
        Return the total number of vectors stored
        in a Qdrant collection.
        """

        if not collection_name or not collection_name.strip():

            raise ValueError(
                "Collection name cannot be empty."
            )

        collection_name = collection_name.strip()

        if not self.collection_exists(
            collection_name
        ):

            raise VectorCollectionNotFoundError(
                f"Qdrant collection "
                f"'{collection_name}' does not exist."
            )

        try:

            result = self.client.count(
                collection_name=collection_name,
                exact=True,
            )

            return int(result.count)

        except Exception as exc:

            raise UnsupportedVectorOperationError(
                f"Failed to count vectors in "
                f"Qdrant collection "
                f"'{collection_name}': {exc}"
            ) from exc

    # ==========================================================
    # Health Check
    # ==========================================================

    def health_check(
        self,
    ) -> bool:
        """
        Verify that Qdrant is reachable.

        Returns False instead of raising an exception.
        """

        try:

            self.client.get_collections()

            return True

        except Exception:

            return False