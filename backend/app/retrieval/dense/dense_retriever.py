"""
dense_retriever.py

Production dense retrieval implementation for SupportMind AI.

Responsibilities
----------------
1. Accept a validated RetrievalRequest and query embedding.
2. Execute dense vector search through VectorPipeline.
3. Use the provider-agnostic VectorPipeline abstraction.
4. Convert VectorSearchResult into RetrievalResult.
5. Preserve retrieval scores and metadata.
6. Assign deterministic retrieval ranks.
7. Keep vector-store/provider details outside the retrieval layer.

Locked Technology Stack
-----------------------
Dense Retrieval : Qdrant
Vector Access   : VectorPipeline
Contracts       : Pydantic retrieval models

Architecture
------------
Query Router
     |
     v
DenseRetriever
     |
     v
VectorPipeline
     |
     v
VectorStoreRegistry
     |
     v
QdrantStore
"""

from __future__ import annotations

from numbers import Real
from typing import Any

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
)
from app.vector_store.vector_filters import VectorFilter
from app.vector_store.vector_pipeline import (
    VectorPipeline,
    vector_pipeline,
)


class DenseRetriever:
    """
    Production dense retriever backed by VectorPipeline.

    Provider-specific vector-store logic remains completely
    outside this class.
    """

    SOURCE = "dense"

    def __init__(
        self,
        pipeline: VectorPipeline | None = None,
    ) -> None:
        """
        Initialize the dense retriever.

        Parameters
        ----------
        pipeline:
            VectorPipeline used for vector search.

            If omitted, the application-level vector_pipeline
            is used.
        """

        self.pipeline = (
            pipeline
            if pipeline is not None
            else vector_pipeline
        )

    # ==========================================================
    # Retrieval
    # ==========================================================

    def retrieve(
        self,
        request: RetrievalRequest,
        query_vector: list[float],
    ) -> list[RetrievalResult]:
        """
        Execute dense vector retrieval.

        Provider resolution is delegated to VectorPipeline.

        Parameters
        ----------
        request:
            Validated retrieval request.

        query_vector:
            Query embedding used for dense similarity search.

        Returns
        -------
        list[RetrievalResult]
            Ordered dense retrieval results.

        Raises
        ------
        ValueError
            If request or query_vector is invalid.
        """

        self._validate_request(request)
        self._validate_query_vector(query_vector)

        filters = self._build_vector_filter(
            request.filters
        )

        vector_results = self.pipeline.search(
            collection_name=request.collection_name,
            query_vector=query_vector,
            top_k=request.top_k,
            filters=filters,
            provider_name=request.provider,
        )

        if not vector_results:
            return []

        return [
            self._convert_result(
                result=result,
                rank=rank,
            )
            for rank, result in enumerate(
                vector_results,
                start=1,
            )
        ]

    # ==========================================================
    # Result Conversion
    # ==========================================================

    @classmethod
    def _convert_result(
        cls,
        result: Any,
        rank: int,
    ) -> RetrievalResult:
        """
        Convert VectorSearchResult into RetrievalResult.

        Provider-specific fields are intentionally not exposed.
        """

        return RetrievalResult(
            id=result.id,
            node_id=result.node_id,
            document_id=result.document_id,
            text=result.text,
            score=float(result.score),
            rank=rank,
            source=cls.SOURCE,
            metadata=result.metadata,
            retrieval_scores={
                "dense_score": float(result.score),
            },
        )

    # ==========================================================
    # Filter Conversion
    # ==========================================================

    @staticmethod
    def _build_vector_filter(
        filters: dict[str, Any] | VectorFilter | None,
    ) -> VectorFilter | None:
        """
        Convert the provider-neutral request filter representation
        into the VectorPipeline VectorFilter contract.

        Supported forms
        ---------------
        None
            No filtering.

        {}
            No filtering.

        VectorFilter
            Already normalized; returned unchanged.

        dict
            Expected to contain VectorFilter-compatible data.
        """

        if filters is None:
            return None

        if isinstance(filters, VectorFilter):
            return filters

        if not isinstance(filters, dict):
            raise ValueError(
                "Retrieval request filters must be a dictionary "
                "or VectorFilter instance."
            )

        if not filters:
            return None

        try:
            return VectorFilter.model_validate(filters)
        except Exception as exc:
            raise ValueError(
                "Invalid retrieval filters."
            ) from exc

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_request(
        request: RetrievalRequest,
    ) -> None:
        """
        Validate the retrieval request before execution.
        """

        if request is None:
            raise ValueError(
                "Retrieval request cannot be None."
            )

        if not isinstance(
            request,
            RetrievalRequest,
        ):
            raise ValueError(
                "request must be a RetrievalRequest instance."
            )

    @staticmethod
    def _validate_query_vector(
        query_vector: list[float],
    ) -> None:
        """
        Validate the query embedding before sending it
        to VectorPipeline.
        """

        if query_vector is None:
            raise ValueError(
                "Query vector cannot be None."
            )

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
            isinstance(value, Real)
            and not isinstance(value, bool)
            for value in query_vector
        ):
            raise ValueError(
                "Query vector must contain only numeric values."
            )


# ==============================================================
# Application-level default
# ==============================================================

dense_retriever = DenseRetriever(
    pipeline=vector_pipeline,
)