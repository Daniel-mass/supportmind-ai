"""
hybrid_retriever.py

Production hybrid retrieval orchestration for SupportMind AI.

Responsibilities
----------------
1. Execute dense retrieval.
2. Execute sparse BM25 retrieval.
3. Preserve independent retrieval result lists.
4. Validate the retrieval request.
5. Support independent dense/sparse top-k configuration.
6. Return provider-agnostic RetrievalResult objects.
7. Remain independent of Qdrant/BM25 implementation details.

Locked Technology Stack
-----------------------
Dense Retrieval   : Qdrant through DenseRetriever
Sparse Retrieval  : rank-bm25 through SparseRetriever
Fusion            : RRF handled by rrf_fusion.py
Contracts         : Pydantic retrieval models

Architecture
------------
Query Router
     |
     v
HybridRetriever
     |
     +--------------------+
     |                    |
     v                    v
DenseRetriever     SparseRetriever
     |                    |
     v                    v
Qdrant                 BM25
     |                    |
     +---------+----------+
               |
               v
          RRF Fusion
"""

from __future__ import annotations

from app.retrieval.dense.dense_retriever import DenseRetriever
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
)
from app.retrieval.sparse.sparse_retriever import SparseRetriever


class HybridRetriever:
    """
    Production hybrid retrieval orchestrator.

    HybridRetriever combines dense and sparse retrieval
    candidates but does not perform score fusion itself.

    RRF fusion belongs to rrf_fusion.py.

    This separation keeps retrieval orchestration and
    ranking/fusion responsibilities independent.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Parameters
        ----------
        dense_retriever:
            Configured dense retrieval implementation.

        sparse_retriever:
            Configured sparse BM25 retrieval implementation.

        Raises
        ------
        ValueError
            If either retriever is invalid.
        """

        if not isinstance(
            dense_retriever,
            DenseRetriever,
        ):
            raise ValueError(
                "dense_retriever must be a DenseRetriever instance."
            )

        if not isinstance(
            sparse_retriever,
            SparseRetriever,
        ):
            raise ValueError(
                "sparse_retriever must be a SparseRetriever instance."
            )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever

    # ==========================================================
    # Hybrid Retrieval
    # ==========================================================

    def retrieve(
        self,
        request: RetrievalRequest,
        query_vector: list[float],
    ) -> tuple[
        list[RetrievalResult],
        list[RetrievalResult],
    ]:
        """
        Execute dense and sparse retrieval independently.

        Parameters
        ----------
        request:
            Validated retrieval request.

        query_vector:
            Query embedding used by dense retrieval.

        Returns
        -------
        tuple[list[RetrievalResult], list[RetrievalResult]]
            First list contains dense candidates.
            Second list contains sparse candidates.

        Notes
        -----
        Fusion is intentionally not performed here.

        RRF fusion belongs to the dedicated RRF fusion component.
        """

        self._validate_request(request)

        self._validate_query_vector(
            query_vector
        )

        dense_results = self.dense_retriever.retrieve(
            request=request,
            query_vector=query_vector,
        )

        sparse_results = self.sparse_retriever.retrieve(
            request=request,
        )

        return (
            dense_results,
            sparse_results,
        )

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_request(
        request: RetrievalRequest,
    ) -> None:
        """
        Validate retrieval request.
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
        Validate query embedding before dense retrieval.
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
            isinstance(
                value,
                (int, float),
            )
            and not isinstance(value, bool)
            for value in query_vector
        ):
            raise ValueError(
                "Query vector must contain only numeric values."
            )