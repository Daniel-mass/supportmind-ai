'''retrivealtools'''
from __future__ import annotations

from typing import Any

from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGRequest,
    GraphRAGResponse,
)
from app.retrieval.graph_rag.graph_rag_service import GraphRAGService
from app.retrieval.context.context_models import BuiltContext
from app.retrieval.models.retrieval_models import RetrievalRequest
from app.retrieval.retrieval_service import RetrievalService


class RetrievalTools:
    """
    Thin execution adapters exposing existing SupportMind retrieval
    services to the Retrieval Agent.

    This class intentionally contains no retrieval algorithms.

    Existing services remain responsible for:

        Dense Retrieval
        Sparse / BM25 Retrieval
        Hybrid Retrieval
        RRF Fusion
        Graph Retrieval
        Reranking
        Context Building
        Graph-RAG Evidence Fusion
        Grounded Generation
        Grounding Validation
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        graph_rag_service: GraphRAGService,
    ) -> None:
        """
        Initialize Retrieval Agent execution tools.

        Dependencies are injected so the tools can be tested with
        controlled test doubles and later connected to the real
        SupportMind runtime.
        """

        if retrieval_service is None:
            raise ValueError(
                "retrieval_service must not be None."
            )

        if graph_rag_service is None:
            raise ValueError(
                "graph_rag_service must not be None."
            )

        self._retrieval_service = retrieval_service
        self._graph_rag_service = graph_rag_service

    @property
    def retrieval_service(self) -> RetrievalService:
        """
        Return the configured RetrievalService.
        """

        return self._retrieval_service

    @property
    def graph_rag_service(self) -> GraphRAGService:
        """
        Return the configured GraphRAGService.
        """

        return self._graph_rag_service

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> BuiltContext:
        """
        Execute the existing SupportMind retrieval pipeline.

        The Retrieval Agent does not implement retrieval itself.
        It delegates directly to RetrievalService.
        """

        if not isinstance(request, RetrievalRequest):
            raise TypeError(
                "request must be an instance of RetrievalRequest."
            )

        result = self._retrieval_service.retrieve(request)

        if not isinstance(result, BuiltContext):
            raise TypeError(
                "RetrievalService.retrieve() must return BuiltContext."
            )

        return result

    def graph_rag(
        self,
        request: GraphRAGRequest,
    ) -> GraphRAGResponse:
        """
        Execute the existing SupportMind Graph-RAG pipeline.

        GraphRAGService remains responsible for retrieval evidence,
        intelligence evidence, evidence fusion, generation, and
        grounding validation.
        """

        if not isinstance(request, GraphRAGRequest):
            raise TypeError(
                "request must be an instance of GraphRAGRequest."
            )

        result = self._graph_rag_service.answer(request)

        if not isinstance(result, GraphRAGResponse):
            raise TypeError(
                "GraphRAGService.answer() must return GraphRAGResponse."
            )

        return result

    def execution_metadata(self) -> dict[str, Any]:
        """
        Return the latest RetrievalService execution metadata.

        This intentionally reads the existing service metadata rather
        than calculating or duplicating retrieval metrics.
        """

        metadata = self._retrieval_service.last_execution_metadata

        if not isinstance(metadata, dict):
            raise TypeError(
                "RetrievalService.last_execution_metadata must be a dict."
            )

        return dict(metadata)