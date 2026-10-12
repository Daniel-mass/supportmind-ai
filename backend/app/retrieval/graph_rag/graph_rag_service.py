"""
graph_rag_service.py

Main orchestration service for SupportMind Graph-RAG generation.

Pipeline
--------
GraphRAGRequest
    |
    +--> Existing RetrievalService
    |
    +--> IntelligenceAdapter
    |
    v
EvidenceFusion
    |
    v
GraphRAGContext
    |
    v
GraphRAGGenerator
    |
    v
GroundednessValidator
    |
    v
GraphRAGResponse

This service deliberately contains orchestration only.

It does not:
    - implement retrieval
    - implement Neo4j traversal
    - implement evidence ranking
    - construct prompts
    - call Ollama directly
    - implement grounding rules
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from app.retrieval.context.context_models import BuiltContext
from app.retrieval.graph_rag.evidence_fusion import (
    EvidenceFusion,
)
from app.retrieval.graph_rag.generator import (
    GraphRAGGenerator,
)
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGRequest,
    GraphRAGResponse,
)


# ============================================================================
# Protocols
# ============================================================================


class RetrievalServiceProtocol(Protocol):
    """
    Minimal interface required from the existing RetrievalService.
    """

    def retrieve(
        self,
        request: Any,
    ) -> BuiltContext:
        """Execute the existing retrieval pipeline."""
        ...


class IntelligenceAdapterProtocol(Protocol):
    """
    Minimal interface required from IntelligenceAdapter.
    """

    def get_ticket_evidence(
        self,
        ticket_id: str,
        *,
        include_neighborhood: bool = True,
        include_workflow: bool = True,
        include_transitions: bool = True,
    ) -> list[GraphRAGEvidence]:
        """Return operational evidence for a ticket."""
        ...

    def get_organization_evidence(
        self,
    ) -> list[GraphRAGEvidence]:
        """Return all organization-level operational evidence."""
        ...

    def get_relevant_organization_evidence(
        self,
        query: str,
    ) -> list[GraphRAGEvidence]:
        """
        Return organization-level operational evidence relevant to a query.
        """
        ...


# ============================================================================
# Graph-RAG service
# ============================================================================


class GraphRAGService:
    """
    Orchestrate retrieval, intelligence evidence, fusion, generation,
    and groundedness validation.
    """

    def __init__(
        self,
        *,
        retrieval_service: RetrievalServiceProtocol,
        intelligence_adapter: IntelligenceAdapterProtocol,
        evidence_fusion: EvidenceFusion,
        generator: GraphRAGGenerator,
        groundedness_validator: GroundednessValidator,
    ) -> None:
        self.retrieval_service = (
            self._validate_dependency(
                retrieval_service,
                "retrieval_service",
            )
        )

        self.intelligence_adapter = (
            self._validate_dependency(
                intelligence_adapter,
                "intelligence_adapter",
            )
        )

        self.evidence_fusion = (
            self._validate_dependency(
                evidence_fusion,
                "evidence_fusion",
            )
        )

        self.generator = (
            self._validate_dependency(
                generator,
                "generator",
            )
        )

        self.groundedness_validator = (
            self._validate_dependency(
                groundedness_validator,
                "groundedness_validator",
            )
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def answer(
        self,
        request: GraphRAGRequest,
    ) -> GraphRAGResponse:
        """
        Execute the complete Graph-RAG generation pipeline.
        """

        if not isinstance(
            request,
            GraphRAGRequest,
        ):
            raise TypeError(
                "request must be a GraphRAGRequest instance."
            )

        self._validate_query_consistency(
            request
        )

        start = perf_counter()

        # ------------------------------------------------------------------
        # 1. Existing retrieval layer
        # ------------------------------------------------------------------

        built_context = (
            self.retrieval_service.retrieve(
                request.retrieval_request
            )
        )

        if not isinstance(
            built_context,
            BuiltContext,
        ):
            raise TypeError(
                "retrieval_service.retrieve() must return "
                "a BuiltContext instance."
            )

        # ------------------------------------------------------------------
        # 2. Intelligence / Help Desk evidence
        # ------------------------------------------------------------------

        intelligence_evidence = (
            self._collect_intelligence_evidence(
                request
            )
        )

        # ------------------------------------------------------------------
        # 3. Evidence fusion
        # ------------------------------------------------------------------

        graph_rag_context = (
            self.evidence_fusion.fuse(
                built_context=built_context,
                intelligence_evidence=(
                    intelligence_evidence
                ),
                max_items=(
                    request.max_evidence_items
                ),
                max_characters=(
                    request.max_evidence_characters
                ),
            )
        )

        # ------------------------------------------------------------------
        # 4. Generation
        # ------------------------------------------------------------------

        answer = self.generator.generate(
            graph_rag_context
        )

        # ------------------------------------------------------------------
        # 5. Groundedness validation
        # ------------------------------------------------------------------

        grounding = (
            self.groundedness_validator.validate(
                answer=answer,
                context=graph_rag_context,
            )
        )

        # ------------------------------------------------------------------
        # 6. Final response
        # ------------------------------------------------------------------

        latency_ms = (
            perf_counter() - start
        ) * 1000.0

        return GraphRAGResponse(
            query=request.query,
            answer=answer,
            evidence=(
                graph_rag_context.evidence
            ),
            grounding=grounding,
            provider=(
                self.generator.provider_name
            ),
            model_name=(
                self.generator.model_name
            ),
            latency_ms=latency_ms,
            metadata={
                "evidence_count": (
                    graph_rag_context.selected_count
                ),
                "evidence_character_count": (
                    graph_rag_context.character_count
                ),
                "evidence_truncated": (
                    graph_rag_context.truncated
                ),
                "retrieval_context_count": (
                    len(
                        built_context.selected_items
                    )
                ),
                "intelligence_evidence_count": (
                    len(
                        intelligence_evidence
                    )
                ),
            },
        )

    # ========================================================================
    # Intelligence collection
    # ========================================================================

    def _collect_intelligence_evidence(
        self,
        request: GraphRAGRequest,
    ) -> list[GraphRAGEvidence]:
        """
        Collect Help Desk operational evidence.

        Current routing:
            ticket_id present
                -> ticket/workflow evidence

            no ticket_id + intelligence enabled
                -> query-relevant organization-level evidence

        Organization-level evidence is selected using the user's query
        rather than loading every available organization-level metric.

        This keeps the first version conservative and bounded. Agent/team-
        specific routing can be added later without changing the fusion
        contract.
        """

        if not request.include_intelligence:
            return []

        if request.ticket_id:
            return (
                self.intelligence_adapter
                .get_ticket_evidence(
                    request.ticket_id,
                    include_neighborhood=True,
                    include_workflow=(
                        request.include_workflow
                    ),
                    include_transitions=(
                        request.include_workflow
                    ),
                )
            )

        MAX_INTELLIGENCE_EVIDENCE = 5

        intelligence_evidence = (
            self.intelligence_adapter
            .get_relevant_organization_evidence(
                request.query
            )
        )

        return intelligence_evidence[
            :MAX_INTELLIGENCE_EVIDENCE
        ]

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_dependency(
        dependency: Any,
        name: str,
    ) -> Any:
        if dependency is None:
            raise ValueError(
                f"{name} cannot be None."
            )

        return dependency

    @staticmethod
    def _validate_query_consistency(
        request: GraphRAGRequest,
    ) -> None:
        """
        Ensure the outer Graph-RAG query and inner retrieval query do not
        silently diverge.
        """

        outer_query = request.query.strip()
        retrieval_query = (
            request.retrieval_request.query.strip()
        )

        if outer_query != retrieval_query:
            raise ValueError(
                "GraphRAGRequest.query must match "
                "GraphRAGRequest.retrieval_request.query."
            )

    # ========================================================================
    # Lifecycle
    # ========================================================================

    def close(self) -> None:
        """
        Close dependencies that expose close().
        """

        dependencies = (
            self.retrieval_service,
            self.intelligence_adapter,
            self.evidence_fusion,
            self.generator,
            self.groundedness_validator,
        )

        closed: set[int] = set()

        for dependency in dependencies:
            dependency_id = id(
                dependency
            )

            if dependency_id in closed:
                continue

            closed.add(
                dependency_id
            )

            close_method = getattr(
                dependency,
                "close",
                None,
            )

            if callable(
                close_method
            ):
                close_method()

    def __enter__(
        self,
    ) -> "GraphRAGService":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()