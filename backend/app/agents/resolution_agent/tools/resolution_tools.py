"""
#resoltion_tools.py
Resolution Agent execution tools.

These tools are thin adapters around existing SupportMind contracts.

Responsibilities:
- Validate Classification Agent output.
- Convert classification predictions into Resolution Agent signals.
- Validate Retrieval Agent state.
- Expose existing retrieval context without modifying it.
- Validate Graph-RAG output.
- Convert Graph-RAG evidence into Resolution Agent evidence.

These tools do NOT:
- perform LLM reasoning
- generate resolution recommendations
- perform retrieval
- perform Graph-RAG
- resolve prediction conflicts
- make escalation decisions
- execute customer-impacting actions
"""

from __future__ import annotations

from typing import Any

from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationResult,
)
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionClassificationSignal,
    ResolutionEvidence,
)
from app.retrieval.context.context_models import (
    BuiltContext,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGResponse,
)


class ResolutionTools:
    """
    Thin execution adapters for the Resolution Agent.

    Existing SupportMind agents and services remain responsible for their
    own execution. This class only validates and translates their outputs
    into contracts consumed by the Resolution Agent.
    """

    # ========================================================================
    # Classification
    # ========================================================================

    @staticmethod
    def validate_classification_result(
        result: ClassificationResult,
    ) -> ClassificationResult:
        """
        Validate a completed Classification Agent result.

        The Classification Agent remains responsible for producing the
        predictions. ResolutionTools only verifies the contract.
        """

        if not isinstance(
            result,
            ClassificationResult,
        ):
            raise TypeError(
                "result must be an instance of ClassificationResult."
            )

        return result

    @staticmethod
    def classification_signals(
        result: ClassificationResult,
    ) -> list[ResolutionClassificationSignal]:
        """
        Convert Classification Agent predictions into Resolution signals.

        No prediction is modified or reinterpreted here.

        ML predictions remain advisory signals for the Resolution Agent.
        """

        result = ResolutionTools.validate_classification_result(
            result
        )

        predictions = (
            result.category,
            result.priority,
            result.sentiment,
            result.complexity,
            result.topic,
        )

        signals: list[ResolutionClassificationSignal] = []

        for prediction in predictions:
            if prediction is None:
                continue

            signals.append(
                ResolutionTools._prediction_to_signal(
                    prediction
                )
            )

        return signals

    @staticmethod
    def _prediction_to_signal(
        prediction: ClassificationPrediction,
    ) -> ResolutionClassificationSignal:
        """
        Convert one ClassificationPrediction into a Resolution signal.
        """

        if not isinstance(
            prediction,
            ClassificationPrediction,
        ):
            raise TypeError(
                "prediction must be an instance of "
                "ClassificationPrediction."
            )

        return ResolutionClassificationSignal(
            name=prediction.classification_type.value,
            value=prediction.value,
            confidence=prediction.confidence,
            source=prediction.source,
            metadata=dict(prediction.metadata),
        )

    # ========================================================================
    # Retrieval Agent
    # ========================================================================

    @staticmethod
    def validate_retrieval_state(
        state: RetrievalAgentState,
    ) -> RetrievalAgentState:
        """
        Validate Retrieval Agent state.

        The Retrieval Agent remains responsible for executing retrieval.
        """

        if not isinstance(
            state,
            RetrievalAgentState,
        ):
            raise TypeError(
                "state must be an instance of RetrievalAgentState."
            )

        return state

    @staticmethod
    def retrieval_context(
        state: RetrievalAgentState,
    ) -> BuiltContext | None:
        """
        Expose the existing BuiltContext produced by RetrievalService.

        The Resolution Agent does not rebuild or modify the context.
        """

        state = ResolutionTools.validate_retrieval_state(
            state
        )

        if state.context is not None and not isinstance(
            state.context,
            BuiltContext,
        ):
            raise TypeError(
                "RetrievalAgentState.context must be a "
                "BuiltContext instance or None."
            )

        return state.context

    @staticmethod
    def retrieval_evidence_ids(
        state: RetrievalAgentState,
    ) -> list[str]:
        """
        Return evidence identifiers already exposed by the Retrieval Agent.

        The identifiers are copied so downstream code cannot mutate the
        Retrieval Agent state accidentally.
        """

        state = ResolutionTools.validate_retrieval_state(
            state
        )

        return list(state.evidence_ids)

    @staticmethod
    def retrieval_execution_metadata(
        state: RetrievalAgentState,
    ) -> dict[str, Any]:
        """
        Return Retrieval Agent execution metadata.

        Metadata is copied to preserve state ownership.
        """

        state = ResolutionTools.validate_retrieval_state(
            state
        )

        return dict(state.execution_metadata)

    # ========================================================================
    # Graph-RAG
    # ========================================================================

    @staticmethod
    def validate_graph_rag_response(
        response: GraphRAGResponse,
    ) -> GraphRAGResponse:
        """
        Validate an existing Graph-RAG response.
        """

        if not isinstance(
            response,
            GraphRAGResponse,
        ):
            raise TypeError(
                "response must be an instance of GraphRAGResponse."
            )

        return response

    @staticmethod
    def graph_rag_evidence(
        response: GraphRAGResponse,
    ) -> list[ResolutionEvidence]:
        """
        Convert Graph-RAG evidence into Resolution Agent evidence.

        Graph-RAG remains responsible for:
        - retrieval
        - intelligence evidence
        - evidence fusion
        - generation
        - grounding validation

        ResolutionTools only translates its evidence contract.
        """

        response = ResolutionTools.validate_graph_rag_response(
            response
        )

        return [
            ResolutionTools._graph_rag_evidence_to_resolution(
                evidence
            )
            for evidence in response.evidence
        ]

    @staticmethod
    def _graph_rag_evidence_to_resolution(
        evidence: GraphRAGEvidence,
    ) -> ResolutionEvidence:
        """
        Convert one Graph-RAG evidence item into Resolution evidence.
        """

        if not isinstance(
            evidence,
            GraphRAGEvidence,
        ):
            raise TypeError(
                "evidence must be an instance of GraphRAGEvidence."
            )

        return ResolutionEvidence(
            evidence_id=evidence.evidence_id,
            source=evidence.source.value,
            content=evidence.text,
            source_reference=evidence.source_reference,
            score=evidence.score,
            metadata={
                "evidence_type": evidence.evidence_type.value,
                **dict(evidence.metadata),
            },
        )

    @staticmethod
    def graph_rag_evidence_ids(
        response: GraphRAGResponse,
    ) -> list[str]:
        """
        Return the evidence IDs exposed by Graph-RAG.
        """

        response = ResolutionTools.validate_graph_rag_response(
            response
        )

        return [
            evidence.evidence_id
            for evidence in response.evidence
        ]

    @staticmethod
    def graph_rag_grounding(
        response: GraphRAGResponse,
    ) -> dict[str, Any]:
        """
        Return grounding information from the existing Graph-RAG response.

        Grounding validation is performed by the existing Graph-RAG layer.
        ResolutionTools does not duplicate those rules.
        """

        response = ResolutionTools.validate_graph_rag_response(
            response
        )

        return {
            "status": response.grounding.status.value,
            "valid": response.grounding.valid,
            "cited_evidence_ids": list(
                response.grounding.cited_evidence_ids
            ),
            "invalid_citation_ids": list(
                response.grounding.invalid_citation_ids
            ),
            "unsupported_claims": list(
                response.grounding.unsupported_claims
            ),
            "issues": list(
                response.grounding.issues
            ),
            "metadata": dict(
                response.grounding.metadata
            ),
        }

    @staticmethod
    def graph_rag_metadata(
        response: GraphRAGResponse,
    ) -> dict[str, Any]:
        """
        Return Graph-RAG execution metadata without modifying it.
        """

        response = ResolutionTools.validate_graph_rag_response(
            response
        )

        return {
            "provider": response.provider,
            "model_name": response.model_name,
            "latency_ms": response.latency_ms,
            **dict(response.metadata),
        }