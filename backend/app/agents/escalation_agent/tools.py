#escalation/tools.py
from __future__ import annotations


from typing import Any

from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationResult,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
)
from app.agents.escalation_agent.contracts.state import (
    EscalationAgentState,
    EscalationClassificationSignal,
    EscalationEvidence,
    EscalationResolutionSignal,
    EscalationSignal,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGResponse,
)


class EscalationTools:
    """
    Adapter and validation utilities used by the Escalation Agent.

    These tools do not calculate operational analytics.

    Existing SupportMind services remain responsible for producing
    classification predictions, resolution recommendations,
    operational intelligence, retrieval evidence, and Graph-RAG
    results.
    """

    # ------------------------------------------------------------------
    # Escalation state validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_state(
        state: EscalationAgentState,
    ) -> EscalationAgentState:
        """Validate that the supplied object is EscalationAgentState."""

        if not isinstance(state, EscalationAgentState):
            raise TypeError(
                "state must be an instance of EscalationAgentState."
            )

        return state

    # ------------------------------------------------------------------
    # Classification adapters
    # ------------------------------------------------------------------

    @staticmethod
    def validate_classification_result(
        result: ClassificationResult,
    ) -> ClassificationResult:
        """Validate an existing ClassificationResult."""

        if not isinstance(result, ClassificationResult):
            raise TypeError(
                "result must be an instance of ClassificationResult."
            )

        return result

    @staticmethod
    def classification_signals(
        result: ClassificationResult,
    ) -> list[EscalationClassificationSignal]:
        """
        Convert existing classification predictions into escalation
        classification signals.

        No prediction is recalculated here.
        """

        result = EscalationTools.validate_classification_result(result)

        predictions = (
            result.category,
            result.priority,
            result.sentiment,
            result.complexity,
            result.topic,
        )

        signals: list[EscalationClassificationSignal] = []

        for prediction in predictions:
            if prediction is None:
                continue

            signals.append(
                EscalationTools._prediction_to_classification_signal(
                    prediction
                )
            )

        return signals

    @staticmethod
    def _prediction_to_classification_signal(
        prediction: ClassificationPrediction,
    ) -> EscalationClassificationSignal:
        """Convert one existing classification prediction."""

        if not isinstance(prediction, ClassificationPrediction):
            raise TypeError(
                "prediction must be an instance of ClassificationPrediction."
            )

        return EscalationClassificationSignal(
            name=prediction.classification_type.value,
            value=prediction.value,
            confidence=prediction.confidence,
            source=prediction.source,
            metadata=dict(prediction.metadata),
        )

    # ------------------------------------------------------------------
    # Resolution adapters
    # ------------------------------------------------------------------

    @staticmethod
    def validate_resolution_state(
        state: ResolutionAgentState,
    ) -> ResolutionAgentState:
        """Validate an existing ResolutionAgentState."""

        if not isinstance(state, ResolutionAgentState):
            raise TypeError(
                "state must be an instance of ResolutionAgentState."
            )

        return state

    @staticmethod
    def resolution_signal(
        state: ResolutionAgentState,
    ) -> EscalationResolutionSignal | None:
        """
        Convert the existing Resolution Agent recommendation into an
        escalation resolution signal.

        If the Resolution Agent did not produce a recommendation,
        return None rather than inventing one.
        """

        state = EscalationTools.validate_resolution_state(state)

        recommendation = state.recommendation

        if recommendation is None:
            return None

        return EscalationResolutionSignal(
            resolution_type=recommendation.resolution_type.value,
            recommendation=recommendation.recommendation,
            rationale=recommendation.rationale,
            confidence=recommendation.confidence,
            requires_human_review=recommendation.requires_human_review,
            proposed_actions=list(recommendation.proposed_actions),
            metadata={
                "evidence_ids": list(recommendation.evidence_ids),
                "resolution_agent_status": state.status.value,
                "resolution_iteration_count": state.iteration_count,
            },
        )

    # ------------------------------------------------------------------
    # Operational signal validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_operational_signal(
        signal: EscalationSignal,
    ) -> EscalationSignal:
        """
        Validate an operational signal that was already produced by
        an upstream intelligence/analytics component.

        This method deliberately does not calculate the signal.
        """

        if not isinstance(signal, EscalationSignal):
            raise TypeError(
                "signal must be an instance of EscalationSignal."
            )

        return signal

    @staticmethod
    def validate_operational_signals(
        signals: list[EscalationSignal],
    ) -> list[EscalationSignal]:
        """Validate a collection of existing operational signals."""

        if not isinstance(signals, list):
            raise TypeError(
                "signals must be a list of EscalationSignal objects."
            )

        return [
            EscalationTools.validate_operational_signal(signal)
            for signal in signals
        ]

    # ------------------------------------------------------------------
    # Graph-RAG / evidence adapters
    # ------------------------------------------------------------------

    @staticmethod
    def validate_graph_rag_response(
        response: GraphRAGResponse,
    ) -> GraphRAGResponse:
        """Validate an existing Graph-RAG response."""

        if not isinstance(response, GraphRAGResponse):
            raise TypeError(
                "response must be an instance of GraphRAGResponse."
            )

        return response

    @staticmethod
    def graph_rag_evidence(
        response: GraphRAGResponse,
    ) -> list[EscalationEvidence]:
        """Convert Graph-RAG evidence into escalation evidence."""

        response = EscalationTools.validate_graph_rag_response(response)

        return [
            EscalationTools._graph_rag_evidence_to_escalation_evidence(
                evidence
            )
            for evidence in response.evidence
        ]

    @staticmethod
    def _graph_rag_evidence_to_escalation_evidence(
        evidence: GraphRAGEvidence,
    ) -> EscalationEvidence:
        """Convert one Graph-RAG evidence object."""

        if not isinstance(evidence, GraphRAGEvidence):
            raise TypeError(
                "evidence must be an instance of GraphRAGEvidence."
            )

        return EscalationEvidence(
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
        """Return evidence IDs from an existing Graph-RAG response."""

        response = EscalationTools.validate_graph_rag_response(response)

        return [
            evidence.evidence_id
            for evidence in response.evidence
        ]

    @staticmethod
    def graph_rag_grounding(
        response: GraphRAGResponse,
    ) -> dict[str, Any]:
        """
        Extract the existing Graph-RAG grounding result.

        Grounding is evaluated by the Graph-RAG layer, not here.
        """

        response = EscalationTools.validate_graph_rag_response(response)

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
            "issues": list(response.grounding.issues),
            "metadata": dict(response.grounding.metadata),
        }

    @staticmethod
    def graph_rag_metadata(
        response: GraphRAGResponse,
    ) -> dict[str, Any]:
        """Extract metadata from an existing Graph-RAG response."""

        response = EscalationTools.validate_graph_rag_response(response)

        return {
            "provider": response.provider,
            "model_name": response.model_name,
            "latency_ms": response.latency_ms,
            **dict(response.metadata),
        }