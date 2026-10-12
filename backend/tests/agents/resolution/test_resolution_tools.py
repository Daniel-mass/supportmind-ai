from __future__ import annotations

import pytest

from app.agents.classification_agent.contracts.state import (
    ClassificationPrediction,
    ClassificationResult,
    ClassificationType,
)
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionClassificationSignal,
    ResolutionEvidence,
)
from app.agents.resolution_agent.tools.resolution_tools import (
    ResolutionTools,
)
from app.retrieval.context.context_models import (
    BuiltContext,
    ContextSection,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GraphRAGResponse,
    GroundingResult,
    GroundingStatus,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


# ============================================================================
# Test helpers
# ============================================================================


def build_classification_result() -> ClassificationResult:
    return ClassificationResult(
        category=ClassificationPrediction(
            classification_type=ClassificationType.CATEGORY,
            value="Account",
            confidence=0.82,
            source="ticket_category_predictor",
            metadata={
                "input_length": 42,
            },
        ),
        priority=ClassificationPrediction(
            classification_type=ClassificationType.PRIORITY,
            value="High",
            confidence=0.76,
            source="priority_detection_predictor",
            metadata={
                "probabilities": {
                    "High": 0.76,
                    "Medium": 0.18,
                    "Low": 0.06,
                }
            },
        ),
        sentiment=ClassificationPrediction(
            classification_type=ClassificationType.SENTIMENT,
            value="negative",
            confidence=0.91,
            source="sentiment_predictor",
        ),
        complexity=ClassificationPrediction(
            classification_type=ClassificationType.COMPLEXITY,
            value="MULTI_TURN",
            confidence=0.95,
            source="conversation_complexity_predictor",
        ),
        topic=ClassificationPrediction(
            classification_type=ClassificationType.TOPIC,
            value="24",
            confidence=0.73,
            source="topic_model_predictor",
            metadata={
                "keywords": [
                    "billing",
                    "payment",
                ]
            },
        ),
        predictions_completed=5,
        predictions_requested=5,
        uncertainty_detected=False,
    )


def build_retrieval_state() -> RetrievalAgentState:
    request = RetrievalRequest(
        query="How should this billing issue be resolved?",
        top_k=5,
        collection_name="supportmind_test",
        include_graph=True,
    )

    context = BuiltContext(
        query="How should this billing issue be resolved?",
        sections=[
            ContextSection(
                title="Evidence",
                content="Billing policy requires transaction verification.",
            )
        ],
        selected_items=[],
        total_candidates=1,
        selected_count=0,
        character_count=51,
        truncated=False,
    )

    return RetrievalAgentState(
        request=request,
        use_graph_rag=False,
        context=context,
        evidence_ids=[
            "retrieval-1",
            "retrieval-2",
        ],
        execution_metadata={
            "latency_ms": 125.5,
            "candidate_count": 10,
            "final_count": 5,
        },
    )


def build_graph_rag_response() -> GraphRAGResponse:
    evidence = [
        GraphRAGEvidence(
            evidence_id="E1",
            source=GraphRAGEvidenceSource.RETRIEVAL_SPARSE,
            evidence_type=GraphRAGEvidenceType.DOCUMENT,
            text="Duplicate billing charges require transaction verification.",
            source_reference="document:billing_policy",
            score=0.91,
            metadata={
                "document_id": "billing_policy",
            },
        ),
        GraphRAGEvidence(
            evidence_id="E2",
            source=GraphRAGEvidenceSource.TICKET_INTELLIGENCE,
            evidence_type=GraphRAGEvidenceType.TICKET_CONTEXT,
            text="The ticket has a High priority classification.",
            source_reference="ticket:TICKET-1001",
            score=0.84,
        ),
    ]

    grounding = GroundingResult(
        status=GroundingStatus.GROUNDED,
        valid=True,
        cited_evidence_ids=[
            "E1",
            "E2",
        ],
        invalid_citation_ids=[],
        unsupported_claims=[],
        issues=[],
        metadata={
            "citation_count": 2,
        },
    )

    return GraphRAGResponse(
        query="How should this billing issue be resolved?",
        answer=(
            "Verify the duplicate transaction before proceeding. "
            "[E1] The ticket is classified as High priority. [E2]"
        ),
        evidence=evidence,
        grounding=grounding,
        provider="ollama",
        model_name="qwen3:8b",
        latency_ms=1500.0,
        metadata={
            "evidence_count": 2,
            "grounding_status": "grounded",
        },
    )


# ============================================================================
# Classification tools
# ============================================================================


def test_validate_classification_result() -> None:
    result = build_classification_result()

    validated = ResolutionTools.validate_classification_result(
        result
    )

    assert validated is result


def test_validate_classification_result_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        ResolutionTools.validate_classification_result(
            object()  # type: ignore[arg-type]
        )


def test_classification_signals_converts_all_predictions() -> None:
    result = build_classification_result()

    signals = ResolutionTools.classification_signals(
        result
    )

    assert len(signals) == 5

    assert all(
        isinstance(
            signal,
            ResolutionClassificationSignal,
        )
        for signal in signals
    )

    assert [
        signal.name
        for signal in signals
    ] == [
        "category",
        "priority",
        "sentiment",
        "complexity",
        "topic",
    ]


def test_classification_signals_preserves_prediction_values() -> None:
    result = build_classification_result()

    signals = ResolutionTools.classification_signals(
        result
    )

    assert signals[0].value == "Account"
    assert signals[0].confidence == 0.82

    assert signals[1].value == "High"
    assert signals[1].confidence == 0.76

    assert signals[2].value == "negative"
    assert signals[2].confidence == 0.91


def test_classification_signals_preserves_metadata() -> None:
    result = build_classification_result()

    signals = ResolutionTools.classification_signals(
        result
    )

    priority_signal = signals[1]

    assert (
        priority_signal.metadata["probabilities"]["High"]
        == 0.76
    )


def test_classification_signals_skips_missing_predictions() -> None:
    result = ClassificationResult(
        category=ClassificationPrediction(
            classification_type=ClassificationType.CATEGORY,
            value="Account",
            confidence=0.8,
            source="ticket_category_predictor",
        ),
        predictions_completed=1,
        predictions_requested=5,
    )

    signals = ResolutionTools.classification_signals(
        result
    )

    assert len(signals) == 1
    assert signals[0].name == "category"


# ============================================================================
# Retrieval tools
# ============================================================================


def test_validate_retrieval_state() -> None:
    state = build_retrieval_state()

    validated = ResolutionTools.validate_retrieval_state(
        state
    )

    assert validated is state


def test_validate_retrieval_state_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        ResolutionTools.validate_retrieval_state(
            object()  # type: ignore[arg-type]
        )


def test_retrieval_context_returns_existing_context() -> None:
    state = build_retrieval_state()

    context = ResolutionTools.retrieval_context(
        state
    )

    assert context is state.context
    assert isinstance(context, BuiltContext)


def test_retrieval_context_returns_none_when_unavailable() -> None:
    state = RetrievalAgentState(
        request=RetrievalRequest(
            query="Test query",
            top_k=5,
            collection_name="supportmind_test",
        )
    )

    assert (
        ResolutionTools.retrieval_context(state)
        is None
    )


def test_retrieval_evidence_ids_are_copied() -> None:
    state = build_retrieval_state()

    evidence_ids = ResolutionTools.retrieval_evidence_ids(
        state
    )

    assert evidence_ids == [
        "retrieval-1",
        "retrieval-2",
    ]

    evidence_ids.append("modified")

    assert state.evidence_ids == [
        "retrieval-1",
        "retrieval-2",
    ]


def test_retrieval_execution_metadata_is_copied() -> None:
    state = build_retrieval_state()

    metadata = (
        ResolutionTools.retrieval_execution_metadata(
            state
        )
    )

    assert metadata["latency_ms"] == 125.5
    assert metadata["candidate_count"] == 10

    metadata["candidate_count"] = 999

    assert (
        state.execution_metadata["candidate_count"]
        == 10
    )


# ============================================================================
# Graph-RAG tools
# ============================================================================


def test_validate_graph_rag_response() -> None:
    response = build_graph_rag_response()

    validated = (
        ResolutionTools.validate_graph_rag_response(
            response
        )
    )

    assert validated is response


def test_validate_graph_rag_response_rejects_invalid_type() -> None:
    with pytest.raises(TypeError):
        ResolutionTools.validate_graph_rag_response(
            object()  # type: ignore[arg-type]
        )


def test_graph_rag_evidence_converts_all_items() -> None:
    response = build_graph_rag_response()

    evidence = ResolutionTools.graph_rag_evidence(
        response
    )

    assert len(evidence) == 2

    assert all(
        isinstance(
            item,
            ResolutionEvidence,
        )
        for item in evidence
    )

    assert evidence[0].evidence_id == "E1"
    assert evidence[0].source == "retrieval_sparse"
    assert (
        evidence[0].content
        == "Duplicate billing charges require transaction verification."
    )
    assert evidence[0].source_reference == (
        "document:billing_policy"
    )
    assert evidence[0].score == 0.91


def test_graph_rag_evidence_preserves_evidence_type() -> None:
    response = build_graph_rag_response()

    evidence = ResolutionTools.graph_rag_evidence(
        response
    )

    assert (
        evidence[0].metadata["evidence_type"]
        == "document"
    )

    assert (
        evidence[1].metadata["evidence_type"]
        == "ticket_context"
    )


def test_graph_rag_evidence_ids() -> None:
    response = build_graph_rag_response()

    evidence_ids = ResolutionTools.graph_rag_evidence_ids(
        response
    )

    assert evidence_ids == [
        "E1",
        "E2",
    ]


def test_graph_rag_grounding_preserves_validation_result() -> None:
    response = build_graph_rag_response()

    grounding = ResolutionTools.graph_rag_grounding(
        response
    )

    assert grounding["status"] == "grounded"
    assert grounding["valid"] is True
    assert grounding["cited_evidence_ids"] == [
        "E1",
        "E2",
    ]
    assert grounding["invalid_citation_ids"] == []
    assert grounding["unsupported_claims"] == []
    assert grounding["issues"] == []


def test_graph_rag_metadata() -> None:
    response = build_graph_rag_response()

    metadata = ResolutionTools.graph_rag_metadata(
        response
    )

    assert metadata["provider"] == "ollama"
    assert metadata["model_name"] == "qwen3:8b"
    assert metadata["latency_ms"] == 1500.0
    assert metadata["evidence_count"] == 2


# ============================================================================
# Combined flow
# ============================================================================


def test_resolution_tools_support_classification_retrieval_and_graph_rag() -> None:
    classification = build_classification_result()
    retrieval = build_retrieval_state()
    graph_rag = build_graph_rag_response()

    signals = ResolutionTools.classification_signals(
        classification
    )

    context = ResolutionTools.retrieval_context(
        retrieval
    )

    evidence = ResolutionTools.graph_rag_evidence(
        graph_rag
    )

    assert len(signals) == 5
    assert isinstance(context, BuiltContext)
    assert len(evidence) == 2

    assert signals[1].value == "High"
    assert evidence[0].evidence_id == "E1"
    assert graph_rag.grounding.valid is True