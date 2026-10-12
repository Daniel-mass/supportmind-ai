"""
Unit tests for Graph-RAG generation-layer models.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGContext,
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GraphRAGRequest,
    GraphRAGResponse,
    GroundingResult,
    GroundingStatus,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    """Return a valid existing retrieval-layer request."""

    return RetrievalRequest(
        query="Why was the ticket delayed?",
        top_k=5,
        collection_name="supportmind",
        include_graph=True,
        rerank=True,
        rerank_top_k=5,
    )


@pytest.fixture
def evidence() -> GraphRAGEvidence:
    """Return a valid normalized evidence item."""

    return GraphRAGEvidence(
        evidence_id="E1",
        source=GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
        evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
        text="Ticket ABC-123 was assigned to the Payments team.",
        source_reference="ticket:ABC-123",
        score=0.91,
        metadata={
            "node_id": "ticket_abc_123",
        },
    )


@pytest.fixture
def grounding_result() -> GroundingResult:
    """Return a valid grounding result."""

    return GroundingResult(
        status=GroundingStatus.GROUNDED,
        valid=True,
        cited_evidence_ids=["E1"],
    )


# ============================================================================
# GraphRAGRequest
# ============================================================================


def test_graph_rag_request_accepts_valid_request(
    retrieval_request: RetrievalRequest,
) -> None:
    request = GraphRAGRequest(
        query="  Why was the ticket delayed?  ",
        retrieval_request=retrieval_request,
        ticket_id=" ABC-123 ",
    )

    assert request.query == "Why was the ticket delayed?"
    assert request.ticket_id == "ABC-123"
    assert request.include_intelligence is True
    assert request.include_workflow is True


def test_graph_rag_request_rejects_empty_query(
    retrieval_request: RetrievalRequest,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGRequest(
            query="   ",
            retrieval_request=retrieval_request,
        )


def test_graph_rag_request_rejects_empty_ticket_id(
    retrieval_request: RetrievalRequest,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGRequest(
            query="Why was the ticket delayed?",
            retrieval_request=retrieval_request,
            ticket_id="   ",
        )


def test_graph_rag_request_rejects_non_positive_evidence_limit(
    retrieval_request: RetrievalRequest,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGRequest(
            query="Why was the ticket delayed?",
            retrieval_request=retrieval_request,
            max_evidence_items=0,
        )


def test_graph_rag_request_rejects_extra_fields(
    retrieval_request: RetrievalRequest,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGRequest(
            query="Why was the ticket delayed?",
            retrieval_request=retrieval_request,
            unexpected="value",
        )


# ============================================================================
# GraphRAGEvidence
# ============================================================================


def test_evidence_accepts_valid_item() -> None:
    item = GraphRAGEvidence(
        evidence_id="E12",
        source=GraphRAGEvidenceSource.HELPDESK_GRAPH,
        evidence_type=GraphRAGEvidenceType.METRIC,
        text="Average waiting time is 18 hours.",
        source_reference="workflow:ticket:ABC-123",
        score=0.82,
    )

    assert item.evidence_id == "E12"
    assert item.source == GraphRAGEvidenceSource.HELPDESK_GRAPH
    assert item.score == pytest.approx(0.82)


@pytest.mark.parametrize(
    "evidence_id",
    [
        "1",
        "ABC",
        "E",
        "E0",
        "E-1",
        "E1.5",
    ],
)
def test_evidence_rejects_invalid_evidence_id(
    evidence_id: str,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGEvidence(
            evidence_id=evidence_id,
            source=GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
            evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
            text="Graph evidence.",
            source_reference="ticket:ABC-123",
        )


def test_evidence_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        GraphRAGEvidence(
            evidence_id="E1",
            source=GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
            evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
            text="   ",
            source_reference="ticket:ABC-123",
        )


def test_evidence_rejects_empty_source_reference() -> None:
    with pytest.raises(ValidationError):
        GraphRAGEvidence(
            evidence_id="E1",
            source=GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
            evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
            text="Graph evidence.",
            source_reference="   ",
        )


@pytest.mark.parametrize(
    "score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evidence_rejects_non_finite_score(
    score: float,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGEvidence(
            evidence_id="E1",
            source=GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
            evidence_type=GraphRAGEvidenceType.GRAPH_FACT,
            text="Graph evidence.",
            source_reference="ticket:ABC-123",
            score=score,
        )


# ============================================================================
# GraphRAGContext
# ============================================================================


def test_graph_rag_context_accepts_evidence(
    evidence: GraphRAGEvidence,
) -> None:
    context = GraphRAGContext(
        query="Why was the ticket delayed?",
        evidence=[evidence],
        selected_count=1,
        character_count=len(evidence.text),
    )

    assert context.selected_count == 1
    assert context.character_count == len(evidence.text)
    assert len(context.evidence) == 1


def test_graph_rag_context_rejects_negative_counts(
    evidence: GraphRAGEvidence,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGContext(
            query="Why?",
            evidence=[evidence],
            selected_count=-1,
            character_count=10,
        )

    with pytest.raises(ValidationError):
        GraphRAGContext(
            query="Why?",
            evidence=[evidence],
            selected_count=1,
            character_count=-1,
        )


# ============================================================================
# GroundingResult
# ============================================================================


def test_grounding_result_accepts_valid_result() -> None:
    result = GroundingResult(
        status=GroundingStatus.GROUNDED,
        valid=True,
        cited_evidence_ids=["E1", "E2"],
    )

    assert result.status == GroundingStatus.GROUNDED
    assert result.valid is True
    assert result.cited_evidence_ids == ["E1", "E2"]


def test_grounding_result_supports_insufficient_evidence() -> None:
    result = GroundingResult(
        status=GroundingStatus.INSUFFICIENT_EVIDENCE,
        valid=False,
        issues=["No supporting evidence was retrieved."],
    )

    assert result.valid is False
    assert result.status == GroundingStatus.INSUFFICIENT_EVIDENCE


# ============================================================================
# GraphRAGResponse
# ============================================================================


def test_graph_rag_response_accepts_valid_response(
    evidence: GraphRAGEvidence,
    grounding_result: GroundingResult,
) -> None:
    response = GraphRAGResponse(
        query="Why was the ticket delayed?",
        answer="The ticket remained in customer approval for 18 hours. [E1]",
        evidence=[evidence],
        grounding=grounding_result,
        provider="ollama",
        model_name="qwen2.5:7b",
        latency_ms=125.4,
    )

    assert response.answer.startswith(
        "The ticket remained"
    )
    assert response.provider == "ollama"
    assert response.model_name == "qwen2.5:7b"
    assert response.latency_ms == pytest.approx(125.4)


def test_graph_rag_response_rejects_negative_latency(
    grounding_result: GroundingResult,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGResponse(
            query="Why?",
            answer="There is insufficient evidence.",
            grounding=grounding_result,
            provider="ollama",
            model_name="qwen2.5:7b",
            latency_ms=-1.0,
        )


def test_graph_rag_response_rejects_non_finite_latency(
    grounding_result: GroundingResult,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGResponse(
            query="Why?",
            answer="There is insufficient evidence.",
            grounding=grounding_result,
            provider="ollama",
            model_name="qwen2.5:7b",
            latency_ms=float("inf"),
        )


def test_graph_rag_response_rejects_empty_answer(
    grounding_result: GroundingResult,
) -> None:
    with pytest.raises(ValidationError):
        GraphRAGResponse(
            query="Why?",
            answer="   ",
            grounding=grounding_result,
            provider="ollama",
            model_name="qwen2.5:7b",
        )