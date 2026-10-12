from __future__ import annotations

from unittest.mock import Mock

import pytest

from app.agents.retrieval_agent.tools.retrieval_tools import RetrievalTools
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGRequest,
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GraphRAGResponse,
    GroundingResult,
    GroundingStatus,
)
from app.retrieval.context.context_models import (
    BuiltContext,
    ContextItem,
    ContextSource,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    """
    Valid request for the standard retrieval tool.
    """

    return RetrievalRequest(
        query="How do I request a refund?",
        top_k=5,
        collection_name="supportmind_retrieval_pdf_test",
    )


@pytest.fixture
def built_context() -> BuiltContext:
    """
    Valid retrieval context returned by the mocked RetrievalService.
    """

    item = ContextItem(
        id="doc-1",
        text="Refund requests are handled according to the refund policy.",
        source=ContextSource.BM25,
        rank=1,
        score=0.91,
    )

    return BuiltContext(
        query="How do I request a refund?",
        selected_items=[item],
        total_candidates=1,
        selected_count=1,
        character_count=len(item.text),
        truncated=False,
    )


@pytest.fixture
def graph_rag_request(
    retrieval_request: RetrievalRequest,
) -> GraphRAGRequest:
    """
    Valid Graph-RAG request.
    """

    return GraphRAGRequest(
        query=retrieval_request.query,
        retrieval_request=retrieval_request,
    )


@pytest.fixture
def graph_rag_response() -> GraphRAGResponse:
    """
    Valid Graph-RAG response returned by the mocked GraphRAGService.
    """

    evidence = GraphRAGEvidence(
        evidence_id="E1",
        source=GraphRAGEvidenceSource.RETRIEVAL_SPARSE,
        evidence_type=GraphRAGEvidenceType.DOCUMENT,
        text="Refund requests are handled according to the refund policy.",
        source_reference="refund_policy.pdf",
        score=0.91,
    )

    grounding = GroundingResult(
        status=GroundingStatus.GROUNDED,
        valid=True,
        cited_evidence_ids=["E1"],
    )

    return GraphRAGResponse(
        query="How do I request a refund?",
        answer="The refund policy explains the refund request process. [E1]",
        evidence=[evidence],
        grounding=grounding,
        provider="ollama",
        model_name="qwen3:8b",
    )


@pytest.fixture
def services() -> tuple[Mock, Mock]:
    """
    Mock retrieval and Graph-RAG services.
    """

    retrieval_service = Mock()
    graph_rag_service = Mock()

    return retrieval_service, graph_rag_service


@pytest.fixture
def tools(
    services: tuple[Mock, Mock],
) -> RetrievalTools:
    """
    RetrievalTools configured with mocked services.
    """

    retrieval_service, graph_rag_service = services

    return RetrievalTools(
        retrieval_service=retrieval_service,
        graph_rag_service=graph_rag_service,
    )


# ============================================================================
# Initialization
# ============================================================================


def test_tools_requires_retrieval_service() -> None:
    """
    RetrievalTools must reject a missing RetrievalService.
    """

    graph_rag_service = Mock()

    with pytest.raises(ValueError, match="retrieval_service"):
        RetrievalTools(
            retrieval_service=None,
            graph_rag_service=graph_rag_service,
        )


def test_tools_requires_graph_rag_service() -> None:
    """
    RetrievalTools must reject a missing GraphRAGService.
    """

    retrieval_service = Mock()

    with pytest.raises(ValueError, match="graph_rag_service"):
        RetrievalTools(
            retrieval_service=retrieval_service,
            graph_rag_service=None,
        )


def test_service_properties_return_injected_dependencies(
    services: tuple[Mock, Mock],
) -> None:
    """
    Injected service instances should remain accessible.
    """

    retrieval_service, graph_rag_service = services

    tools = RetrievalTools(
        retrieval_service=retrieval_service,
        graph_rag_service=graph_rag_service,
    )

    assert tools.retrieval_service is retrieval_service
    assert tools.graph_rag_service is graph_rag_service


# ============================================================================
# Standard Retrieval
# ============================================================================


def test_retrieve_delegates_to_retrieval_service(
    tools: RetrievalTools,
    services: tuple[Mock, Mock],
    retrieval_request: RetrievalRequest,
    built_context: BuiltContext,
) -> None:
    """
    Standard retrieval must delegate to RetrievalService.retrieve().
    """

    retrieval_service, _ = services
    retrieval_service.retrieve.return_value = built_context

    result = tools.retrieve(retrieval_request)

    retrieval_service.retrieve.assert_called_once_with(
        retrieval_request,
    )

    assert result is built_context


def test_retrieve_rejects_invalid_request(
    tools: RetrievalTools,
) -> None:
    """
    Standard retrieval must reject non-RetrievalRequest inputs.
    """

    with pytest.raises(TypeError, match="RetrievalRequest"):
        tools.retrieve("not a retrieval request")


def test_retrieve_rejects_invalid_service_result(
    tools: RetrievalTools,
    services: tuple[Mock, Mock],
    retrieval_request: RetrievalRequest,
) -> None:
    """
    The tool must fail fast if RetrievalService returns the wrong type.
    """

    retrieval_service, _ = services
    retrieval_service.retrieve.return_value = "invalid result"

    with pytest.raises(TypeError, match="BuiltContext"):
        tools.retrieve(retrieval_request)


# ============================================================================
# Graph-RAG
# ============================================================================


def test_graph_rag_delegates_to_graph_rag_service(
    tools: RetrievalTools,
    services: tuple[Mock, Mock],
    graph_rag_request: GraphRAGRequest,
    graph_rag_response: GraphRAGResponse,
) -> None:
    """
    Graph-RAG execution must delegate to GraphRAGService.answer().
    """

    _, graph_rag_service = services
    graph_rag_service.answer.return_value = graph_rag_response

    result = tools.graph_rag(graph_rag_request)

    graph_rag_service.answer.assert_called_once_with(
        graph_rag_request,
    )

    assert result is graph_rag_response


def test_graph_rag_rejects_invalid_request(
    tools: RetrievalTools,
) -> None:
    """
    Graph-RAG execution must reject non-GraphRAGRequest inputs.
    """

    with pytest.raises(TypeError, match="GraphRAGRequest"):
        tools.graph_rag("not a graph rag request")


def test_graph_rag_rejects_invalid_service_result(
    tools: RetrievalTools,
    services: tuple[Mock, Mock],
    graph_rag_request: GraphRAGRequest,
) -> None:
    """
    The tool must fail fast if GraphRAGService returns the wrong type.
    """

    _, graph_rag_service = services
    graph_rag_service.answer.return_value = "invalid result"

    with pytest.raises(TypeError, match="GraphRAGResponse"):
        tools.graph_rag(graph_rag_request)


# ============================================================================
# Execution Metadata
# ============================================================================


def test_execution_metadata_returns_retrieval_service_metadata(
    tools: RetrievalTools,
    services: tuple[Mock, Mock],
) -> None:
    """
    Execution metadata should come directly from RetrievalService.
    """

    retrieval_service, _ = services

    metadata = {
        "latency_ms": 27.4,
        "dense_count": 0,
        "sparse_count": 10,
        "fused_count": 10,
        "graph_count": 3,
        "candidate_count": 13,
        "reranked": True,
        "final_count": 5,
    }

    retrieval_service.last_execution_metadata = metadata

    result = tools.execution_metadata()

    assert result == metadata
    assert result is not metadata


def test_execution_metadata_rejects_invalid_metadata(
    tools: RetrievalTools,
    services: tuple[Mock, Mock],
) -> None:
    """
    Invalid retrieval execution metadata should be rejected.
    """

    retrieval_service, _ = services
    retrieval_service.last_execution_metadata = "invalid"

    with pytest.raises(TypeError, match="must be a dict"):
        tools.execution_metadata()