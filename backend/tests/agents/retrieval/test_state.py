from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentError,
    RetrievalAgentState,
    RetrievalAgentStatus,
    RetrievalStrategy,
)
from app.retrieval.models.retrieval_models import RetrievalRequest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    """
    Valid retrieval request used throughout the state tests.
    """

    return RetrievalRequest(
        query="How do I request a refund?",
        top_k=5,
        collection_name="supportmind_retrieval_pdf_test",
    )


# ============================================================================
# Status / Strategy
# ============================================================================


def test_retrieval_agent_status_values() -> None:
    """
    Retrieval Agent status enum should expose the expected lifecycle states.
    """

    assert RetrievalAgentStatus.PENDING.value == "pending"
    assert RetrievalAgentStatus.RUNNING.value == "running"
    assert RetrievalAgentStatus.COMPLETED.value == "completed"
    assert RetrievalAgentStatus.FAILED.value == "failed"


def test_retrieval_strategy_values() -> None:
    """
    Retrieval strategy enum should expose the supported execution paths.
    """

    assert RetrievalStrategy.STANDARD.value == "standard"
    assert RetrievalStrategy.GRAPH_RAG.value == "graph_rag"


# ============================================================================
# Error Contract
# ============================================================================


def test_retrieval_agent_error_defaults() -> None:
    """
    RetrievalAgentError should create correctly with its default
    recoverability value.
    """

    error = RetrievalAgentError(
        source="retrieval_service",
        error_type="retrieval_failure",
        message="Retrieval failed.",
    )

    assert error.source == "retrieval_service"
    assert error.error_type == "retrieval_failure"
    assert error.message == "Retrieval failed."
    assert error.recoverable is False


def test_retrieval_agent_error_recoverable() -> None:
    """
    RetrievalAgentError should preserve explicit recoverability.
    """

    error = RetrievalAgentError(
        source="graph_rag_service",
        error_type="timeout",
        message="Graph-RAG request timed out.",
        recoverable=True,
    )

    assert error.recoverable is True


def test_retrieval_agent_error_rejects_empty_source() -> None:
    """
    Empty error sources must be rejected.
    """

    with pytest.raises(ValidationError):
        RetrievalAgentError(
            source="",
            error_type="failure",
            message="Something failed.",
        )


# ============================================================================
# State Defaults
# ============================================================================


def test_retrieval_agent_state_defaults(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    State should initialize with safe execution defaults.
    """

    state = RetrievalAgentState(
        request=retrieval_request,
    )

    assert state.request is retrieval_request
    assert state.use_graph_rag is False
    assert state.status == RetrievalAgentStatus.PENDING
    assert state.strategy == RetrievalStrategy.STANDARD
    assert state.iteration_count == 0
    assert state.context is None
    assert state.graph_rag_response is None
    assert state.evidence_ids == []
    assert state.execution_metadata == {}
    assert state.errors == []


# ============================================================================
# State Validation
# ============================================================================


def test_retrieval_agent_state_requires_retrieval_request() -> None:
    """
    Retrieval Agent state must always have a validated RetrievalRequest.
    """

    with pytest.raises(ValidationError):
        RetrievalAgentState()


def test_retrieval_agent_state_rejects_negative_iteration_count(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    Iteration count cannot be negative.
    """

    with pytest.raises(ValidationError):
        RetrievalAgentState(
            request=retrieval_request,
            iteration_count=-1,
        )


def test_retrieval_agent_state_rejects_unknown_fields(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    State contracts must reject accidental undeclared fields.
    """

    with pytest.raises(ValidationError):
        RetrievalAgentState(
            request=retrieval_request,
            unexpected_field="invalid",
        )


# ============================================================================
# Result State
# ============================================================================


def test_retrieval_agent_state_accepts_execution_metadata(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    Execution metadata should be preserved without imposing
    provider-specific structure.
    """

    metadata = {
        "latency_ms": 42.7,
        "dense_count": 0,
        "sparse_count": 10,
        "graph_count": 4,
        "reranked": True,
        "final_count": 5,
    }

    state = RetrievalAgentState(
        request=retrieval_request,
        status=RetrievalAgentStatus.COMPLETED,
        execution_metadata=metadata,
    )

    assert state.execution_metadata == metadata


def test_retrieval_agent_state_accepts_errors(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    Structured Retrieval Agent errors should be stored directly in state.
    """

    error = RetrievalAgentError(
        source="retrieval_service",
        error_type="timeout",
        message="Retrieval timed out.",
        recoverable=True,
    )

    state = RetrievalAgentState(
        request=retrieval_request,
        status=RetrievalAgentStatus.FAILED,
        errors=[error],
    )

    assert len(state.errors) == 1
    assert state.errors[0].error_type == "timeout"
    assert state.errors[0].recoverable is True


# ============================================================================
# Convenience Properties
# ============================================================================


def test_successful_property(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    successful should reflect the completed state only.
    """

    state = RetrievalAgentState(
        request=retrieval_request,
        status=RetrievalAgentStatus.COMPLETED,
    )

    assert state.successful is True


def test_successful_property_false_for_failed_state(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    Failed executions must not be reported as successful.
    """

    state = RetrievalAgentState(
        request=retrieval_request,
        status=RetrievalAgentStatus.FAILED,
    )

    assert state.successful is False


def test_has_results_false_without_results(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    Empty retrieval state should report no results.
    """

    state = RetrievalAgentState(
        request=retrieval_request,
    )

    assert state.has_results is False


def test_has_results_false_for_completed_empty_state(
    retrieval_request: RetrievalRequest,
) -> None:
    """
    Completion alone does not mean useful retrieval output exists.
    """

    state = RetrievalAgentState(
        request=retrieval_request,
        status=RetrievalAgentStatus.COMPLETED,
    )

    assert state.has_results is False