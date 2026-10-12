from __future__ import annotations

import pytest

from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentError,
    RetrievalAgentState,
    RetrievalAgentStatus,
    RetrievalStrategy,
)
from app.agents.retrieval_agent.prompts import (
    RETRIEVAL_AGENT_SYSTEM_PROMPT,
    build_retrieval_prompt,
)
from app.retrieval.models.retrieval_models import RetrievalRequest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def retrieval_request() -> RetrievalRequest:
    """
    Valid retrieval request for prompt tests.
    """

    return RetrievalRequest(
        query="What is the refund policy?",
        top_k=5,
        collection_name="supportmind_retrieval_pdf_test",
        include_graph=False,
        rerank=True,
        rerank_top_k=3,
    )


@pytest.fixture
def state(
    retrieval_request: RetrievalRequest,
) -> RetrievalAgentState:
    """
    Default Retrieval Agent state.
    """

    return RetrievalAgentState(
        request=retrieval_request,
    )


# ============================================================================
# System Prompt
# ============================================================================


def test_system_prompt_is_non_empty() -> None:
    """
    Retrieval Agent system prompt must contain usable instructions.
    """

    assert RETRIEVAL_AGENT_SYSTEM_PROMPT.strip()


def test_system_prompt_defines_agent_boundary() -> None:
    """
    Prompt must distinguish orchestration from retrieval execution.
    """

    assert "You are NOT the retrieval engine." in (
        RETRIEVAL_AGENT_SYSTEM_PROMPT
    )


def test_system_prompt_mentions_standard_retrieval() -> None:
    """
    Prompt must describe the standard RetrievalService capability.
    """

    assert "Standard Retrieval" in RETRIEVAL_AGENT_SYSTEM_PROMPT
    assert "RetrievalService" in RETRIEVAL_AGENT_SYSTEM_PROMPT


def test_system_prompt_mentions_graph_rag() -> None:
    """
    Prompt must describe the Graph-RAG capability.
    """

    assert "Graph-RAG" in RETRIEVAL_AGENT_SYSTEM_PROMPT
    assert "GraphRAGService" in RETRIEVAL_AGENT_SYSTEM_PROMPT


def test_system_prompt_contains_evidence_rules() -> None:
    """
    Prompt must prevent fabricated retrieval evidence.
    """

    assert "Never:" in RETRIEVAL_AGENT_SYSTEM_PROMPT
    assert "invent retrieved documents" in RETRIEVAL_AGENT_SYSTEM_PROMPT
    assert "fabricate retrieval results" in RETRIEVAL_AGENT_SYSTEM_PROMPT


def test_system_prompt_contains_prompt_injection_protection() -> None:
    """
    Retrieved content must be treated as untrusted data.
    """

    assert "PROMPT INJECTION RESISTANCE" in (
        RETRIEVAL_AGENT_SYSTEM_PROMPT
    )
    assert "untrusted data" in RETRIEVAL_AGENT_SYSTEM_PROMPT


def test_system_prompt_defines_downstream_boundaries() -> None:
    """
    Retrieval Agent must not absorb Resolution/Escalation responsibilities.
    """

    assert "Resolution Agent" in RETRIEVAL_AGENT_SYSTEM_PROMPT
    assert "Escalation Agent" in RETRIEVAL_AGENT_SYSTEM_PROMPT


# ============================================================================
# Prompt Builder
# ============================================================================


def test_build_retrieval_prompt_contains_query(
    state: RetrievalAgentState,
) -> None:
    """
    Prompt must contain the actual retrieval query.
    """

    prompt = build_retrieval_prompt(state)

    assert "What is the refund policy?" in prompt


def test_build_retrieval_prompt_contains_request_configuration(
    state: RetrievalAgentState,
) -> None:
    """
    Prompt must expose the actual RetrievalRequest configuration.
    """

    prompt = build_retrieval_prompt(state)

    assert "top_k: 5" in prompt
    assert "collection_name: supportmind_retrieval_pdf_test" in prompt
    assert "include_graph: False" in prompt
    assert "rerank: True" in prompt
    assert "rerank_top_k: 3" in prompt


def test_build_retrieval_prompt_contains_execution_state(
    state: RetrievalAgentState,
) -> None:
    """
    Prompt must expose current agent execution state.
    """

    prompt = build_retrieval_prompt(state)

    assert "status: pending" in prompt
    assert "strategy: standard" in prompt
    assert "use_graph_rag: False" in prompt
    assert "iteration_count: 0" in prompt


def test_build_retrieval_prompt_contains_metadata(
    state: RetrievalAgentState,
) -> None:
    """
    Prompt must include existing execution metadata.
    """

    state.execution_metadata = {
        "latency_ms": 32.5,
        "sparse_count": 10,
        "final_count": 5,
    }

    prompt = build_retrieval_prompt(state)

    assert "latency_ms: 32.5" in prompt
    assert "sparse_count: 10" in prompt
    assert "final_count: 5" in prompt


def test_build_retrieval_prompt_contains_errors(
    state: RetrievalAgentState,
) -> None:
    """
    Prompt must expose structured retrieval errors.
    """

    state.status = RetrievalAgentStatus.FAILED
    state.errors.append(
        RetrievalAgentError(
            source="retrieval_service",
            error_type="timeout",
            message="Retrieval timed out.",
            recoverable=True,
        )
    )

    prompt = build_retrieval_prompt(state)

    assert "retrieval_service" in prompt
    assert "timeout" in prompt
    assert "Retrieval timed out." in prompt
    assert "recoverable=True" in prompt


def test_build_retrieval_prompt_rejects_invalid_state() -> None:
    """
    Prompt builder must reject arbitrary objects.
    """

    with pytest.raises(
        TypeError,
        match="RetrievalAgentState",
    ):
        build_retrieval_prompt("invalid state")