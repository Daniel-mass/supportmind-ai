from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.retrieval_agent.agents import (
    RETRIEVAL_AGENT_NAME,
    RetrievalAgent,
)
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
    RetrievalAgentStatus,
    RetrievalStrategy,
)
from app.agents.retrieval_agent.tools.retrieval_tools import RetrievalTools
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GraphRAGRequest,
    GraphRAGResponse,
    GroundingResult,
    GroundingStatus,
)
from app.retrieval.context.context_models import (
    BuiltContext,
    ContextItem,
    ContextSource,
)
from app.retrieval.models.retrieval_models import RetrievalRequest


def make_retrieval_request() -> RetrievalRequest:
    return RetrievalRequest(
        query="How do I reset my account password?",
        top_k=5,
        collection_name="supportmind_test",
        include_graph=False,
        rerank=False,
    )


def make_state(
    *,
    use_graph_rag: bool = False,
) -> RetrievalAgentState:
    return RetrievalAgentState(
        request=make_retrieval_request(),
        use_graph_rag=use_graph_rag,
    )


def make_built_context() -> BuiltContext:
    return BuiltContext(
        query="How do I reset my account password?",
        sections=[],
        selected_items=[
            ContextItem(
                id="doc-1",
                text="Password reset instructions.",
                source=ContextSource.BM25,
                rank=1,
                score=0.95,
                metadata={},
            )
        ],
        total_candidates=1,
        selected_count=1,
        character_count=34,
        truncated=False,
    )


def make_graph_rag_response() -> GraphRAGResponse:
    evidence = [
        GraphRAGEvidence(
            evidence_id="E1",
            source=GraphRAGEvidenceSource.RETRIEVAL_SPARSE,
            evidence_type=GraphRAGEvidenceType.DOCUMENT,
            text="Password reset instructions.",
            source_reference="doc-1",
            score=0.95,
            metadata={},
        )
    ]

    grounding = GroundingResult(
        status=GroundingStatus.GROUNDED,
        valid=True,
        cited_evidence_ids=["E1"],
        invalid_citation_ids=[],
        unsupported_claims=[],
        issues=[],
        metadata={},
    )

    return GraphRAGResponse(
        query="How do I reset my account password?",
        answer="Follow the password reset instructions. [E1]",
        evidence=evidence,
        grounding=grounding,
        provider="ollama",
        model_name="qwen3:8b",
        latency_ms=100.0,
        metadata={},
    )


def make_mock_model_client() -> MagicMock:
    return MagicMock()


def make_mock_tools() -> MagicMock:
    tools = MagicMock(spec=RetrievalTools)

    tools.retrieve.return_value = make_built_context()
    tools.graph_rag.return_value = make_graph_rag_response()
    tools.execution_metadata.return_value = {
        "dense_count": 0,
        "sparse_count": 1,
        "graph_count": 0,
        "final_count": 1,
    }

    return tools


def make_agent() -> RetrievalAgent:
    model_client = make_mock_model_client()
    tools = make_mock_tools()

    return RetrievalAgent(
        model_client=model_client,
        retrieval_tools=tools,
    )


def make_autogen_result(text: str = "Retrieval strategy selected.") -> SimpleNamespace:
    from autogen_agentchat.messages import TextMessage

    return SimpleNamespace(
        messages=[
            TextMessage(
                source="retrieval_agent",
                content=text,
            )
        ]
    )


def test_agent_name_is_correct() -> None:
    agent = make_agent()

    assert agent.name == RETRIEVAL_AGENT_NAME


def test_agent_exposes_autogen_agent() -> None:
    agent = make_agent()

    assert agent.agent is not None


def test_agent_exposes_retrieval_tools() -> None:
    tools = make_mock_tools()

    agent = RetrievalAgent(
        model_client=make_mock_model_client(),
        retrieval_tools=tools,
    )

    assert agent.tools is tools


def test_constructor_rejects_missing_model_client() -> None:
    tools = make_mock_tools()

    with pytest.raises(ValueError, match="model_client"):
        RetrievalAgent(
            model_client=None,
            retrieval_tools=tools,
        )


def test_constructor_rejects_missing_tools() -> None:
    with pytest.raises(ValueError, match="retrieval_tools"):
        RetrievalAgent(
            model_client=make_mock_model_client(),
            retrieval_tools=None,
        )


def test_constructor_rejects_invalid_tools_type() -> None:
    with pytest.raises(TypeError, match="RetrievalTools"):
        RetrievalAgent(
            model_client=make_mock_model_client(),
            retrieval_tools=object(),
        )


def test_build_task_accepts_valid_state() -> None:
    agent = make_agent()
    state = make_state()

    task = agent.build_task(state)

    assert isinstance(task, str)
    assert state.request.query in task


def test_build_task_rejects_invalid_state() -> None:
    agent = make_agent()

    with pytest.raises(TypeError, match="RetrievalAgentState"):
        agent.build_task(object())


def test_select_strategy_standard() -> None:
    agent = make_agent()
    state = make_state(use_graph_rag=False)

    assert agent.select_strategy(state) == RetrievalStrategy.STANDARD


def test_select_strategy_graph_rag() -> None:
    agent = make_agent()
    state = make_state(use_graph_rag=True)

    assert agent.select_strategy(state) == RetrievalStrategy.GRAPH_RAG


def test_build_graph_rag_request_preserves_query() -> None:
    agent = make_agent()
    state = make_state(use_graph_rag=True)

    request = agent._build_graph_rag_request(state)

    assert isinstance(request, GraphRAGRequest)
    assert request.query == state.request.query
    assert request.retrieval_request == state.request
    assert request.include_intelligence is True
    assert request.include_workflow is True


def test_standard_execution_updates_state() -> None:
    agent = make_agent()
    state = make_state()

    agent._update_standard_result(state)

    assert state.context is not None
    assert state.graph_rag_response is None
    assert state.evidence_ids == ["doc-1"]
    assert state.execution_metadata["sparse_count"] == 1


def test_graph_rag_execution_updates_state() -> None:
    agent = make_agent()
    state = make_state(use_graph_rag=True)

    agent._update_graph_rag_result(state)

    assert state.graph_rag_response is not None
    assert state.context is None
    assert state.evidence_ids == ["E1"]
    assert state.execution_metadata["provider"] == "ollama"
    assert state.execution_metadata["grounding_status"] == "grounded"


def test_extract_final_response_returns_text() -> None:
    result = make_autogen_result("Evidence retrieved.")

    response = RetrievalAgent._extract_final_response(result)

    assert response == "Evidence retrieved."


def test_extract_final_response_returns_none_for_empty_result() -> None:
    result = SimpleNamespace(messages=[])

    response = RetrievalAgent._extract_final_response(result)

    assert response is None


def test_extract_final_response_returns_none_without_messages() -> None:
    result = SimpleNamespace()

    response = RetrievalAgent._extract_final_response(result)

    assert response is None


def test_standard_run_completes() -> None:
    agent = make_agent()
    state = make_state()

    agent._agent.run = AsyncMock(
        return_value=make_autogen_result()
    )

    result = asyncio.run(agent.run(state))

    assert result.status == RetrievalAgentStatus.COMPLETED
    assert result.strategy == RetrievalStrategy.STANDARD
    assert result.context is not None
    assert result.evidence_ids == ["doc-1"]
    agent._agent.run.assert_awaited_once()


def test_graph_rag_run_completes() -> None:
    agent = make_agent()
    state = make_state(use_graph_rag=True)

    agent._agent.run = AsyncMock(
        return_value=make_autogen_result()
    )

    result = asyncio.run(agent.run(state))

    assert result.status == RetrievalAgentStatus.COMPLETED
    assert result.strategy == RetrievalStrategy.GRAPH_RAG
    assert result.graph_rag_response is not None
    assert result.evidence_ids == ["E1"]
    agent._agent.run.assert_awaited_once()


def test_run_increments_iteration_count() -> None:
    agent = make_agent()
    state = make_state()

    agent._agent.run = AsyncMock(
        return_value=make_autogen_result()
    )

    result = asyncio.run(agent.run(state))

    assert result.iteration_count == 1


def test_run_records_autogen_failure() -> None:
    agent = make_agent()
    state = make_state()

    agent._agent.run = AsyncMock(
        side_effect=RuntimeError("LLM unavailable")
    )

    result = asyncio.run(agent.run(state))

    assert result.status == RetrievalAgentStatus.FAILED
    assert len(result.errors) == 1
    assert result.errors[0].source == RETRIEVAL_AGENT_NAME
    assert result.errors[0].error_type == "RuntimeError"
    assert "LLM unavailable" in result.errors[0].message


def test_run_records_empty_autogen_response() -> None:
    agent = make_agent()
    state = make_state()

    agent._agent.run = AsyncMock(
        return_value=SimpleNamespace(messages=[])
    )

    result = asyncio.run(agent.run(state))

    assert result.status == RetrievalAgentStatus.FAILED
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "RuntimeError"


def test_run_rejects_invalid_state() -> None:
    agent = make_agent()

    with pytest.raises(TypeError, match="RetrievalAgentState"):
        asyncio.run(agent.run(object()))