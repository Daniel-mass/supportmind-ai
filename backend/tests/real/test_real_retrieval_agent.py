from __future__ import annotations

import asyncio
import os

import pytest

from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.retrieval_agent.agents import RetrievalAgent
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
    RetrievalAgentStatus,
    RetrievalStrategy,
)
from app.agents.retrieval_agent.tools.retrieval_tools import (
    RetrievalTools,
)

from app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)

from app.retrieval.graph_rag.evidence_fusion import (
    EvidenceFusion,
)
from app.retrieval.graph_rag.generator import (
    GraphRAGGenerator,
)
from app.retrieval.graph_rag.graph_rag_service import (
    GraphRAGService,
)
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)
from app.retrieval.graph_rag.graph_retriever import (
    create_graph_retriever,
)
from app.retrieval.graph_rag.intelligence_adapter import (
    IntelligenceAdapter,
)
from app.retrieval.graph_rag.providers.ollama import (
    OllamaGenerationConfig,
    OllamaProvider,
)

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)

from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)

from app.retrieval.sparse.sparse_retriever import (
    SparseRetriever,
)


# ============================================================================
# Environment
# ============================================================================

RUN_REAL_RETRIEVAL_AGENT = (
    os.getenv(
        "RUN_REAL_RETRIEVAL_AGENT",
        "0",
    ).strip()
    == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_REAL_RETRIEVAL_AGENT,
    reason=(
        "Set RUN_REAL_RETRIEVAL_AGENT=1 "
        "to run the real Retrieval Agent integration test."
    ),
)


COLLECTION_NAME = os.getenv(
    "SUPPORTMIND_RETRIEVAL_COLLECTION",
    "supportmind_retrieval_pdf_test",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_LLM_MODEL",
    "qwen3:8b",
)


# ============================================================================
# Qdrant
# ============================================================================

def build_qdrant_store():
    """
    Build the existing SupportMind Qdrant store.

    Uses the existing production Qdrant implementation.
    """

    from app.vector_store.qdrant_store import (
        QdrantStore,
    )

    return QdrantStore()


# ============================================================================
# Retrieval Service
# ============================================================================

def build_retrieval_service(
    qdrant_store,
) -> RetrievalService:
    """
    Build the real RetrievalService using the existing
    SupportMind retrieval components.

    Dense retrieval is intentionally disabled.

    The integration test validates:

        Sparse Retrieval
            +
        Graph Retrieval
            ↓
        RetrievalService
    """

    sparse_retriever = (
        SparseRetriever.from_qdrant(
            qdrant_store=qdrant_store,
            collection_name=COLLECTION_NAME,
        )
    )

    graph_retriever = (
        create_graph_retriever()
    )

    config = RetrievalServiceConfig(
        use_dense=False,
        use_sparse=True,
        use_graph=True,
        use_reranker=False,
    )

    return RetrievalService(
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        config=config,
    )


# ============================================================================
# Graph-RAG Service
# ============================================================================

def build_graph_rag_service(
    retrieval_service: RetrievalService,
) -> GraphRAGService:
    """
    Build the existing real Graph-RAG service.

    Architecture:

        RetrievalService
                ↓
        IntelligenceAdapter
                ↓
        EvidenceFusion
                ↓
        GraphRAGGenerator
                ↓
        OllamaProvider
                ↓
        Ollama / Qwen3
                ↓
        GroundednessValidator
    """

    # ------------------------------------------------------------------------
    # Existing Neo4j operational graph
    # ------------------------------------------------------------------------

    operational_graph = (
        create_neo4j_operational_graph()
    )

    # ------------------------------------------------------------------------
    # Intelligence Adapter
    # ------------------------------------------------------------------------

    intelligence_adapter = (
        IntelligenceAdapter(
            operational_graph=operational_graph,
        )
    )

    # ------------------------------------------------------------------------
    # Existing Graph-RAG Ollama Provider
    # ------------------------------------------------------------------------

    ollama_config = OllamaGenerationConfig(
        host=OLLAMA_HOST,
        model_name=OLLAMA_MODEL,
        temperature=0.1,
        max_tokens=1024,
        timeout=120.0,
    )

    ollama_provider = (
        OllamaProvider(
            config=ollama_config,
        )
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # GraphRAGService expects a generator that accepts:
    #
    #     generate(GraphRAGContext)
    #
    # The raw OllamaProvider instead accepts:
    #
    #     generate(
    #         system_prompt=...,
    #         user_prompt=...,
    #     )
    #
    # Therefore GraphRAGGenerator is the required adapter between them.
    # ------------------------------------------------------------------------

    generator = (
        GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=OLLAMA_MODEL,
        )
    )

    # ------------------------------------------------------------------------
    # Evidence Fusion
    # ------------------------------------------------------------------------

    evidence_fusion = (
        EvidenceFusion()
    )

    # ------------------------------------------------------------------------
    # Grounding Validation
    # ------------------------------------------------------------------------

    groundedness_validator = (
        GroundednessValidator()
    )

    # ------------------------------------------------------------------------
    # Graph-RAG Service
    # ------------------------------------------------------------------------

    return GraphRAGService(
        retrieval_service=retrieval_service,
        intelligence_adapter=intelligence_adapter,
        evidence_fusion=evidence_fusion,
        generator=generator,
        groundedness_validator=groundedness_validator,
    )


# ============================================================================
# AutoGen Model Client
# ============================================================================

def build_model_client() -> OllamaChatCompletionClient:
    """
    Build the real AutoGen Ollama model client.

    This client is used by the Retrieval Agent itself.

    It is intentionally separate from the Graph-RAG OllamaProvider.
    """

    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )


# ============================================================================
# AutoGen Client Cleanup
# ============================================================================

def close_model_client(
    model_client: OllamaChatCompletionClient,
) -> None:
    """
    Properly close the asynchronous AutoGen model client.

    OllamaChatCompletionClient.close() is a coroutine.
    """

    close_method = getattr(
        model_client,
        "close",
        None,
    )

    if close_method is None:
        return

    result = close_method()

    if asyncio.iscoroutine(result):
        asyncio.run(result)


# ============================================================================
# Standard Retrieval Agent Integration
# ============================================================================

def test_real_retrieval_agent_standard() -> None:
    """
    Validate the complete Retrieval Agent standard-retrieval path.

    Flow:

        RetrievalRequest
            ↓
        RetrievalAgent
            ↓
        AutoGen / Qwen3
            ↓
        RetrievalTools
            ↓
        RetrievalService
            ↓
        SparseRetriever + GraphRetriever
            ↓
        BuiltContext
    """

    qdrant_store = build_qdrant_store()

    retrieval_service = build_retrieval_service(
        qdrant_store
    )

    graph_rag_service = build_graph_rag_service(
        retrieval_service
    )

    tools = RetrievalTools(
        retrieval_service=retrieval_service,
        graph_rag_service=graph_rag_service,
    )

    model_client = build_model_client()

    agent = RetrievalAgent(
        model_client=model_client,
        retrieval_tools=tools,
    )

    request = RetrievalRequest(
        query=(
            "What information is available "
            "about customer support issues?"
        ),
        top_k=5,
        collection_name=COLLECTION_NAME,
        include_graph=True,
        rerank=False,
    )

    state = RetrievalAgentState(
        request=request,
        use_graph_rag=False,
    )

    try:
        result = asyncio.run(
            agent.run(state)
        )

        print()
        print("=" * 80)
        print(
            "REAL RETRIEVAL AGENT — STANDARD"
        )
        print("=" * 80)

        print(
            f"Status: {result.status.value}"
        )

        print(
            f"Strategy: {result.strategy.value}"
        )

        print(
            f"Evidence IDs: "
            f"{result.evidence_ids}"
        )

        print(
            f"Execution metadata: "
            f"{result.execution_metadata}"
        )

        print(
            f"Errors: {result.errors}"
        )

        if result.context is not None:
            print()
            print(
                f"Selected context items: "
                f"{result.context.selected_count}"
            )

            print(
                f"Context characters: "
                f"{result.context.character_count}"
            )

            print(
                f"Context truncated: "
                f"{result.context.truncated}"
            )

        # --------------------------------------------------------------
        # Assertions
        # --------------------------------------------------------------

        assert (
            result.status
            == RetrievalAgentStatus.COMPLETED
        )

        assert (
            result.strategy
            == RetrievalStrategy.STANDARD
        )

        assert result.context is not None

        assert (
            result.context.selected_count
            > 0
        )

        assert len(
            result.evidence_ids
        ) > 0

        assert not result.errors

    finally:
        close_model_client(
            model_client
        )

        retrieval_service.close()

        qdrant_close = getattr(
            qdrant_store,
            "close",
            None,
        )

        if callable(qdrant_close):
            qdrant_close()


# ============================================================================
# Graph-RAG Retrieval Agent Integration
# ============================================================================

def test_real_retrieval_agent_graph_rag() -> None:
    """
    Validate the complete Retrieval Agent Graph-RAG path.

    Flow:

        RetrievalRequest
            ↓
        RetrievalAgent
            ↓
        AutoGen / Qwen3
            ↓
        RetrievalTools
            ↓
        GraphRAGService
            ↓
        RetrievalService
            ↓
        Sparse Retrieval + Graph Retrieval
            ↓
        IntelligenceAdapter
            ↓
        Neo4j Operational Graph
            ↓
        Evidence Fusion
            ↓
        GraphRAGGenerator
            ↓
        OllamaProvider / Qwen3
            ↓
        Groundedness Validation
    """

    qdrant_store = build_qdrant_store()

    retrieval_service = build_retrieval_service(
        qdrant_store
    )

    graph_rag_service = build_graph_rag_service(
        retrieval_service
    )

    tools = RetrievalTools(
        retrieval_service=retrieval_service,
        graph_rag_service=graph_rag_service,
    )

    model_client = build_model_client()

    agent = RetrievalAgent(
        model_client=model_client,
        retrieval_tools=tools,
    )

    request = RetrievalRequest(
        query=(
            "What does the support data show "
            "about ticket priorities, workflow "
            "activity, and operational delays?"
        ),
        top_k=5,
        collection_name=COLLECTION_NAME,
        include_graph=True,
        rerank=False,
    )

    state = RetrievalAgentState(
        request=request,
        use_graph_rag=True,
    )

    try:
        result = asyncio.run(
            agent.run(state)
        )

        print()
        print("=" * 80)
        print(
            "REAL RETRIEVAL AGENT — GRAPH-RAG"
        )
        print("=" * 80)

        print(
            f"Status: {result.status.value}"
        )

        print(
            f"Strategy: {result.strategy.value}"
        )

        print(
            f"Evidence IDs: "
            f"{result.evidence_ids}"
        )

        print(
            f"Execution metadata: "
            f"{result.execution_metadata}"
        )

        print(
            f"Errors: {result.errors}"
        )

        if result.graph_rag_response is not None:
            response = (
                result.graph_rag_response
            )

            print()
            print("Graph-RAG answer:")
            print(
                response.answer
            )

            print()
            print(
                "Grounding status:",
                response.grounding.status.value,
            )

            print(
                "Grounding valid:",
                response.grounding.valid,
            )

            print(
                "Cited evidence IDs:",
                response.grounding.cited_evidence_ids,
            )

            print(
                "Invalid citation IDs:",
                response.grounding.invalid_citation_ids,
            )

            print(
                "Unsupported claims:",
                response.grounding.unsupported_claims,
            )

            print(
                "Grounding issues:",
                response.grounding.issues,
            )

        # --------------------------------------------------------------
        # Assertions
        # --------------------------------------------------------------

        assert (
            result.status
            == RetrievalAgentStatus.COMPLETED
        )

        assert (
            result.strategy
            == RetrievalStrategy.GRAPH_RAG
        )

        assert (
            result.graph_rag_response
            is not None
        )

        assert len(
            result.evidence_ids
        ) > 0

        assert (
            result.graph_rag_response
            .grounding
            .valid
            is True
        )

        assert not result.errors

    finally:
        close_model_client(
            model_client
        )

        retrieval_service.close()

        qdrant_close = getattr(
            qdrant_store,
            "close",
            None,
        )

        if callable(qdrant_close):
            qdrant_close()