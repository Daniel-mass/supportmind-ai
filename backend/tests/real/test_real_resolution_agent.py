"""
Real Resolution Agent integration test.

This test connects the Resolution Agent to the actual SupportMind
Classification Agent, Retrieval Agent, Graph-RAG stack, and Ollama/Qwen3.

It is intentionally gated behind RUN_REAL_RESOLUTION=1 because the
integration test loads ML models, connects to Qdrant/Neo4j, and invokes Ollama.

Important async design:

    One asyncio event loop is used for the complete AutoGen pipeline.

    Classification
        ↓
    Retrieval Agent
        ↓
    Resolution Agent
        ↓
    Cleanup

The same OllamaChatCompletionClient is therefore never reused across
different asyncio event loops.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from autogen_ext.models.ollama import OllamaChatCompletionClient

from app.agents.classification_agent.agents import ClassificationAgent
from app.agents.classification_agent.contracts.state import (
    ClassificationState,
    ClassificationStatus,
    PriorityFeatures,
)
from app.agents.classification_agent.tools.classification_tools import (
    ConversationComplexityTool,
    PriorityDetectionTool,
    SentimentTool,
    TicketCategoryTool,
    TopicModelTool,
)

from app.agents.retrieval_agent.agents import RetrievalAgent
from app.agents.retrieval_agent.contracts.state import (
    RetrievalAgentState,
    RetrievalAgentStatus,
)
from app.agents.retrieval_agent.tools.retrieval_tools import (
    RetrievalTools,
)

from app.agents.resolution_agent.agents import (
    ResolutionAgent,
)
from app.agents.resolution_agent.contracts.state import (
    ResolutionAgentState,
    ResolutionAgentStatus,
)
from app.agents.resolution_agent.tools.resolution_tools import (
    ResolutionTools,
)

from app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)

from app.nlp.conversation_intelligence.predictor import (
    ConversationComplexityPredictor,
)
from app.nlp.priority_detection.predictor import (
    PriorityDetectionPredictor,
)
from app.nlp.sentimental_help_analysis.predictor import (
    SentimentPredictor,
)
from app.nlp.ticket_category.predictor import (
    TicketCategoryPredictor,
)
from app.nlp.topic_modeling.predictor import (
    TopicModelPredictor,
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

RUN_REAL_RESOLUTION = (
    os.getenv(
        "RUN_REAL_RESOLUTION",
        "0",
    ).strip()
    == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_REAL_RESOLUTION,
    reason=(
        "Set RUN_REAL_RESOLUTION=1 "
        "to run the real Resolution Agent integration test."
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
# Real Classification Agent
# ============================================================================

def build_classification_agent() -> ClassificationAgent:
    """
    Build the real Classification Agent using all five
    SupportMind NLP predictors.
    """

    category_predictor = TicketCategoryPredictor()
    priority_predictor = PriorityDetectionPredictor()
    sentiment_predictor = SentimentPredictor()
    complexity_predictor = ConversationComplexityPredictor()
    topic_predictor = TopicModelPredictor()

    category_tool = TicketCategoryTool(
        predictor=category_predictor,
    )

    priority_tool = PriorityDetectionTool(
        predictor=priority_predictor,
    )

    sentiment_tool = SentimentTool(
        predictor=sentiment_predictor,
    )

    complexity_tool = ConversationComplexityTool(
        predictor=complexity_predictor,
    )

    topic_tool = TopicModelTool(
        predictor=topic_predictor,
    )

    return ClassificationAgent(
        category_tool=category_tool,
        priority_tool=priority_tool,
        sentiment_tool=sentiment_tool,
        complexity_tool=complexity_tool,
        topic_tool=topic_tool,
    )


# ============================================================================
# Qdrant
# ============================================================================

def build_qdrant_store():
    """
    Build the existing SupportMind Qdrant store.
    """

    from app.vector_store.qdrant_store import QdrantStore

    return QdrantStore()


# ============================================================================
# Retrieval Service
# ============================================================================

def build_retrieval_service(
    qdrant_store,
) -> RetrievalService:
    """
    Build the real SupportMind RetrievalService.

    Dense retrieval is intentionally disabled.

    The real integration validates:

        Sparse Retrieval
            +
        Graph Retrieval
            ↓
        RetrievalService
    """

    sparse_retriever = SparseRetriever.from_qdrant(
        qdrant_store=qdrant_store,
        collection_name=COLLECTION_NAME,
    )

    graph_retriever = create_graph_retriever()

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
    """

    operational_graph = create_neo4j_operational_graph()

    intelligence_adapter = IntelligenceAdapter(
        operational_graph=operational_graph,
    )

    ollama_config = OllamaGenerationConfig(
        host=OLLAMA_HOST,
        model_name=OLLAMA_MODEL,
        temperature=0.1,
        max_tokens=1024,
        timeout=120.0,
    )

    ollama_provider = OllamaProvider(
        config=ollama_config,
    )

    generator = GraphRAGGenerator(
        provider=ollama_provider,
        provider_name="ollama",
        model_name=OLLAMA_MODEL,
    )

    evidence_fusion = EvidenceFusion()

    groundedness_validator = GroundednessValidator()

    return GraphRAGService(
        retrieval_service=retrieval_service,
        intelligence_adapter=intelligence_adapter,
        evidence_fusion=evidence_fusion,
        generator=generator,
        groundedness_validator=groundedness_validator,
    )


# ============================================================================
# AutoGen Ollama Model Client
# ============================================================================

def build_model_client() -> OllamaChatCompletionClient:
    """
    Build the real AutoGen Ollama model client.

    The client is shared by the Retrieval Agent and Resolution Agent.

    Both agents execute inside the SAME asyncio event loop.
    """

    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )


# ============================================================================
# AutoGen Client Cleanup
# ============================================================================

async def close_model_client(
    model_client: OllamaChatCompletionClient,
) -> None:
    """
    Properly close the asynchronous AutoGen Ollama client.

    IMPORTANT:

    This function is async so that cleanup occurs inside the SAME
    event loop used by the Retrieval and Resolution Agents.

    It must NOT call asyncio.run().
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
        await result


# ============================================================================
# Real Classification Execution
# ============================================================================

def run_real_classification(
    agent: ClassificationAgent,
) -> ClassificationState:
    """
    Execute the real Classification Agent for the same ticket
    that will be passed through the Resolution pipeline.

    ClassificationAgent.run() returns ClassificationState.

    The combined ClassificationResult is stored inside:

        result.predictions
    """

    state = ClassificationState(
        request_id="real-resolution-test-001",
        ticket_text=(
            "I was charged twice for the same payment. "
            "The amount was deducted from my account twice and "
            "I need the duplicate charge resolved."
        ),
        priority_features=PriorityFeatures(
            issue_proj="billing",
            issue_type="payment",
            created_year=2026,
            created_month=9,
            created_dayofweek=6,
            created_hour=10,
            issue_contr_count=1,
        ),
    )

    result = agent.run(state)

    # ------------------------------------------------------------------------
    # Validate Classification Agent workflow state
    # ------------------------------------------------------------------------

    assert isinstance(
        result,
        ClassificationState,
    )

    assert result.status is ClassificationStatus.COMPLETED

    assert not result.errors

    # ------------------------------------------------------------------------
    # Validate combined ClassificationResult
    # ------------------------------------------------------------------------

    assert result.predictions.predictions_requested == 5

    assert result.predictions.predictions_completed == 5

    return result


# ============================================================================
# Real Retrieval Agent Execution
# ============================================================================

async def run_real_retrieval(
    retrieval_agent: RetrievalAgent,
    *,
    ticket_text: str,
):
    """
    Execute the real Retrieval Agent using Graph-RAG.

    This function runs inside the same asyncio event loop that will
    subsequently execute the Resolution Agent.
    """

    retrieval_state = RetrievalAgentState(
        request=RetrievalRequest(
            query=ticket_text,
            top_k=5,
            collection_name=COLLECTION_NAME,
            include_graph=True,
            rerank=False,
        ),
        use_graph_rag=True,
    )

    result = await retrieval_agent.run(
        retrieval_state
    )

    # ------------------------------------------------------------------------
    # Validate Retrieval Agent
    # ------------------------------------------------------------------------

    assert (
        result.status
        is RetrievalAgentStatus.COMPLETED
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

    return result


# ============================================================================
# Real Resolution Pipeline
# ============================================================================

async def run_real_resolution_pipeline(
    *,
    retrieval_agent: RetrievalAgent,
    resolution_agent: ResolutionAgent,
    model_client: OllamaChatCompletionClient,
    classification_result,
    classification_signals,
    ticket_text: str,
):
    """
    Execute Retrieval Agent and Resolution Agent inside ONE asyncio
    event loop.

    This is the critical fix for the Event loop is closed failure.
    """

    # ------------------------------------------------------------------------
    # 1. Execute Retrieval Agent
    # ------------------------------------------------------------------------

    retrieval_result = await run_real_retrieval(
        retrieval_agent,
        ticket_text=ticket_text,
    )

    # ------------------------------------------------------------------------
    # 2. Extract Graph-RAG response
    # ------------------------------------------------------------------------

    graph_rag_response = (
        retrieval_result.graph_rag_response
    )

    assert graph_rag_response is not None

    # ------------------------------------------------------------------------
    # 3. Convert Graph-RAG evidence into Resolution evidence
    # ------------------------------------------------------------------------

    resolution_evidence = (
        ResolutionTools.graph_rag_evidence(
            graph_rag_response
        )
    )

    assert len(
        resolution_evidence
    ) > 0

    resolution_evidence_ids = (
        ResolutionTools.graph_rag_evidence_ids(
            graph_rag_response
        )
    )

    assert resolution_evidence_ids

    assert set(
        resolution_evidence_ids
    ) == {
        evidence.evidence_id
        for evidence in resolution_evidence
    }

    # ------------------------------------------------------------------------
    # 4. Validate Graph-RAG grounding
    # ------------------------------------------------------------------------

    grounding = (
        ResolutionTools.graph_rag_grounding(
            graph_rag_response
        )
    )

    assert grounding["valid"] is True

    assert isinstance(
        grounding["cited_evidence_ids"],
        list,
    )

    # ------------------------------------------------------------------------
    # 5. Build Resolution Agent state
    # ------------------------------------------------------------------------

    resolution_state = ResolutionAgentState(
        request=ticket_text,
        ticket_id="real-resolution-test-001",
        classification_signals=classification_signals,
        evidence=resolution_evidence,
    )

    assert resolution_state.has_evidence is True

    assert len(
        resolution_state.classification_signals
    ) == 5

    # ------------------------------------------------------------------------
    # 6. Execute Resolution Agent
    #
    # IMPORTANT:
    #
    # This executes in the SAME event loop as the Retrieval Agent.
    # ------------------------------------------------------------------------

    resolution_result = await resolution_agent.run(
        resolution_state
    )

    # ------------------------------------------------------------------------
    # 7. Validate Resolution Agent
    # ------------------------------------------------------------------------

    assert (
        resolution_result.status
        is ResolutionAgentStatus.COMPLETED
    ), (
        "Resolution Agent failed: "
        f"{resolution_result.errors}"
    )

    assert (
        resolution_result.iteration_count
        == 1
    )

    assert not resolution_result.errors

    # ------------------------------------------------------------------------
    # 8. Validate textual resolution response
    # ------------------------------------------------------------------------

    agent_response = (
        resolution_result
        .execution_metadata
        .get("agent_response")
    )

    assert isinstance(
        agent_response,
        str,
    )

    assert agent_response.strip()

    # ------------------------------------------------------------------------
    # 9. Validate evidence remained available
    # ------------------------------------------------------------------------

    assert resolution_result.has_evidence

    assert len(
        resolution_result.evidence
    ) > 0

    return (
        retrieval_result,
        resolution_result,
        graph_rag_response,
        agent_response,
    )


# ============================================================================
# Real Resolution Agent Integration
# ============================================================================

def test_real_resolution_agent() -> None:
    """
    Validate the complete real Resolution Agent pipeline.

    Flow:

        Real Classification Agent
                    ↓
        ClassificationState
                    ↓
        state.predictions
                    ↓
        ClassificationResult
                    ↓
        ResolutionTools
                    ↓
        Resolution Classification Signals
                    +
        Real Retrieval Agent
                    ↓
        Graph-RAG
                    ↓
        GraphRAGResponse
                    ↓
        ResolutionTools
                    ↓
        Resolution Evidence
                    ↓
        ResolutionAgentState
                    ↓
        Resolution Agent / Qwen3
                    ↓
        Textual Resolution Recommendation


    Async architecture:

        ONE asyncio.run()
                │
                ├── Retrieval Agent
                │
                ├── Resolution Agent
                │
                └── Ollama client cleanup
    """

    # ------------------------------------------------------------------------
    # 1. Build real Classification Agent
    # ------------------------------------------------------------------------

    classification_agent = build_classification_agent()

    # ------------------------------------------------------------------------
    # 2. Execute real classification
    # ------------------------------------------------------------------------

    classification_state = run_real_classification(
        classification_agent
    )

    # ------------------------------------------------------------------------
    # 3. Extract the actual ClassificationResult
    # ------------------------------------------------------------------------

    classification_result = (
        classification_state.predictions
    )

    classification_result = (
        ResolutionTools.validate_classification_result(
            classification_result
        )
    )

    # ------------------------------------------------------------------------
    # 4. Convert classification predictions into Resolution signals
    # ------------------------------------------------------------------------

    classification_signals = (
        ResolutionTools.classification_signals(
            classification_result
        )
    )

    assert len(
        classification_signals
    ) == 5

    # ------------------------------------------------------------------------
    # 5. Build real Qdrant
    # ------------------------------------------------------------------------

    qdrant_store = build_qdrant_store()

    # ------------------------------------------------------------------------
    # 6. Build real RetrievalService
    # ------------------------------------------------------------------------

    retrieval_service = build_retrieval_service(
        qdrant_store
    )

    # ------------------------------------------------------------------------
    # 7. Build real Graph-RAG
    # ------------------------------------------------------------------------

    graph_rag_service = build_graph_rag_service(
        retrieval_service
    )

    # ------------------------------------------------------------------------
    # 8. Build real RetrievalTools
    # ------------------------------------------------------------------------

    retrieval_tools = RetrievalTools(
        retrieval_service=retrieval_service,
        graph_rag_service=graph_rag_service,
    )

    # ------------------------------------------------------------------------
    # 9. Build ONE shared AutoGen Ollama client
    # ------------------------------------------------------------------------

    model_client = build_model_client()

    # ------------------------------------------------------------------------
    # 10. Build Retrieval Agent
    # ------------------------------------------------------------------------

    retrieval_agent = RetrievalAgent(
        model_client=model_client,
        retrieval_tools=retrieval_tools,
    )

    # ------------------------------------------------------------------------
    # 11. Build Resolution Agent
    # ------------------------------------------------------------------------

    resolution_agent = ResolutionAgent(
        model_client=model_client,
    )

    ticket_text = (
        "I was charged twice for the same payment. "
        "The amount was deducted from my account twice and "
        "I need the duplicate charge resolved."
    )

    try:
        # --------------------------------------------------------------------
        # 12. Run the complete async pipeline in ONE event loop
        # --------------------------------------------------------------------

        (
            retrieval_result,
            resolution_result,
            graph_rag_response,
            agent_response,
        ) = asyncio.run(
            run_real_resolution_pipeline(
                retrieval_agent=retrieval_agent,
                resolution_agent=resolution_agent,
                model_client=model_client,
                classification_result=classification_result,
                classification_signals=classification_signals,
                ticket_text=ticket_text,
            )
        )

        # --------------------------------------------------------------------
        # 13. Print complete real pipeline output
        # --------------------------------------------------------------------

        print()
        print("=" * 80)
        print(
            "REAL RESOLUTION AGENT INTEGRATION RESULT"
        )
        print("=" * 80)

        print()
        print("Classification")
        print("-" * 80)

        print(
            "Status:",
            classification_state.status.value,
        )

        print(
            "Category:",
            classification_result.category.value,
            f"(confidence="
            f"{classification_result.category.confidence})",
        )

        print(
            "Priority:",
            classification_result.priority.value,
            f"(confidence="
            f"{classification_result.priority.confidence})",
        )

        print(
            "Sentiment:",
            classification_result.sentiment.value,
            f"(confidence="
            f"{classification_result.sentiment.confidence})",
        )

        print(
            "Complexity:",
            classification_result.complexity.value,
            f"(confidence="
            f"{classification_result.complexity.confidence})",
        )

        print(
            "Topic:",
            classification_result.topic.value,
            f"(confidence="
            f"{classification_result.topic.confidence})",
        )

        print()
        print("Classification → Resolution Signals")
        print("-" * 80)

        for signal in classification_signals:
            print(
                f"{signal.name}: "
                f"{signal.value} "
                f"(confidence={signal.confidence}, "
                f"source={signal.source})"
            )

        print()
        print("Retrieval Agent")
        print("-" * 80)

        print(
            "Status:",
            retrieval_result.status.value,
        )

        print(
            "Evidence count:",
            len(retrieval_result.evidence_ids),
        )

        print()
        print("Graph-RAG")
        print("-" * 80)

        print(
            "Provider:",
            graph_rag_response.provider,
        )

        print(
            "Model:",
            graph_rag_response.model_name,
        )

        print(
            "Evidence count:",
            len(graph_rag_response.evidence),
        )

        print(
            "Grounding valid:",
            graph_rag_response.grounding.valid,
        )

        print(
            "Cited evidence IDs:",
            graph_rag_response.grounding.cited_evidence_ids,
        )

        print()
        print("Resolution Agent")
        print("-" * 80)

        print(
            "Status:",
            resolution_result.status.value,
        )

        print(
            "Iteration count:",
            resolution_result.iteration_count,
        )

        print()
        print("Resolution response:")
        print(agent_response)

        print()
        print("=" * 80)

    finally:
        # --------------------------------------------------------------------
        # 14. Synchronous resources
        #
        # These do not use the AutoGen event loop.
        # --------------------------------------------------------------------

        retrieval_service.close()

        qdrant_close = getattr(
            qdrant_store,
            "close",
            None,
        )

        if callable(qdrant_close):
            qdrant_close()
