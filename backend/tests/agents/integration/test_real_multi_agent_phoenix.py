r"""
Real end-to-end SupportMind multi-agent integration test.

This test executes the actual SupportMind agent workflow:

    User Request
        ↓
    Coordinator
        ↓
    Router
        ↓
    Orchestrator
        ↓
    Classification Agent
        ↓
    Retrieval Agent
        ↓
    Graph-RAG
        ↓
    Resolution Agent
        ↓
    Escalation Agent
        ↓
    Response Composer Agent
        ↓
    Coordinator
        ↓
    Router
        ↓
    Final Response

Phoenix traces the complete workflow.

MLflow evaluates the completed CoordinatorState produced by
the real multi-agent workflow.

The priority classifier requires structured ticket features.
For this integration test, those features are intentionally
fabricated as synthetic test inputs.

The priority prediction itself is NOT fabricated. It is produced
by the real trained PriorityDetectionPredictor.

Prerequisites:

    Qdrant
        http://localhost:6333

    Neo4j retrieval graph
        configured through existing SupportMind settings

    Neo4j operational/intelligence graph
        configured through existing SupportMind settings

    Ollama
        http://127.0.0.1:11434

    Ollama model
        qwen3:8b

    Phoenix
        http://localhost:6006

    MLflow
        http://127.0.0.1:5000

Run:

    $env:RUN_REAL_MULTI_AGENT_PHOENIX="1"
    $env:PYTHONPATH="$PWD\backend"

    pytest `
        backend\tests\agents\integration\test_real_multi_agent_phoenix.py `
        -v -s
"""

from __future__ import annotations

import asyncio
import os

import pytest

from autogen_ext.models.ollama import OllamaChatCompletionClient
from qdrant_client import QdrantClient


# ============================================================================
# Agent layer
# ============================================================================

from app.agents.classification_agent.agents import (
    ClassificationAgent,
)

from app.agents.classification_agent.contracts.state import (
    PriorityFeatures,
)

from app.agents.classification_agent.tools.classification_tools import (
    ConversationComplexityTool,
    PriorityDetectionTool,
    SentimentTool,
    TicketCategoryTool,
    TopicModelTool,
)

from app.agents.coordinator.agent import (
    CoordinatorAgent,
)

from app.agents.coordinator.router import (
    CoordinatorRouter,
)

from app.agents.escalation_agent.agent import (
    EscalationAgent,
)

from app.agents.orchestrator.orchestrator import (
    AgentOrchestrator,
)

from app.agents.resolution_agent.agents import (
    ResolutionAgent,
)

from app.agents.response_composer_agent.agent import (
    ResponseComposerAgent,
)

from app.agents.retrieval_agent.agents import (
    RetrievalAgent,
)

from app.agents.retrieval_agent.tools.retrieval_tools import (
    RetrievalTools,
)


# ============================================================================
# Coordinator contracts
# ============================================================================

from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
    WorkflowStatus,
)


# ============================================================================
# Real NLP predictors
# ============================================================================

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


# ============================================================================
# Retrieval
# ============================================================================

from app.retrieval.graph_rag.evidence_fusion import (
    EvidenceFusion,
)

from app.retrieval.graph_rag.generator import (
    GraphRAGGenerator,
)

from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)

from app.retrieval.graph_rag.graph_retriever import (
    create_graph_retriever,
)

from app.retrieval.graph_rag.graph_rag_service import (
    GraphRAGService,
)

from app.retrieval.graph_rag.intelligence_adapter import (
    IntelligenceAdapter,
)

from app.retrieval.graph_rag.providers.ollama import (
    OllamaGenerationConfig,
    OllamaProvider,
)

from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)

from app.retrieval.sparse.sparse_retriever import (
    SparseRetriever,
)


# ============================================================================
# Intelligence graph
# ============================================================================

from app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)


# ============================================================================
# Phoenix
# ============================================================================

from app.observability.phoenix import (
    setup_phoenix,
)


# ============================================================================
# MLflow evaluation
# ============================================================================

from backend.mlflow.evaluation.evaluator import (
    evaluate_and_track,
)


# ============================================================================
# Environment
# ============================================================================

RUN_REAL_MULTI_AGENT_PHOENIX = (
    os.getenv(
        "RUN_REAL_MULTI_AGENT_PHOENIX",
        "0",
    ).strip()
    == "1"
)

pytestmark = pytest.mark.skipif(
    not RUN_REAL_MULTI_AGENT_PHOENIX,
    reason=(
        "Set RUN_REAL_MULTI_AGENT_PHOENIX=1 "
        "to run the real multi-agent Phoenix integration test."
    ),
)


QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

OLLAMA_MODEL = os.getenv(
    "SUPPORTMIND_LLM_MODEL",
    "qwen3:8b",
)

COLLECTION_NAME = os.getenv(
    "SUPPORTMIND_QDRANT_COLLECTION",
    "supportmind_retrieval_pdf_test",
)

PHOENIX_URL = os.getenv(
    "PHOENIX_URL",
    "http://localhost:6006",
)


# ============================================================================
# Qdrant adapter
# ============================================================================


class QdrantStoreAdapter:
    """
    Minimal adapter required by SparseRetriever.from_qdrant().
    """

    def __init__(
        self,
        client: QdrantClient,
    ) -> None:
        self.client = client


# ============================================================================
# Predictor construction
# ============================================================================


def build_classification_agent() -> ClassificationAgent:
    """
    Construct the Classification Agent using the actual
    SupportMind NLP predictors.
    """

    category_predictor = TicketCategoryPredictor()
    priority_predictor = PriorityDetectionPredictor()
    sentiment_predictor = SentimentPredictor()
    complexity_predictor = ConversationComplexityPredictor()
    topic_predictor = TopicModelPredictor()

    category_tool = TicketCategoryTool(
        category_predictor
    )

    priority_tool = PriorityDetectionTool(
        priority_predictor
    )

    sentiment_tool = SentimentTool(
        sentiment_predictor
    )

    complexity_tool = ConversationComplexityTool(
        complexity_predictor
    )

    topic_tool = TopicModelTool(
        topic_predictor
    )

    return ClassificationAgent(
        category_tool=category_tool,
        priority_tool=priority_tool,
        sentiment_tool=sentiment_tool,
        complexity_tool=complexity_tool,
        topic_tool=topic_tool,
    )


# ============================================================================
# Retrieval service
# ============================================================================


def build_retrieval_service(
    qdrant_client: QdrantClient,
) -> RetrievalService:
    """
    Build the actual SupportMind RetrievalService.

    Dense retrieval is intentionally disabled here.

    This integration uses the same real sparse + graph retrieval
    path already validated separately.
    """

    sparse_retriever = SparseRetriever.from_qdrant(
        qdrant_store=QdrantStoreAdapter(
            qdrant_client
        ),
        collection_name=COLLECTION_NAME,
        batch_size=100,
    )

    assert sparse_retriever.is_indexed

    graph_retriever = create_graph_retriever()

    assert graph_retriever.verify_connection()

    return RetrievalService(
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        config=RetrievalServiceConfig(
            use_dense=False,
            use_sparse=True,
            use_graph=True,
            use_reranker=False,
            context_max_items=8,
            context_max_characters=12000,
        ),
    )


# ============================================================================
# Graph-RAG service
# ============================================================================


def build_graph_rag_service(
    retrieval_service: RetrievalService,
) -> tuple[
    GraphRAGService,
    object,
    OllamaProvider,
]:
    """
    Build the real Graph-RAG service.
    """

    operational_graph = create_neo4j_operational_graph()

    health = operational_graph.health_check()

    assert health["healthy"], (
        "Help Desk Intelligence graph is not healthy."
    )

    intelligence_adapter = IntelligenceAdapter(
        operational_graph=operational_graph,
    )

    ollama_provider = OllamaProvider(
        config=OllamaGenerationConfig(
            host=OLLAMA_HOST,
            model_name=OLLAMA_MODEL,
            temperature=0.1,
            max_tokens=1024,
            timeout=180.0,
        )
    )

    generator = GraphRAGGenerator(
        provider=ollama_provider,
        provider_name="ollama",
        model_name=OLLAMA_MODEL,
    )

    service = GraphRAGService(
        retrieval_service=retrieval_service,
        intelligence_adapter=intelligence_adapter,
        evidence_fusion=EvidenceFusion(
            default_max_items=20,
            default_max_characters=16000,
        ),
        generator=generator,
        groundedness_validator=GroundednessValidator(),
    )

    return (
        service,
        operational_graph,
        ollama_provider,
    )


# ============================================================================
# AutoGen model client
# ============================================================================


def build_model_client() -> OllamaChatCompletionClient:
    """
    Build the real AutoGen Ollama client.

    The same client is supplied to the Coordinator and
    all LLM-backed specialized agents.
    """

    return OllamaChatCompletionClient(
        model=OLLAMA_MODEL,
        host=OLLAMA_HOST,
    )


async def close_model_client(
    model_client: OllamaChatCompletionClient,
) -> None:
    """
    Close the asynchronous AutoGen model client.
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
# Fabricated test input
# ============================================================================


def build_fabricated_priority_features() -> PriorityFeatures:
    """
    Build synthetic structured features required by the real
    priority detection model.

    These values are test inputs only.

    The priority prediction itself is still generated by the
    real trained PriorityDetectionPredictor.
    """

    return PriorityFeatures(
        issue_proj="billing",
        issue_type="payment",
        created_year=2026,
        created_month=9,
        created_dayofweek=6,
        created_hour=10,
        issue_contr_count=1,
    )


# ============================================================================
# Diagnostic helpers
# ============================================================================


def print_agent_results(
    result: CoordinatorState,
) -> None:
    """
    Print the actual AgentResult records produced by the
    orchestrator.
    """

    print()
    print("Agent results:")

    for agent_name, agent_result in (
        result.agent_results.items()
    ):
        print()
        print(f"[{agent_name}]")

        print(
            f"  success: "
            f"{agent_result.success}"
        )

        print(
            f"  evidence_ids: "
            f"{agent_result.evidence_ids}"
        )

        print(
            f"  error: "
            f"{agent_result.error}"
        )


def assert_agent_succeeded(
    result: CoordinatorState,
    agent_name: str,
) -> None:
    """
    Assert that a named specialized agent executed successfully.
    """

    assert agent_name in result.agent_results, (
        f"Expected agent result {agent_name!r} "
        f"was not recorded. "
        f"Available agents: "
        f"{list(result.agent_results)}"
    )

    agent_result = result.agent_results[agent_name]

    assert agent_result.success, (
        f"{agent_name} failed. "
        f"error={agent_result.error!r}"
    )


# ============================================================================
# Real multi-agent Phoenix + MLflow integration
# ============================================================================


def test_real_multi_agent_phoenix_end_to_end() -> None:
    """
    Execute the complete real SupportMind multi-agent workflow.

    Real components are used for:

        NLP predictors
        Classification tools
        Classification Agent
        Retrieval Agent
        Qdrant
        Neo4j
        Graph-RAG
        Resolution Agent
        Escalation Agent
        Response Composer Agent
        Coordinator
        Router
        Orchestrator
        Ollama / Qwen3
        Phoenix instrumentation
        MLflow evaluation
    """

    # ------------------------------------------------------------------
    # 1. Phoenix instrumentation
    # ------------------------------------------------------------------

    tracer_provider = setup_phoenix()

    assert tracer_provider is not None

    print()
    print("=" * 80)
    print("PHOENIX")
    print("=" * 80)

    print(
        f"Phoenix UI: {PHOENIX_URL}"
    )

    print(
        "Phoenix project: supportmind-agents"
    )

    print(
        "Phoenix is instrumentation-only for this test. "
        "Trace export availability is not used as a workflow assertion."
    )

    # ------------------------------------------------------------------
    # 2. Qdrant
    # ------------------------------------------------------------------

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
    )

    retrieval_service = None
    operational_graph = None
    graph_rag_ollama = None
    model_client = None

    try:
        collections = qdrant_client.get_collections()

        assert collections.collections, (
            "Qdrant is reachable but no collections exist."
        )

        available_collections = {
            item.name
            for item in collections.collections
        }

        assert COLLECTION_NAME in available_collections, (
            f"Required collection {COLLECTION_NAME!r} "
            f"was not found. "
            f"Available collections: "
            f"{sorted(available_collections)}"
        )

        print(
            f"[Multi-Agent] Qdrant collection: "
            f"{COLLECTION_NAME}"
        )

        # --------------------------------------------------------------
        # 3. RetrievalService
        # --------------------------------------------------------------

        retrieval_service = build_retrieval_service(
            qdrant_client
        )

        print(
            "[Multi-Agent] RetrievalService: ready"
        )

        # --------------------------------------------------------------
        # 4. Graph-RAG
        # --------------------------------------------------------------

        (
            graph_rag_service,
            operational_graph,
            graph_rag_ollama,
        ) = build_graph_rag_service(
            retrieval_service
        )

        print(
            "[Multi-Agent] Graph-RAG: ready"
        )

        # --------------------------------------------------------------
        # 5. Ollama / AutoGen
        # --------------------------------------------------------------

        model_client = build_model_client()

        print(
            f"[Multi-Agent] Ollama model: "
            f"{OLLAMA_MODEL}"
        )

        # --------------------------------------------------------------
        # 6. Classification Agent
        # --------------------------------------------------------------

        classification_agent = build_classification_agent()

        print(
            "[Multi-Agent] ClassificationAgent: ready"
        )

        # --------------------------------------------------------------
        # 7. Retrieval Agent
        # --------------------------------------------------------------

        retrieval_tools = RetrievalTools(
            retrieval_service=retrieval_service,
            graph_rag_service=graph_rag_service,
        )

        retrieval_agent = RetrievalAgent(
            model_client=model_client,
            retrieval_tools=retrieval_tools,
        )

        print(
            "[Multi-Agent] RetrievalAgent: ready"
        )

        # --------------------------------------------------------------
        # 8. Resolution Agent
        # --------------------------------------------------------------

        resolution_agent = ResolutionAgent(
            model_client=model_client,
        )

        print(
            "[Multi-Agent] ResolutionAgent: ready"
        )

        # --------------------------------------------------------------
        # 9. Escalation Agent
        # --------------------------------------------------------------

        escalation_agent = EscalationAgent(
            model_client=model_client,
        )

        print(
            "[Multi-Agent] EscalationAgent: ready"
        )

        # --------------------------------------------------------------
        # 10. Response Composer
        # --------------------------------------------------------------

        response_composer_agent = ResponseComposerAgent(
            model_client=model_client,
        )

        print(
            "[Multi-Agent] ResponseComposerAgent: ready"
        )

        # --------------------------------------------------------------
        # 11. Coordinator
        # --------------------------------------------------------------

        coordinator = CoordinatorAgent(
            model_client=model_client,
        )

        print(
            "[Multi-Agent] CoordinatorAgent: ready"
        )

        # --------------------------------------------------------------
        # 12. Router
        # --------------------------------------------------------------

        router = CoordinatorRouter()

        print(
            "[Multi-Agent] CoordinatorRouter: ready"
        )

        # --------------------------------------------------------------
        # 13. Orchestrator
        # --------------------------------------------------------------

        orchestrator = AgentOrchestrator(
            coordinator=coordinator,
            router=router,
            classification_agent=classification_agent,
            retrieval_agent=retrieval_agent,
            resolution_agent=resolution_agent,
            escalation_agent=escalation_agent,
            response_composer_agent=response_composer_agent,
            max_iterations=12,
            retrieval_collection_name=COLLECTION_NAME,
        )

        print(
            "[Multi-Agent] Orchestrator: ready"
        )

        # --------------------------------------------------------------
        # 14. Real customer-support request
        # --------------------------------------------------------------

        request = (
            "A customer is asking about a refund for a subscription "
            "and is frustrated because the issue has not been resolved. "
            "Use the available support data to understand the issue, "
            "retrieve relevant evidence, determine an appropriate "
            "resolution, assess whether escalation is warranted, "
            "and provide a grounded final response."
        )

        # --------------------------------------------------------------
        # 15. Synthetic structured priority input
        # --------------------------------------------------------------

        priority_features = (
            build_fabricated_priority_features()
        )

        print()
        print("=" * 80)
        print("SYNTHETIC PRIORITY FEATURES")
        print("=" * 80)

        print(
            priority_features.model_dump()
        )

        # --------------------------------------------------------------
        # 16. Initial Coordinator state
        # --------------------------------------------------------------

        state = CoordinatorState(
            request=request,
            requested_capabilities=[
                Capability.CLASSIFICATION,
                Capability.RETRIEVAL,
                Capability.RESOLUTION,
                Capability.ESCALATION,
                Capability.GENERATION,
            ],
            priority_features=priority_features,
        )

        # --------------------------------------------------------------
        # 17. Execute the REAL multi-agent workflow
        # --------------------------------------------------------------

        print()
        print("=" * 80)
        print("STARTING REAL MULTI-AGENT WORKFLOW")
        print("=" * 80)

        result = asyncio.run(
            orchestrator.run(state)
        )

        # --------------------------------------------------------------
        # 18. Workflow diagnostics
        # --------------------------------------------------------------

        print()
        print("=" * 80)
        print("REAL MULTI-AGENT RESULT")
        print("=" * 80)

        print(
            f"Status: {result.status.value}"
        )

        print(
            f"Iterations: "
            f"{result.iteration_count}"
        )

        print(
            "\nCompleted capabilities:"
        )

        for capability in result.completed_capabilities:
            print(
                f"  - {capability.value}"
            )

        print_agent_results(result)

        print()
        print("Evidence count:")
        print(
            len(result.evidence)
        )

        print()
        print("Errors:")

        if result.errors:
            for error in result.errors:
                print(
                    f"  {error}"
                )
        else:
            print(
                "  None"
            )

        print()
        print("FINAL RESPONSE:")
        print(
            result.final_response
        )

        # --------------------------------------------------------------
        # 19. Core workflow assertion
        # --------------------------------------------------------------

        assert (
            result.status
            == WorkflowStatus.COMPLETED
        ), (
            "Multi-agent workflow did not complete. "
            f"status={result.status}, "
            f"errors={result.errors}"
        )

        assert (
            result.final_response
            and result.final_response.strip()
        ), (
            "Multi-agent workflow produced no final response."
        )

        assert result.iteration_count > 0

        # --------------------------------------------------------------
        # 20. All requested capabilities completed
        # --------------------------------------------------------------

        expected_capabilities = {
            Capability.CLASSIFICATION,
            Capability.RETRIEVAL,
            Capability.RESOLUTION,
            Capability.ESCALATION,
            Capability.GENERATION,
        }

        completed_capabilities = set(
            result.completed_capabilities
        )

        missing_capabilities = (
            expected_capabilities
            - completed_capabilities
        )

        assert not missing_capabilities, (
            "The orchestrator did not complete all "
            "requested capabilities. "
            f"Missing: {missing_capabilities}. "
            f"Completed: {completed_capabilities}. "
            f"Errors: {result.errors}"
        )

        # --------------------------------------------------------------
        # 21. Classification Agent
        # --------------------------------------------------------------

        assert_agent_succeeded(
            result,
            "classification_agent",
        )

        assert result.predictions is not None

        print()
        print(
            "Classification predictions:"
        )

        print(
            result.predictions.model_dump()
        )

        # The real priority predictor must have produced
        # a prediction from the supplied structured features.
        assert result.predictions.priority is not None

        assert (
            result.predictions.priority.value
            and result.predictions.priority.value.strip()
        )

        # --------------------------------------------------------------
        # 22. Retrieval Agent
        # --------------------------------------------------------------

        assert_agent_succeeded(
            result,
            "retrieval_agent",
        )

        retrieval_result = (
            result.agent_results[
                "retrieval_agent"
            ]
        )

        assert retrieval_result.evidence_ids, (
            "Retrieval Agent succeeded but produced "
            "no evidence IDs."
        )

        # --------------------------------------------------------------
        # 23. Resolution Agent
        # --------------------------------------------------------------

        assert_agent_succeeded(
            result,
            "resolution_agent",
        )

        resolution_result = (
            result.agent_results[
                "resolution_agent"
            ]
        )

        assert resolution_result.result, (
            "Resolution Agent produced no structured result."
        )

        resolution_payload = (
            resolution_result.result
        )

        # The Resolution Agent now has a strict structured
        # ResolutionRecommendation contract.
        required_resolution_fields = {
            "resolution_type",
            "recommendation",
            "rationale",
            "evidence_ids",
            "confidence",
            "requires_human_review",
            "proposed_actions",
        }

        missing_resolution_fields = (
            required_resolution_fields
            - set(resolution_payload)
        )

        assert not missing_resolution_fields, (
            "Resolution Agent result is missing required "
            "structured fields. "
            f"Missing: {missing_resolution_fields}. "
            f"Result: {resolution_payload}"
        )

        assert (
            resolution_payload["recommendation"]
            and resolution_payload["recommendation"].strip()
        )

        assert (
            resolution_payload["rationale"]
            and resolution_payload["rationale"].strip()
        )

        # Every evidence ID cited by the resolution agent must
        # originate from the evidence available to the workflow.
        workflow_evidence_ids = {
            evidence.evidence_id
            for evidence in result.evidence
        }

        resolution_evidence_ids = set(
            resolution_payload["evidence_ids"]
        )

        assert resolution_evidence_ids <= (
            workflow_evidence_ids
        ), (
            "Resolution Agent referenced evidence IDs that "
            "are not present in the workflow evidence. "
            f"Unknown IDs: "
            f"{resolution_evidence_ids - workflow_evidence_ids}"
        )

        # --------------------------------------------------------------
        # 24. Escalation Agent
        # --------------------------------------------------------------

        assert_agent_succeeded(
            result,
            "escalation_agent",
        )

        escalation_result = (
            result.agent_results[
                "escalation_agent"
            ]
        )

        assert escalation_result.result, (
            "Escalation Agent produced no result."
        )

        # --------------------------------------------------------------
        # 25. Response Composer Agent
        # --------------------------------------------------------------

        assert_agent_succeeded(
            result,
            "response_composer_agent",
        )

        response_composer_result = (
            result.agent_results[
                "response_composer_agent"
            ]
        )

        assert response_composer_result.result, (
            "Response Composer Agent produced no result."
        )

        # --------------------------------------------------------------
        # 26. Evidence survived the entire workflow
        # --------------------------------------------------------------

        assert result.evidence, (
            "No evidence survived the complete "
            "multi-agent workflow."
        )

        # --------------------------------------------------------------
        # 27. MLflow evaluation
        # --------------------------------------------------------------

        print()
        print("=" * 80)
        print("MLFLOW EVALUATION")
        print("=" * 80)

        mlflow_result = evaluate_and_track(
            result,
            run_name="real_agentic_workflow_evaluation",
        )

        print(
            f"MLflow experiment ID: "
            f"{mlflow_result['experiment_id']}"
        )

        print(
            f"MLflow run ID: "
            f"{mlflow_result['run_id']}"
        )

        print()
        print("MLflow metrics:")

        for metric_name, metric_value in (
            mlflow_result["metrics"].items()
        ):
            print(
                f"  {metric_name}: "
                f"{metric_value}"
            )

        assert mlflow_result["run_id"]
        assert mlflow_result["experiment_id"]

        assert (
            mlflow_result["metrics"]["workflow_success"]
            == 1.0
        )

        assert (
            mlflow_result["metrics"]["classification_success"]
            == 1.0
        )

        assert (
            mlflow_result["metrics"]["retrieval_success"]
            == 1.0
        )

        assert (
            mlflow_result["metrics"]["resolution_success"]
            == 1.0
        )

        assert (
            mlflow_result["metrics"]["escalation_success"]
            == 1.0
        )

        assert (
            mlflow_result["metrics"]["response_generation_success"]
            == 1.0
        )

        assert (
            mlflow_result["metrics"]["evidence_validity"]
            == 1.0
        )

        # --------------------------------------------------------------
        # 28. Phoenix trace information
        # --------------------------------------------------------------

        print()
        print("=" * 80)
        print("PHOENIX TRACE")
        print("=" * 80)

        print(
            "Expected parent span:"
        )

        print(
            "  supportmind.agent-orchestrator.run"
        )

        print(
            "\nExpected agent child spans:"
        )

        print(
            "  agent.classification"
        )

        print(
            "  agent.retrieval"
        )

        print(
            "  agent.resolution"
        )

        print(
            "  agent.escalation"
        )

        print(
            "  agent.generation"
        )

        print(
            f"\nPhoenix UI: {PHOENIX_URL}"
        )

        print()
        print(
            "REAL MULTI-AGENT WORKFLOW PASSED."
        )

    finally:
        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

        if model_client is not None:
            asyncio.run(
                close_model_client(
                    model_client
                )
            )

        if retrieval_service is not None:
            try:
                retrieval_service.close()
            except Exception:
                pass

        if operational_graph is not None:
            try:
                operational_graph.close()
            except Exception:
                pass

        if graph_rag_ollama is not None:
            try:
                graph_rag_ollama.close()
            except Exception:
                pass

        try:
            qdrant_client.close()
        except Exception:
            pass