#services/support_assist.py
from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence

from autogen_ext.models.ollama import OllamaChatCompletionClient
from qdrant_client import QdrantClient

from app.agents.classification_agent.agents import (
    ClassificationAgent,
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
from app.agents.coordinator.state import (
    Capability,
    CoordinatorState,
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

from app.observability.phoenix import (
    setup_phoenix,
)

from app.retrieval.graph_rag.evidence_fusion import (
    EvidenceFusion,
)
from app.retrieval.graph_rag.generator import (
    GraphRAGGenerator,
)
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)
from app.retrieval.graph_rag.graph_rag_service import (
    GraphRAGService,
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
from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)
from app.retrieval.sparse.sparse_retriever import (
    SparseRetriever,
)

from app.support.schemas.assist import (
    SupportAssistIntelligence,
    SupportAssistResponse,
    SupportAssistResponseContent,
    SupportAssistRetrievalSummary,
    SupportAssistWorkflowSummary,
)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_COLLECTION = "supportmind_retrieval_pdf_test"


class QdrantStoreAdapter:
    """
    Minimal adapter required by SparseRetriever.from_qdrant().
    """

    def __init__(
        self,
        client: QdrantClient,
    ) -> None:
        self.client = client


class SupportAssistService:
    """
    Application-level composition service for the complete
    SupportMind multi-agent workflow.

    This class owns dependency construction and lifecycle.

    It does not implement classification, retrieval, Graph-RAG,
    resolution, escalation, or response-generation logic itself.
    Those responsibilities remain inside the existing components.
    """

    def __init__(
        self,
        *,
        qdrant_url: str | None = None,
        ollama_host: str | None = None,
        ollama_model: str | None = None,
        collection_name: str | None = None,
        max_iterations: int = 12,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(
                "max_iterations must be at least 1."
            )

        self.qdrant_url = (
            qdrant_url
            or os.getenv(
                "QDRANT_URL",
                DEFAULT_QDRANT_URL,
            )
        )

        self.ollama_host = (
            ollama_host
            or os.getenv(
                "OLLAMA_HOST",
                DEFAULT_OLLAMA_HOST,
            )
        )

        self.ollama_model = (
            ollama_model
            or os.getenv(
                "SUPPORTMIND_LLM_MODEL",
                DEFAULT_MODEL,
            )
        )

        self.collection_name = (
            collection_name
            or os.getenv(
                "SUPPORTMIND_QDRANT_COLLECTION",
                DEFAULT_COLLECTION,
            )
        )

        self.max_iterations = max_iterations

        self._phoenix_initialized = False

        self._qdrant_client: QdrantClient | None = None
        self._retrieval_service: RetrievalService | None = None
        self._operational_graph = None
        self._graph_rag_ollama: OllamaProvider | None = None
        self._model_client: OllamaChatCompletionClient | None = None
        self._orchestrator: AgentOrchestrator | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Build the complete real SupportMind workflow.

        Initialization is explicit so FastAPI startup can construct
        the workflow once instead of rebuilding it for every request.
        """

        if self._orchestrator is not None:
            return

        try:
            self._initialize_phoenix()

            self._qdrant_client = QdrantClient(
                url=self.qdrant_url,
            )

            self._validate_qdrant()

            self._retrieval_service = (
                self._build_retrieval_service(
                    self._qdrant_client
                )
            )

            (
                graph_rag_service,
                self._operational_graph,
                self._graph_rag_ollama,
            ) = self._build_graph_rag_service(
                self._retrieval_service
            )

            self._model_client = self._build_model_client()

            classification_agent = (
                self._build_classification_agent()
            )

            retrieval_tools = RetrievalTools(
                retrieval_service=self._retrieval_service,
                graph_rag_service=graph_rag_service,
            )

            retrieval_agent = RetrievalAgent(
                model_client=self._model_client,
                retrieval_tools=retrieval_tools,
            )

            resolution_agent = ResolutionAgent(
                model_client=self._model_client,
            )

            escalation_agent = EscalationAgent(
                model_client=self._model_client,
            )

            response_composer_agent = (
                ResponseComposerAgent(
                    model_client=self._model_client,
                )
            )

            coordinator = CoordinatorAgent(
                model_client=self._model_client,
            )

            router = CoordinatorRouter()

            self._orchestrator = AgentOrchestrator(
                coordinator=coordinator,
                router=router,
                classification_agent=classification_agent,
                retrieval_agent=retrieval_agent,
                resolution_agent=resolution_agent,
                escalation_agent=escalation_agent,
                response_composer_agent=response_composer_agent,
                max_iterations=self.max_iterations,
                retrieval_collection_name=self.collection_name,
            )

        except Exception:
            self.close()
            raise

    async def assist(
        self,
        *,
        request: str,
    ) -> SupportAssistResponse:
        """
        Execute the complete SupportMind multi-agent workflow.

        The public API provides only a natural-language request.
        Workflow capabilities and model-specific structured features
        are managed internally.
        """

        if self._orchestrator is None:
            self.initialize()

        assert self._orchestrator is not None

        state = CoordinatorState(
            request=request,
            requested_capabilities=[
                Capability.CLASSIFICATION,
                Capability.RETRIEVAL,
                Capability.RESOLUTION,
                Capability.ESCALATION,
                Capability.GENERATION,
            ],
            priority_features=None,
        )

        result = await self._orchestrator.run(state)

        retrieval_result = result.agent_results.get(
            "retrieval_agent"
        )
        retrieval_data: dict = {}
        if retrieval_result is not None and retrieval_result.success:
            if isinstance(retrieval_result.result, dict):
                retrieval_data = retrieval_result.result

        execution_metadata = retrieval_data.get(
            "execution_metadata",
            {}
        )
        retrieval_summary = SupportAssistRetrievalSummary(
            strategy=retrieval_data.get("strategy"),
            evidence_count=retrieval_data.get(
                "evidence_count",
                len(result.evidence),
            ),
            grounding_status=execution_metadata.get(
                "grounding_status"
            ),
            grounding_valid=execution_metadata.get(
                "grounding_valid"
            ),
            provider=execution_metadata.get(
                "provider"
            ),
            model_name=execution_metadata.get(
                "model_name"
            ),
            latency_ms=execution_metadata.get(
                "latency_ms"
            ),
        )
        workflow_summary = SupportAssistWorkflowSummary(
            status=result.status,
            current_step=result.current_step,
            completed_capabilities=result.completed_capabilities,
            iteration_count=result.iteration_count,
            errors=result.errors,
        )
        return SupportAssistResponse(
            request=request,

            response=SupportAssistResponseContent(
                text=result.final_response,
            ),

            intelligence=SupportAssistIntelligence(
                predictions=result.predictions,
                retrieval=retrieval_summary,
                evidence=result.evidence,
                agents=result.agent_results,
                workflow=workflow_summary,
            ),
        )

    def close(self) -> None:
        """
        Release resources owned by the application workflow.
        """

        if self._model_client is not None:
            try:
                asyncio.run(
                    self._close_model_client(
                        self._model_client
                    )
                )
            except RuntimeError:
                # The application may already be inside an event loop.
                pass
            except Exception:
                pass

        if self._retrieval_service is not None:
            try:
                self._retrieval_service.close()
            except Exception:
                pass

        if self._operational_graph is not None:
            try:
                self._operational_graph.close()
            except Exception:
                pass

        if self._graph_rag_ollama is not None:
            try:
                self._graph_rag_ollama.close()
            except Exception:
                pass

        if self._qdrant_client is not None:
            try:
                self._qdrant_client.close()
            except Exception:
                pass

        self._model_client = None
        self._retrieval_service = None
        self._operational_graph = None
        self._graph_rag_ollama = None
        self._qdrant_client = None
        self._orchestrator = None

    # ------------------------------------------------------------------
    # Phoenix
    # ------------------------------------------------------------------

    def _initialize_phoenix(self) -> None:
        if self._phoenix_initialized:
            return

        tracer_provider = setup_phoenix()

        if tracer_provider is None:
            raise RuntimeError(
                "SupportMind Phoenix initialization failed."
            )

        self._phoenix_initialized = True

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------

    def _validate_qdrant(self) -> None:
        assert self._qdrant_client is not None

        collections = (
            self._qdrant_client.get_collections()
        )

        available_collections = {
            item.name
            for item in collections.collections
        }

        if self.collection_name not in available_collections:
            raise RuntimeError(
                "Required SupportMind Qdrant collection "
                f"{self.collection_name!r} was not found. "
                f"Available collections: "
                f"{sorted(available_collections)}"
            )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def _build_classification_agent(
        self,
    ) -> ClassificationAgent:
        category_predictor = (
            TicketCategoryPredictor()
        )

        priority_predictor = (
            PriorityDetectionPredictor()
        )

        sentiment_predictor = (
            SentimentPredictor()
        )

        complexity_predictor = (
            ConversationComplexityPredictor()
        )

        topic_predictor = (
            TopicModelPredictor()
        )

        return ClassificationAgent(
            category_tool=TicketCategoryTool(
                category_predictor
            ),
            priority_tool=PriorityDetectionTool(
                priority_predictor
            ),
            sentiment_tool=SentimentTool(
                sentiment_predictor
            ),
            complexity_tool=ConversationComplexityTool(
                complexity_predictor
            ),
            topic_tool=TopicModelTool(
                topic_predictor
            ),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _build_retrieval_service(
        self,
        qdrant_client: QdrantClient,
    ) -> RetrievalService:
        sparse_retriever = (
            SparseRetriever.from_qdrant(
                qdrant_store=QdrantStoreAdapter(
                    qdrant_client
                ),
                collection_name=self.collection_name,
                batch_size=100,
            )
        )

        if not sparse_retriever.is_indexed:
            raise RuntimeError(
                "SupportMind sparse retriever is not indexed."
            )

        graph_retriever = create_graph_retriever()

        if not graph_retriever.verify_connection():
            raise RuntimeError(
                "SupportMind retrieval graph connection "
                "could not be verified."
            )

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

    # ------------------------------------------------------------------
    # Graph-RAG
    # ------------------------------------------------------------------

    def _build_graph_rag_service(
        self,
        retrieval_service: RetrievalService,
    ) -> tuple[
        GraphRAGService,
        object,
        OllamaProvider,
    ]:
        operational_graph = (
            create_neo4j_operational_graph()
        )

        health = (
            operational_graph.health_check()
        )

        if not health["healthy"]:
            raise RuntimeError(
                "SupportMind operational intelligence "
                "graph is not healthy."
            )

        intelligence_adapter = IntelligenceAdapter(
            operational_graph=operational_graph,
        )

        ollama_provider = OllamaProvider(
            config=OllamaGenerationConfig(
                host=self.ollama_host,
                model_name=self.ollama_model,
                temperature=0.1,
                max_tokens=1024,
                timeout=180.0,
            )
        )

        generator = GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=self.ollama_model,
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

    # ------------------------------------------------------------------
    # AutoGen / Ollama
    # ------------------------------------------------------------------

    def _build_model_client(
        self,
    ) -> OllamaChatCompletionClient:
        return OllamaChatCompletionClient(
            model=self.ollama_model,
            host=self.ollama_host,
        )

    async def _close_model_client(
        self,
        model_client: OllamaChatCompletionClient,
    ) -> None:
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