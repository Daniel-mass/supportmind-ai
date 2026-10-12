from __future__ import annotations

import asyncio
import os

import pytest
from qdrant_client import QdrantClient

from app.evaluation.contracts.evaluation import (
    EvaluationCase,
)
from app.evaluation.rag.ragas_evaluator import (
    create_ragas_evaluator,
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
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGRequest,
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
from app.retrieval.graph_rag.graph_retriever import (
    create_graph_retriever,
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

from app.vector_store.qdrant_store import (
    QdrantStore,
)

from app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)


# ============================================================================
# Environment
# ============================================================================

RUN_REAL_RAGAS = os.getenv(
    "RUN_REAL_RAGAS",
    "0",
)

if RUN_REAL_RAGAS != "1":
    pytest.skip(
        "Real RAGAS integration test is disabled. "
        "Set RUN_REAL_RAGAS=1 to run.",
        allow_module_level=True,
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
    "SUPPORTMIND_GRAPH_RAG_MODEL",
    "qwen3:8b",
)

SUPPORTMIND_COLLECTION = os.getenv(
    "SUPPORTMIND_QDRANT_COLLECTION",
)


# ============================================================================
# Helpers
# ============================================================================


def resolve_supportmind_collection(
    client: QdrantClient,
) -> str:
    """Resolve the real SupportMind Qdrant collection."""

    if SUPPORTMIND_COLLECTION:
        collections = {
            item.name
            for item in client.get_collections().collections
        }

        if SUPPORTMIND_COLLECTION not in collections:
            raise RuntimeError(
                "Configured SupportMind collection does not exist: "
                f"{SUPPORTMIND_COLLECTION!r}"
            )

        return SUPPORTMIND_COLLECTION

    collections = [
        item.name
        for item in client.get_collections().collections
    ]

    if not collections:
        raise RuntimeError(
            "Qdrant is reachable but contains no collections."
        )

    if len(collections) == 1:
        return collections[0]

    supportmind_candidates = [
        name
        for name in collections
        if "supportmind" in name.lower()
    ]

    if len(supportmind_candidates) == 1:
        return supportmind_candidates[0]

    raise RuntimeError(
        "Unable to resolve SupportMind Qdrant collection. "
        "Set SUPPORTMIND_QDRANT_COLLECTION explicitly. "
        f"Available collections: {sorted(collections)}"
    )


# ============================================================================
# Real Graph-RAG → RAGAS integration
# ============================================================================


def test_real_graph_rag_with_ragas() -> None:
    """
    Validate the complete real SupportMind Graph-RAG + RAGAS pipeline.

    Architecture:

        Qdrant
          ↓
        SparseRetriever
          +
        Retrieval Graph
          ↓
        RetrievalService
          ↓
        Intelligence Graph
          ↓
        IntelligenceAdapter
          ↓
        EvidenceFusion
          ↓
        GraphRAGGenerator
          ↓
        Real Ollama
          ↓
        GraphRAGResponse
          ↓
        EvaluationCase
          ↓
        RAGAS
          ↓
        Faithfulness
        Context Precision
    """

    qdrant_client: QdrantClient | None = None
    retrieval_service: RetrievalService | None = None
    operational_graph = None
    ollama_provider: OllamaProvider | None = None

    try:
        # ------------------------------------------------------------------
        # 1. Connect to real Qdrant
        # ------------------------------------------------------------------

        qdrant_client = QdrantClient(
            url=QDRANT_URL,
        )

        collections = qdrant_client.get_collections()

        assert collections.collections, (
            "Qdrant is reachable but no collections exist."
        )

        collection_name = (
            resolve_supportmind_collection(
                qdrant_client
            )
        )

        print()
        print("=" * 90)
        print("REAL GRAPH-RAG → RAGAS INTEGRATION")
        print("=" * 90)

        print()
        print(
            f"Qdrant collection: {collection_name}"
        )

        # ------------------------------------------------------------------
        # 2. Real BM25 / Sparse Retriever
        # ------------------------------------------------------------------

        sparse_retriever = (
            SparseRetriever.from_qdrant(
                qdrant_store=QdrantStore(
                    client=qdrant_client
                ),
                collection_name=collection_name,
                batch_size=100,
            )
        )

        assert sparse_retriever.is_indexed

        print(
            "BM25 documents: "
            f"{sparse_retriever.document_count}"
        )

        # ------------------------------------------------------------------
        # 3. Existing SupportMind retrieval graph
        # ------------------------------------------------------------------

        retrieval_graph = create_graph_retriever()

        assert retrieval_graph.verify_connection(), (
            "SupportMind retrieval graph could not connect."
        )

        print(
            "[OK] Retrieval graph connected."
        )

        # ------------------------------------------------------------------
        # 4. Real RetrievalService
        # ------------------------------------------------------------------

        retrieval_service = RetrievalService(
            sparse_retriever=sparse_retriever,
            graph_retriever=retrieval_graph,
            config=RetrievalServiceConfig(
                use_dense=False,
                use_sparse=True,
                use_graph=True,
                use_reranker=False,
                context_max_items=8,
                context_max_characters=12000,
            ),
        )

        print(
            "[OK] RetrievalService created."
        )

        # ------------------------------------------------------------------
        # 5. Real Intelligence Graph
        # ------------------------------------------------------------------

        operational_graph = (
            create_neo4j_operational_graph()
        )

        health = operational_graph.health_check()

        assert health["healthy"], (
            "Help Desk Intelligence graph is not healthy."
        )

        print(
            "[OK] Intelligence graph healthy."
        )

        intelligence_adapter = (
            IntelligenceAdapter(
                operational_graph
            )
        )

        # ------------------------------------------------------------------
        # 6. Real Ollama Graph-RAG generator
        # ------------------------------------------------------------------

        ollama_provider = OllamaProvider(
            config=OllamaGenerationConfig(
                host=OLLAMA_HOST,
                model_name=OLLAMA_MODEL,
                temperature=0.1,
                max_tokens=700,
                timeout=180.0,
            )
        )

        generator = GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=OLLAMA_MODEL,
        )

        print(
            "[OK] Graph-RAG generator created."
        )

        # ------------------------------------------------------------------
        # 7. Complete real GraphRAGService
        # ------------------------------------------------------------------

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

        print(
            "[OK] GraphRAGService created."
        )

        # ------------------------------------------------------------------
        # 8. Real SupportMind question
        # ------------------------------------------------------------------

        query = (
            "What does the support data show about "
            "ticket priorities, statuses, workflow activity, "
            "and common operational delays?"
        )

        retrieval_request = RetrievalRequest(
            query=query,
            top_k=5,
            collection_name=collection_name,
            include_graph=True,
            rerank=False,
        )

        graph_rag_request = GraphRAGRequest(
            query=query,
            retrieval_request=retrieval_request,
            include_intelligence=True,
            include_workflow=True,
            max_evidence_items=20,
            max_evidence_characters=16000,
        )

        # ------------------------------------------------------------------
        # 9. REAL Graph-RAG execution
        # ------------------------------------------------------------------

        print()
        print(
            "[RUNNING] Real Graph-RAG..."
        )

        response = service.answer(
            graph_rag_request
        )

        assert response.answer.strip()

        assert response.evidence, (
            "Graph-RAG returned no evidence."
        )

        print()
        print("-" * 90)
        print("REAL GRAPH-RAG ANSWER")
        print("-" * 90)

        print(
            response.answer
        )

        print()
        print(
            "Evidence items: "
            f"{len(response.evidence)}"
        )

        print(
            "Grounding: "
            f"{response.grounding.status}"
        )

        # ------------------------------------------------------------------
        # 10. Convert REAL Graph-RAG output into EvaluationCase
        # ------------------------------------------------------------------
        #
        # This is the critical integration point.
        #
        # RAGAS does NOT retrieve anything itself here.
        #
        # We evaluate the exact evidence that SupportMind
        # actually supplied to GraphRAGGenerator.
        # ------------------------------------------------------------------

        evaluation_case = EvaluationCase(
            case_id="real_graph_rag_ragas_001",
            query=response.query,
            retrieved_context=[
                evidence.text
                for evidence in response.evidence
            ],
            generated_answer=response.answer,
            reference_answer=None,
            metadata={
                "pipeline": "supportmind_graph_rag",
                "provider": response.provider,
                "model_name": response.model_name,
                "evidence_count": len(
                    response.evidence
                ),
                "grounding_status": (
                    response.grounding.status.value
                ),
            },
        )

        print()
        print(
            "EvaluationCase created."
        )

        print(
            "RAGAS context items: "
            f"{len(evaluation_case.retrieved_context)}"
        )

        # ------------------------------------------------------------------
        # 11. Create REAL RAGAS evaluator
        # ------------------------------------------------------------------

        evaluator = create_ragas_evaluator(
            model_name=OLLAMA_MODEL,
            base_url=OLLAMA_HOST + "/v1",
        )

        print(
            "[OK] RAGAS evaluator created."
        )

        # ------------------------------------------------------------------
        # 12. REAL RAGAS evaluation
        # ------------------------------------------------------------------

        print()
        print(
            "[RUNNING] Real RAGAS evaluation..."
        )

        ragas_result = asyncio.run(
            evaluator.evaluate_case(
                evaluation_case
            )
        )

        # ------------------------------------------------------------------
        # 13. Print actual RAGAS metrics
        # ------------------------------------------------------------------

        print()
        print("=" * 90)
        print("REAL RAGAS RESULTS")
        print("=" * 90)

        print()
        print(
            f"Case ID       : {ragas_result.case_id}"
        )

        print(
            f"Metric count  : {ragas_result.metric_count}"
        )

        print(
            f"Overall passed: {ragas_result.overall_passed}"
        )

        for metric in ragas_result.metrics:

            print()
            print(
                f"Metric: {metric.metric_name}"
            )

            print(
                f"Score : {metric.score}"
            )

            print(
                f"Reason: {metric.reason}"
            )

        print()
        print("=" * 90)

        # ------------------------------------------------------------------
        # 14. Assertions
        # ------------------------------------------------------------------

        assert ragas_result.metric_count >= 2, (
            "Expected at least Faithfulness and "
            "Context Precision."
        )

        metric_names = {
            metric.metric_name
            for metric in ragas_result.metrics
        }

        assert "faithfulness" in metric_names
        assert "context_precision" in metric_names

        for metric in ragas_result.metrics:

            assert metric.score is not None

            assert 0.0 <= metric.score <= 1.0

        print()
        print(
            "[PASS] Real Graph-RAG → RAGAS integration."
        )

    finally:
        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------

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

        if ollama_provider is not None:

            try:
                ollama_provider.close()
            except Exception:
                pass

        if qdrant_client is not None:

            try:
                qdrant_client.close()
            except Exception:
                pass