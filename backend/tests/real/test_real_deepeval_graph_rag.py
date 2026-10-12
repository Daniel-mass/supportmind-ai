"""
Real end-to-end Graph-RAG integration test with DeepEval.

This test exercises:

    Qdrant
        ↓
    BM25 / SparseRetriever
        +
    Existing SupportMind GraphRetriever / Neo4j
        ↓
    RetrievalService
        +
    Help Desk Intelligence Neo4j graph
        ↓
    IntelligenceAdapter
        ↓
    EvidenceFusion
        ↓
    GraphRAGGenerator
        ↓
    Real Ollama
        ↓
    GroundednessValidator
        ↓
    DeepEval
        ↓
    Faithfulness
    Answer Relevancy

Run explicitly with:

    $env:RUN_REAL_GRAPH_RAG_DEEPEVAL="1"
    $env:PYTHONPATH="$PWD\backend"

    python -m pytest `
        backend\tests\real\test_real_deepeval_graph_rag.py `
        -v -s --tb=long
"""

from __future__ import annotations

import os

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from qdrant_client import QdrantClient

from app.evaluation.deepeval.result_store import (
    LATEST_RESULT_PATH,
    save_deepeval_result,
)
from app.evaluation.models.qwen3_deepeval import (
    Qwen3DeepEvalModel,
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
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidenceSource,
    GraphRAGRequest,
    GroundingStatus,
)
from app.retrieval.graph_rag.graph_rag_service import (
    GraphRAGService,
)
from app.retrieval.graph_rag.graph_retriever import (
    GraphRetriever,
    Neo4jConfig,
)
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
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

RUN_REAL_GRAPH_RAG_DEEPEVAL = os.getenv(
    "RUN_REAL_GRAPH_RAG_DEEPEVAL",
    "0",
)

if RUN_REAL_GRAPH_RAG_DEEPEVAL != "1":
    pytest.skip(
        "Real Graph-RAG DeepEval tests are disabled. "
        "Set RUN_REAL_GRAPH_RAG_DEEPEVAL=1 to run.",
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
# Test-only Graph-RAG Neo4j normalization
# ============================================================================
#
# IMPORTANT:
#
# The production GraphRetriever is shared by the normal Retrieval/RAG
# pipeline, so we intentionally DO NOT modify graph_retriever.py.
#
# The operational SupportMind Neo4j graph uses:
#
#     status
#     priority
#     channel
#
# while the retrieval model previously listed:
#
#     ticket_status
#     ticket_priority
#     ticket_channel
#
# For this real DeepEval integration test, we explicitly configure the
# GraphRetriever with the actual Neo4j property names.
#
# This normalization affects ONLY the GraphRetriever instance created by
# this test.
#
# It does NOT modify:
#
#     - production GraphRetriever defaults
#     - RetrievalService
#     - normal Graph-RAG APIs
#     - Neo4j data
#     - other tests
#
# ============================================================================

DEEPEVAL_GRAPH_SEARCH_PROPERTIES: tuple[str, ...] = (
    "name",
    "email",
    "subject",
    "description",
    "ticket_id",
    "status",
    "ticket_type",
    "priority",
    "channel",
    "product_purchased",
    "source_file",
    "document_id",
    "node_id",
)


# ============================================================================
# Helpers
# ============================================================================


def resolve_supportmind_collection(
    client: QdrantClient,
) -> str:
    """
    Resolve the SupportMind Qdrant collection.

    Preferred order:

        1. SUPPORTMIND_QDRANT_COLLECTION
        2. exactly one collection
        3. exactly one collection containing "supportmind"

    Multiple ambiguous collections require explicit configuration.
    """

    if SUPPORTMIND_COLLECTION:
        collections = {
            item.name
            for item in client.get_collections().collections
        }

        if SUPPORTMIND_COLLECTION not in collections:
            raise RuntimeError(
                "Configured SUPPORTMIND_QDRANT_COLLECTION does not "
                f"exist: {SUPPORTMIND_COLLECTION!r}. "
                f"Available collections: {sorted(collections)}"
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
        "Unable to resolve the SupportMind Qdrant collection "
        "automatically. Set SUPPORTMIND_QDRANT_COLLECTION. "
        f"Available collections: {sorted(collections)}"
    )


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
# Real integration test
# ============================================================================


def test_real_graph_rag_with_deepeval() -> None:
    """
    Run the real SupportMind Graph-RAG pipeline and evaluate
    its actual generated answer using DeepEval.

    DeepEval metrics:

        - Faithfulness
        - Answer Relevancy

    Contextual Relevancy is intentionally excluded because the
    local Qwen3 evaluation takes excessively long for this test.
    """

    qdrant_client: QdrantClient | None = None
    retrieval_service: RetrievalService | None = None
    retrieval_graph: GraphRetriever | None = None
    operational_graph = None
    ollama_provider: OllamaProvider | None = None

    try:

        # ------------------------------------------------------------------
        # 1. Qdrant
        # ------------------------------------------------------------------

        qdrant_client = QdrantClient(
            url=QDRANT_URL,
        )

        collections = qdrant_client.get_collections()

        assert collections.collections, (
            "Qdrant is reachable but no collections were found."
        )

        collection_name = resolve_supportmind_collection(
            qdrant_client
        )

        print(
            f"\n[DeepEval Graph-RAG] "
            f"Qdrant collection: {collection_name}"
        )

        # ------------------------------------------------------------------
        # 2. Sparse retriever
        # ------------------------------------------------------------------

        sparse_retriever = SparseRetriever.from_qdrant(
            qdrant_store=QdrantStoreAdapter(
                qdrant_client
            ),
            collection_name=collection_name,
            batch_size=100,
        )

        assert sparse_retriever.is_indexed

        print(
            "[DeepEval Graph-RAG] BM25 documents: "
            f"{sparse_retriever.document_count}"
        )

        # ------------------------------------------------------------------
        # 3. Existing SupportMind retrieval graph
        # ------------------------------------------------------------------
        #
        # IMPORTANT:
        #
        # We intentionally do NOT use:
        #
        #     create_graph_retriever()
        #
        # because this test needs to normalize the searchable property
        # names for the current operational Neo4j schema.
        #
        # The production GraphRetriever itself is NOT modified.
        #
        # ------------------------------------------------------------------

        retrieval_graph = GraphRetriever(
            config=Neo4jConfig.from_environment(),
            search_properties=DEEPEVAL_GRAPH_SEARCH_PROPERTIES,
        )

        assert retrieval_graph.verify_connection(), (
            "Existing SupportMind Neo4j GraphRetriever "
            "could not connect."
        )

        print(
            "[DeepEval Graph-RAG] "
            "Retrieval graph: connected"
        )

        print(
            "[DeepEval Graph-RAG] "
            "Test search properties: "
            f"{retrieval_graph.search_properties}"
        )

        # ------------------------------------------------------------------
        # 4. Existing RetrievalService
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

        # ------------------------------------------------------------------
        # 5. Help Desk Intelligence Graph
        # ------------------------------------------------------------------

        operational_graph = (
            create_neo4j_operational_graph()
        )

        assert operational_graph.health_check()[
            "healthy"
        ], (
            "Help Desk Intelligence Neo4j graph "
            "is not healthy."
        )

        print(
            "[DeepEval Graph-RAG] "
            "Intelligence graph: healthy"
        )

        intelligence_adapter = IntelligenceAdapter(
            operational_graph
        )

        # ------------------------------------------------------------------
        # 6. Real Ollama generation
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

        print(
            "[DeepEval Graph-RAG] "
            f"Generation model: {OLLAMA_MODEL}"
        )

        generator = GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=OLLAMA_MODEL,
        )

        # ------------------------------------------------------------------
        # 7. Complete Graph-RAG service
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

        # ------------------------------------------------------------------
        # 8. Real evaluation question
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
        # 9. REAL SupportMind Graph-RAG execution
        # ------------------------------------------------------------------

        response = service.answer(
            graph_rag_request
        )

        # ------------------------------------------------------------------
        # 10. Basic response validation
        # ------------------------------------------------------------------

        assert response.answer.strip(), (
            "Graph-RAG returned an empty answer."
        )

        assert response.evidence, (
            "Graph-RAG returned no evidence."
        )

        # ------------------------------------------------------------------
        # 11. Diagnostics
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] GENERATED ANSWER")
        print("=" * 80)
        print(response.answer)

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] AVAILABLE EVIDENCE")
        print("=" * 80)

        for evidence in response.evidence:

            print(
                f"\n[{evidence.evidence_id}] "
                f"source={evidence.source.value} "
                f"type={evidence.evidence_type.value}"
            )

            print(
                f"Reference: "
                f"{evidence.source_reference}"
            )

            print(
                f"Content: "
                f"{evidence.text}"
            )

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] GROUNDING RESULT")
        print("=" * 80)

        print(
            f"Status: "
            f"{response.grounding.status}"
        )

        print(
            f"Valid: "
            f"{response.grounding.valid}"
        )

        print(
            f"Cited evidence IDs: "
            f"{response.grounding.cited_evidence_ids}"
        )

        print(
            f"Invalid citation IDs: "
            f"{response.grounding.invalid_citation_ids}"
        )

        print(
            f"Unsupported claims: "
            f"{response.grounding.unsupported_claims}"
        )

        print(
            f"Issues: "
            f"{response.grounding.issues}"
        )

        print(
            f"Metadata: "
            f"{response.grounding.metadata}"
        )

        # ------------------------------------------------------------------
        # 12. Retrieval evidence validation
        # ------------------------------------------------------------------

        retrieval_sources = {
            GraphRAGEvidenceSource.RETRIEVAL_DENSE,
            GraphRAGEvidenceSource.RETRIEVAL_SPARSE,
            GraphRAGEvidenceSource.RETRIEVAL_HYBRID,
            GraphRAGEvidenceSource.RETRIEVAL_GRAPH,
        }

        actual_sources = {
            evidence.source
            for evidence in response.evidence
        }

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] EVIDENCE SOURCES")
        print("=" * 80)

        print(
            f"Actual sources: "
            f"{sorted(source.value for source in actual_sources)}"
        )

        assert actual_sources.intersection(
            retrieval_sources
        ), (
            "No evidence from the existing Retrieval Layer "
            "was present."
        )

        # ------------------------------------------------------------------
        # 13. Intelligence evidence validation
        # ------------------------------------------------------------------

        intelligence_sources = {
            GraphRAGEvidenceSource.ORGANIZATION_INTELLIGENCE,
            GraphRAGEvidenceSource.HELPDESK_GRAPH,
        }

        assert actual_sources.intersection(
            intelligence_sources
        ), (
            "No Help Desk Intelligence evidence was "
            "present in the fused Graph-RAG context."
        )

        # ------------------------------------------------------------------
        # 14. Existing deterministic grounding validation
        # ------------------------------------------------------------------

        assert (
            response.grounding.status
            == GroundingStatus.GROUNDED
        ), (
            "Graph-RAG answer failed structural grounding validation.\n"
            f"Status: {response.grounding.status}\n"
            f"Cited IDs: "
            f"{response.grounding.cited_evidence_ids}\n"
            f"Invalid IDs: "
            f"{response.grounding.invalid_citation_ids}\n"
            f"Issues: "
            f"{response.grounding.issues}\n"
            f"Answer: "
            f"{response.answer}"
        )

        assert response.grounding.valid is True

        assert response.grounding.cited_evidence_ids, (
            "Grounded response contains no cited evidence IDs."
        )

        # ------------------------------------------------------------------
        # 15. Build REAL DeepEval context
        # ------------------------------------------------------------------

        retrieval_context = [
            evidence.text
            for evidence in response.evidence
            if evidence.text.strip()
        ]

        assert retrieval_context, (
            "No textual retrieval context was available "
            "for DeepEval."
        )

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] DEEPEVAL CONTEXT")
        print("=" * 80)

        print(
            f"Context items: "
            f"{len(retrieval_context)}"
        )

        # ------------------------------------------------------------------
        # 16. DeepEval local Qwen3 model
        # ------------------------------------------------------------------

        evaluation_model = Qwen3DeepEvalModel(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.0,
        )

        # ------------------------------------------------------------------
        # 17. DeepEval test case
        # ------------------------------------------------------------------

        test_case = LLMTestCase(
            input=query,
            actual_output=response.answer,
            retrieval_context=retrieval_context,
        )

        # ------------------------------------------------------------------
        # 18. Faithfulness
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] FAITHFULNESS")
        print("=" * 80)

        faithfulness_metric = FaithfulnessMetric(
            threshold=0.7,
            model=evaluation_model,
            include_reason=True,
            async_mode=False,
        )

        faithfulness_metric.measure(
            test_case
        )

        print(
            f"Score: "
            f"{faithfulness_metric.score}"
        )

        print(
            f"Reason: "
            f"{faithfulness_metric.reason}"
        )

        assert faithfulness_metric.score is not None

        assert faithfulness_metric.score >= 0.7, (
            "DeepEval faithfulness score is below "
            "the required threshold.\n"
            f"Score: {faithfulness_metric.score}\n"
            f"Reason: {faithfulness_metric.reason}"
        )

        # ------------------------------------------------------------------
        # 19. Answer Relevancy
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] ANSWER RELEVANCY")
        print("=" * 80)

        answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7,
            model=evaluation_model,
            include_reason=True,
            async_mode=False,
        )

        answer_relevancy_metric.measure(
            test_case
        )

        print(
            f"Score: "
            f"{answer_relevancy_metric.score}"
        )

        print(
            f"Reason: "
            f"{answer_relevancy_metric.reason}"
        )

        assert answer_relevancy_metric.score is not None

        assert answer_relevancy_metric.score >= 0.7, (
            "DeepEval answer relevancy score is below "
            "the required threshold.\n"
            f"Score: {answer_relevancy_metric.score}\n"
            f"Reason: {answer_relevancy_metric.reason}"
        )

        # ------------------------------------------------------------------
        # 20. Save evaluation result
        # ------------------------------------------------------------------
        #
        # Contextual Relevancy is intentionally NOT included.
        #
        # Only the metrics that were actually evaluated are stored.
        #
        # ------------------------------------------------------------------

        deepeval_metrics = {
            "faithfulness": {
                "score": faithfulness_metric.score,
                "threshold": faithfulness_metric.threshold,
                "success": faithfulness_metric.is_successful(),
                "reason": faithfulness_metric.reason,
            },
            "answer_relevancy": {
                "score": answer_relevancy_metric.score,
                "threshold": answer_relevancy_metric.threshold,
                "success": answer_relevancy_metric.is_successful(),
                "reason": answer_relevancy_metric.reason,
            },
        }

        saved_result = save_deepeval_result(
            query=query,
            answer=response.answer,
            retrieval_context=retrieval_context,
            metrics=deepeval_metrics,
            grounding={
                "status": response.grounding.status.value,
                "valid": response.grounding.valid,
                "cited_evidence_ids": (
                    response.grounding.cited_evidence_ids
                ),
            },
            metadata={
                "qdrant_collection": collection_name,
                "model": OLLAMA_MODEL,
                "provider": "ollama",
                "top_k": 5,
                "reranking": False,
                "metrics_evaluated": [
                    "faithfulness",
                    "answer_relevancy",
                ],
            },
        )

        assert saved_result is not None

        print()
        print("=" * 80)
        print("DEEPEVAL RESULT SAVED")
        print("=" * 80)

        print(
            f"Result file: {LATEST_RESULT_PATH}"
        )

        # ------------------------------------------------------------------
        # 21. Final evaluation summary
        # ------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("[DeepEval Graph-RAG] FINAL EVALUATION")
        print("=" * 80)

        print(
            f"Faithfulness: "
            f"{faithfulness_metric.score:.4f}"
        )

        print(
            f"Answer Relevancy: "
            f"{answer_relevancy_metric.score:.4f}"
        )

        print(
            "\n[DeepEval Graph-RAG] "
            "ALL EVALUATIONS PASSED"
        )

        # ------------------------------------------------------------------
        # 22. Latency
        # ------------------------------------------------------------------

        assert response.latency_ms is not None
        assert response.latency_ms >= 0

    finally:

        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------

        if retrieval_service is not None:
            try:
                retrieval_service.close()
            except Exception:
                pass

        if retrieval_graph is not None:
            try:
                retrieval_graph.close()
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