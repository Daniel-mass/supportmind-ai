"""
Real end-to-end Graph-RAG integration test.

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
    GraphRAGResponse

Run explicitly with:

    $env:RUN_REAL_GRAPH_RAG="1"
    $env:PYTHONPATH="$PWD\backend"
    pytest backend\tests\real\test_real_graph_rag.py -v -s
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from qdrant_client import QdrantClient

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
    GraphRAGEvidenceSource,
    GraphRAGRequest,
    GroundingStatus,
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
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidenceType,
)
from app.retrieval.graph_rag.graph_rag_service import (
    GraphRAGService,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
)

from app.retrieval.graph_rag.graph_retriever import (
    create_graph_retriever,
)
from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)
from app.retrieval.sparse.sparse_retriever import (
    SparseRetriever,
)

from backend.app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)


# ============================================================================
# Environment
# ============================================================================


RUN_REAL_GRAPH_RAG = os.getenv(
    "RUN_REAL_GRAPH_RAG",
    "0",
)

if RUN_REAL_GRAPH_RAG != "1":
    pytest.skip(
        "Real Graph-RAG integration tests are disabled. "
        "Set RUN_REAL_GRAPH_RAG=1 to run.",
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

# Optional explicit collection name.
SUPPORTMIND_COLLECTION = os.getenv(
    "SUPPORTMIND_QDRANT_COLLECTION"
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


def test_real_graph_rag_end_to_end() -> None:
    """
    Execute the complete real Graph-RAG generation pipeline.
    """

    qdrant_client: QdrantClient | None = None
    retrieval_service: RetrievalService | None = None
    retrieval_graph = None
    operational_graph = None
    ollama_provider: OllamaProvider | None = None

    try:
        # --------------------------------------------------------------
        # 1. Qdrant
        # --------------------------------------------------------------

        qdrant_client = QdrantClient(
            url=QDRANT_URL,
        )

        collections = (
            qdrant_client.get_collections()
        )

        assert collections.collections, (
            "Qdrant is reachable but no collections "
            "were found."
        )

        collection_name = (
            resolve_supportmind_collection(
                qdrant_client
            )
        )

        print(
            f"\n[Graph-RAG] Qdrant collection: "
            f"{collection_name}"
        )

        # --------------------------------------------------------------
        # 2. Build the real sparse retriever from Qdrant
        # --------------------------------------------------------------

        sparse_retriever = (
            SparseRetriever.from_qdrant(
                qdrant_store=(
                    QdrantStoreAdapter(
                        qdrant_client
                    )
                ),
                collection_name=collection_name,
                batch_size=100,
            )
        )

        assert sparse_retriever.is_indexed

        print(
            "[Graph-RAG] BM25 documents: "
            f"{sparse_retriever.document_count}"
        )

        # --------------------------------------------------------------
        # 3. Existing SupportMind retrieval Graph-RAG
        # --------------------------------------------------------------

        retrieval_graph = (
            create_graph_retriever()
        )

        assert retrieval_graph.verify_connection(), (
            "Existing SupportMind Neo4j GraphRetriever "
            "could not connect."
        )

        print(
            "[Graph-RAG] Existing retrieval graph: connected"
        )

        # --------------------------------------------------------------
        # 4. Existing RetrievalService
        #
        # Dense is disabled here because query-vector generation is
        # deliberately outside RetrievalRequest and has already been
        # validated by the dense retrieval integration tests.
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # 5. Real Help Desk Intelligence Graph
        # --------------------------------------------------------------

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
            "[Graph-RAG] Intelligence graph: healthy"
        )

        intelligence_adapter = (
            IntelligenceAdapter(
                operational_graph
            )
        )

        # --------------------------------------------------------------
        # 6. Real Ollama
        # --------------------------------------------------------------

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
            "[Graph-RAG] Ollama model: "
            f"{OLLAMA_MODEL}"
        )

        generator = GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=OLLAMA_MODEL,
        )

        # --------------------------------------------------------------
        # 7. Complete Graph-RAG service
        # --------------------------------------------------------------

        service = GraphRAGService(
            retrieval_service=retrieval_service,
            intelligence_adapter=(
                intelligence_adapter
            ),
            evidence_fusion=EvidenceFusion(
                default_max_items=20,
                default_max_characters=16000,
            ),
            generator=generator,
            groundedness_validator=(
                GroundednessValidator()
            ),
        )

        # --------------------------------------------------------------
        # 8. Real question
        #
        # This intentionally asks an operational support question that
        # can be supported by both:
        #   - customer-support retrieval/graph evidence
        #   - Help Desk organization/workflow evidence
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # 9. REAL END-TO-END EXECUTION
        # --------------------------------------------------------------

        response = service.answer(
            graph_rag_request
        )

        # --------------------------------------------------------------
        # 10. Diagnostics
        # --------------------------------------------------------------

        print("\n" + "=" * 72)
        print("REAL GRAPH-RAG RESPONSE")
        print("=" * 72)

        print("\nANSWER:")
        print(response.answer)

        print("\nGROUNDING:")
        print(
            response.grounding.model_dump()
        )

        print("\nEVIDENCE:")
        for evidence in response.evidence:
            print(
                f"\n[{evidence.evidence_id}] "
                f"{evidence.source.value} / "
                f"{evidence.evidence_type.value}"
            )
            print(
                f"Reference: "
                f"{evidence.source_reference}"
            )
            print(
                f"Content: "
                f"{evidence.text}"
            )

        print("\nMETADATA:")
        print(
            response.metadata
        )

        # --------------------------------------------------------------
        # 11. Assertions
        # --------------------------------------------------------------

        assert response.answer.strip()

        assert response.evidence, (
            "Graph-RAG returned no evidence."
        )

        # We expect evidence from the existing retrieval layer.
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

        assert actual_sources.intersection(
            retrieval_sources
        ), (
            "No evidence from the existing Retrieval Layer "
            "was present."
        )

        # We also expect evidence from the Help Desk /
        # Intelligence graph.
        assert (
            GraphRAGEvidenceSource
            .ORGANIZATION_INTELLIGENCE
            in actual_sources
            or
            GraphRAGEvidenceSource.HELPDESK_GRAPH
            in actual_sources
        ), (
            "No Help Desk Intelligence evidence was "
            "present in the fused Graph-RAG context."
        )

        print("\n[DeepEval Graph-RAG] Generated answer:")
        print(response.answer)

        print("\n[DeepEval Graph-RAG] Available evidence IDs:")
        for evidence in response.evidence:
            print(
                f"  {evidence.evidence_id}: "
                f"{evidence.text[:300]}"
            )

        print("\n[DeepEval Graph-RAG] Grounding result:")
        print(f"  status: {response.grounding.status}")
        print(f"  valid: {response.grounding.valid}")
        print(
            f"  cited IDs: "
            f"{response.grounding.cited_evidence_ids}"
        )
        print(
            f"  invalid IDs: "
            f"{response.grounding.invalid_citation_ids}"
        )
        print(
            f"  issues: "
            f"{response.grounding.issues}"
        )

        assert (
            response.grounding.status
            == GroundingStatus.GROUNDED
        ), (
            "Graph-RAG answer failed structural grounding validation. "
            f"status={response.grounding.status}, "
            f"cited_ids={response.grounding.cited_evidence_ids}, "
            f"invalid_ids={response.grounding.invalid_citation_ids}, "
            f"issues={response.grounding.issues}"
        )

        assert (
            response.grounding.valid
            is True
        )

        assert response.grounding.cited_evidence_ids

        assert response.latency_ms is not None
        assert response.latency_ms >= 0

    finally:
        # --------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------

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