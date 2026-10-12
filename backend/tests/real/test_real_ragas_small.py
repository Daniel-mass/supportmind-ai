from __future__ import annotations

import asyncio
import os

import pytest
from qdrant_client import QdrantClient

from app.evaluation.contracts.evaluation import EvaluationCase
from app.evaluation.rag.ragas_evaluator import create_ragas_evaluator

from app.retrieval.graph_rag.evidence_fusion import EvidenceFusion
from app.retrieval.graph_rag.groundedness import (
    GroundednessValidator,
)
from app.retrieval.graph_rag.generator import GraphRAGGenerator
from app.retrieval.graph_rag.graph_rag_models import GraphRAGRequest
from app.retrieval.graph_rag.graph_rag_service import GraphRAGService
from app.retrieval.graph_rag.intelligence_adapter import (
    IntelligenceAdapter,
)
from app.retrieval.graph_rag.providers.ollama import (
    OllamaGenerationConfig,
    OllamaProvider,
)
from app.retrieval.models.retrieval_models import RetrievalRequest
from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)
from app.retrieval.graph_rag.graph_retriever import (
    create_graph_retriever,
)
from app.retrieval.sparse.sparse_retriever import SparseRetriever

from backend.app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)


pytestmark = pytest.mark.real


RUN_REAL_RAGAS = os.getenv(
    "RUN_REAL_RAGAS",
    "0",
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://127.0.0.1:11434",
)

# -------------------------------------------------------------
# Graph-RAG uses the real SupportMind generation model.
# -------------------------------------------------------------
GRAPH_RAG_MODEL = os.getenv(
    "SUPPORTMIND_GRAPH_RAG_MODEL",
    "qwen3:8b",
)

# -------------------------------------------------------------
# RAGAS uses a separate, smaller local judge model.
# -------------------------------------------------------------
RAGAS_JUDGE_MODEL = os.getenv(
    "SUPPORTMIND_RAGAS_JUDGE_MODEL",
    "llama3.2:latest",
)

COLLECTION_NAME = os.getenv(
    "SUPPORTMIND_QDRANT_COLLECTION",
)


class QdrantStoreAdapter:
    def __init__(
        self,
        client: QdrantClient,
    ) -> None:
        self.client = client


def test_real_ragas_faithfulness_small_context() -> None:
    if RUN_REAL_RAGAS != "1":
        pytest.skip(
            "Set RUN_REAL_RAGAS=1 to run the real RAGAS test."
        )

    if not COLLECTION_NAME:
        raise RuntimeError(
            "SUPPORTMIND_QDRANT_COLLECTION must be set."
        )

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
    )

    retrieval_service = None
    operational_graph = None
    ollama_provider = None

    try:
        print("\n")
        print("=" * 90)
        print(
            "REAL SUPPORTMIND GRAPH-RAG → "
            "RAGAS SMALL-CONTEXT TEST"
        )
        print("=" * 90)

        print(
            f"Graph-RAG model: {GRAPH_RAG_MODEL}"
        )
        print(
            f"RAGAS judge model: {RAGAS_JUDGE_MODEL}"
        )
        print(
            f"Qdrant collection: {COLLECTION_NAME}"
        )

        # ---------------------------------------------------------
        # 1. Create sparse retriever
        # ---------------------------------------------------------
        sparse_retriever = SparseRetriever.from_qdrant(
            qdrant_store=QdrantStoreAdapter(qdrant_client),
            collection_name=COLLECTION_NAME,
            batch_size=100,
        )

        print(
            f"BM25 documents: "
            f"{sparse_retriever.document_count}"
        )

        # ---------------------------------------------------------
        # 2. Create retrieval graph
        # ---------------------------------------------------------
        retrieval_graph = create_graph_retriever()

        if not retrieval_graph.verify_connection():
            raise RuntimeError(
                "Retrieval graph connection failed."
            )

        print("[OK] Retrieval graph connected.")

        # ---------------------------------------------------------
        # 3. Create RetrievalService
        # ---------------------------------------------------------
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

        print("[OK] RetrievalService created.")

        # ---------------------------------------------------------
        # 4. Create operational graph
        # ---------------------------------------------------------
        operational_graph = (
            create_neo4j_operational_graph()
        )

        health = operational_graph.health_check()

        if not health.get("healthy", False):
            raise RuntimeError(
                f"Operational graph is unhealthy: {health}"
            )

        print("[OK] Intelligence graph healthy.")

        intelligence_adapter = IntelligenceAdapter(
            operational_graph
        )

        # ---------------------------------------------------------
        # 5. Create Graph-RAG generator
        # ---------------------------------------------------------
        ollama_provider = OllamaProvider(
            config=OllamaGenerationConfig(
                host=OLLAMA_HOST,
                model_name=GRAPH_RAG_MODEL,
                temperature=0.1,
                max_tokens=700,
                timeout=180.0,
            )
        )

        generator = GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=GRAPH_RAG_MODEL,
        )

        print("[OK] Graph-RAG generator created.")

        # ---------------------------------------------------------
        # 6. Create GraphRAGService
        # ---------------------------------------------------------
        graph_rag_service = GraphRAGService(
            retrieval_service=retrieval_service,
            intelligence_adapter=intelligence_adapter,
            evidence_fusion=EvidenceFusion(
                default_max_items=20,
                default_max_characters=16000,
            ),
            generator=generator,
            groundedness_validator=GroundednessValidator(),
        )

        print("[OK] GraphRAGService created.")

        # ---------------------------------------------------------
        # 7. Run REAL Graph-RAG
        # ---------------------------------------------------------
        query = (
            "What does the support data show about ticket "
            "priorities, statuses, workflow activity, and "
            "common operational delays?"
        )

        retrieval_request = RetrievalRequest(
            query=query,
            top_k=5,
            collection_name=COLLECTION_NAME,
            include_graph=True,
            rerank=False,
        )

        request = GraphRAGRequest(
            query=query,
            retrieval_request=retrieval_request,
            include_intelligence=True,
            include_workflow=True,
            max_evidence_items=20,
            max_evidence_characters=16000,
        )

        print("\n[RUNNING] Real Graph-RAG...")

        response = graph_rag_service.answer(
            request
        )

        assert response.answer.strip()
        assert response.evidence

        print(
            "\n[OK] Real Graph-RAG completed."
        )

        print(
            f"Full evidence items: "
            f"{len(response.evidence)}"
        )

        print("\nGenerated answer:")
        print("-" * 90)
        print(response.answer)
        print("-" * 90)

        # ---------------------------------------------------------
        # 8. Keep only 3 REAL evidence items.
        #
        # This is intentionally a diagnostic test.
        # The final evaluation will use the full relevant
        # evidence set.
        # ---------------------------------------------------------
        small_evidence = response.evidence[:3]

        print(
            "\nRAGAS diagnostic context items: "
            f"{len(small_evidence)}"
        )

        for index, evidence in enumerate(
            small_evidence,
            start=1,
        ):
            print(
                f"\nEvidence {index}: "
                f"{evidence.source} / "
                f"{evidence.evidence_type}"
            )

            print(
                evidence.text[:1000]
            )

        # ---------------------------------------------------------
        # 9. Build EvaluationCase from REAL Graph-RAG response
        # ---------------------------------------------------------
        evaluation_case = EvaluationCase(
            case_id="real_graph_rag_ragas_small_001",
            query=response.query,
            retrieved_context=[
                evidence.text
                for evidence in small_evidence
            ],
            generated_answer=response.answer,
        )

        print(
            "\n[OK] EvaluationCase created."
        )

        # ---------------------------------------------------------
        # 10. Create REAL RAGAS evaluator
        #
        # IMPORTANT:
        # Graph-RAG remains Qwen3:8b.
        # RAGAS uses the separate smaller judge model.
        # ---------------------------------------------------------
        evaluator = create_ragas_evaluator(
            model_name=RAGAS_JUDGE_MODEL,
            base_url=f"{OLLAMA_HOST}/v1",
        )

        print(
            "[OK] RAGAS evaluator created."
        )

        # ---------------------------------------------------------
        # 11. Run ONLY Faithfulness
        # ---------------------------------------------------------
        async def run_faithfulness():
            return await evaluator.faithfulness.ascore(
                user_input=evaluation_case.query,
                response=evaluation_case.generated_answer,
                retrieved_contexts=(
                    evaluation_case.retrieved_context
                ),
            )

        print(
            "\n[RUNNING] "
            "RAGAS Faithfulness only..."
        )

        result = asyncio.run(
            run_faithfulness()
        )

        print("\n")
        print("=" * 90)
        print("RAGAS FAITHFULNESS RESULT")
        print("=" * 90)
        print(
            f"Judge model: {RAGAS_JUDGE_MODEL}"
        )
        print(
            f"Score:  {result.value}"
        )
        print(
            f"Reason: {result.reason}"
        )
        print("=" * 90)

        assert result.value is not None

        assert (
            0.0
            <= float(result.value)
            <= 1.0
        )

    finally:
        # ---------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------
        if retrieval_service is not None:
            retrieval_service.close()

        if operational_graph is not None:
            operational_graph.close()

        if ollama_provider is not None:
            ollama_provider.close()

        qdrant_client.close()

