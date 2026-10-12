from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from app.retrieval.graph_rag.api import (
    create_graph_rag_app,
    get_graph_rag_runtime,
)
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGEvidence,
    GraphRAGEvidenceSource,
    GraphRAGEvidenceType,
    GraphRAGResponse,
    GroundingResult,
    GroundingStatus,
)


class FakeGraphRAGService:
    """
    Deterministic fake Graph-RAG service for API unit tests.
    """

    def __init__(self) -> None:
        self.last_request = None

    def answer(
        self,
        request: Any,
    ) -> GraphRAGResponse:
        self.last_request = request

        evidence = [
            GraphRAGEvidence(
                evidence_id="E1",
                source=GraphRAGEvidenceSource.RETRIEVAL_SPARSE,
                evidence_type=GraphRAGEvidenceType.DOCUMENT,
                text="Synthetic grounded evidence.",
                source_reference="test-document",
                score=1.0,
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
            query=request.query,
            answer=(
                "The test answer is grounded in "
                "the supplied evidence. [E1]"
            ),
            evidence=evidence,
            grounding=grounding,
            provider="fake",
            model_name="fake-model",
            latency_ms=1.0,
            metadata={},
        )


@dataclass
class FakeRuntime:
    service: FakeGraphRAGService
    collection_name: str = (
        "supportmind_retrieval_pdf_test"
    )
    use_dense: bool = False
    use_reranker: bool = False
    ollama_model: str = "fake-model"

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "qdrant": {
                "healthy": True,
                "collection": self.collection_name,
            },
            "retrieval_graph": {
                "healthy": True,
            },
            "intelligence_graph": {
                "healthy": True,
            },
            "generation": {
                "provider": "fake",
                "model": self.ollama_model,
                "configured": True,
            },
        }


def test_graph_rag_query_api() -> None:
    fake_service = FakeGraphRAGService()

    fake_runtime = FakeRuntime(
        service=fake_service
    )

    app = create_graph_rag_app()

    app.dependency_overrides[
        get_graph_rag_runtime
    ] = lambda: fake_runtime

    client = TestClient(app)

    response = client.post(
        "/api/graph-rag/query",
        json={
            "query": (
                "What are the main support issues?"
            ),
            "top_k": 5,
            "include_graph": True,
            "rerank": False,
            "include_intelligence": True,
            "include_workflow": True,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert (
        body["query"]
        == "What are the main support issues?"
    )

    assert body["answer"]

    assert body["grounding"]["valid"] is True

    assert body["grounding"][
        "cited_evidence_ids"
    ] == ["E1"]

    assert (
        fake_service.last_request is not None
    )

    assert (
        fake_service.last_request.query
        == "What are the main support issues?"
    )

    assert (
        fake_service.last_request.retrieval_request.top_k
        == 5
    )

    app.dependency_overrides.clear()


def test_graph_rag_health_api() -> None:
    fake_runtime = FakeRuntime(
        service=FakeGraphRAGService()
    )

    app = create_graph_rag_app()

    app.dependency_overrides[
        get_graph_rag_runtime
    ] = lambda: fake_runtime

    client = TestClient(app)

    response = client.get(
        "/api/graph-rag/health"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert body["qdrant"]["healthy"] is True

    assert (
        body["retrieval_graph"]["healthy"]
        is True
    )

    assert (
        body["intelligence_graph"]["healthy"]
        is True
    )

    app.dependency_overrides.clear()