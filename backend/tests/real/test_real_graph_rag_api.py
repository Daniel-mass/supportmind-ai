from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
import json
from app.retrieval.graph_rag.api import (
    create_graph_rag_app,
)


@pytest.mark.skipif(
    os.getenv(
        "RUN_REAL_GRAPH_RAG_API"
    ) != "1",
    reason=(
        "Set RUN_REAL_GRAPH_RAG_API=1 "
        "to run the real Graph-RAG API integration test."
    ),
)
def test_real_graph_rag_api_end_to_end() -> None:
    """
    Exercise the Graph-RAG system through the real HTTP API.

    Infrastructure:
        Qdrant
        Neo4j retrieval graph
        Neo4j Help Desk Intelligence graph
        Ollama

    Pipeline:
        HTTP
        -> API validation
        -> RetrievalService
        -> IntelligenceAdapter
        -> EvidenceFusion
        -> Ollama
        -> Groundedness
        -> GraphRAGResponse
    """

    app = create_graph_rag_app()

    with TestClient(app) as client:

        # --------------------------------------------------------------
        # 1. Health
        # --------------------------------------------------------------

        health_response = client.get(
            "/api/graph-rag/health"
        )

        assert (
            health_response.status_code == 200
        ), health_response.text

        health = health_response.json()

        print(
            "\n[Graph-RAG API] health:"
        )

        print(
            health
        )

        assert health["qdrant"][
            "healthy"
        ] is True

        assert health[
            "retrieval_graph"
        ]["healthy"] is True

        assert health[
            "intelligence_graph"
        ]["healthy"] is True

        # --------------------------------------------------------------
        # 2. Real Graph-RAG query
        # --------------------------------------------------------------

        query = (
            "What does the support data show about "
            "ticket priorities, statuses, workflow activity, "
            "and common operational delays?"
        )

        response = client.post(
            "/api/graph-rag/query",
            json={
                "query": query,
                "top_k": 5,
                "include_graph": True,
                "rerank": False,
                "include_intelligence": True,
                "include_workflow": True,
                "max_evidence_items": 20,
                "max_evidence_characters": 16000,
            },
        )

        assert (
            response.status_code == 200
        ), response.text

        body = response.json()

        # --------------------------------------------------------------
        # 3. Core response assertions
        # --------------------------------------------------------------

        assert body["query"] == query

        assert isinstance(
            body["answer"],
            str,
        )

        assert body["answer"].strip()

        assert body["evidence"]

        assert body["provider"] == "ollama"

        assert (
            body["model_name"]
        )

        assert body[
            "latency_ms"
        ] >= 0

        # --------------------------------------------------------------
        # 4. Grounding assertions
        # --------------------------------------------------------------

        grounding = body[
            "grounding"
        ]
        print("\n[Graph-RAG API] answer:")
        print(body["answer"])

        print("\n[Graph-RAG API] evidence IDs:")
        print([
            item["evidence_id"]
            for item in body["evidence"]
])

        print("\n[Graph-RAG API] grounding:")
        print(
            json.dumps(
            grounding,
            indent=2,
    )
)

        assert grounding[
            "valid"
        ] is True

        assert grounding[
            "status"
        ] == "grounded"

        assert grounding[
            "cited_evidence_ids"
        ]

        assert not grounding[
            "invalid_citation_ids"
        ]

        # --------------------------------------------------------------
        # 5. Evidence assertions
        # --------------------------------------------------------------

        evidence = body[
            "evidence"
        ]

        evidence_ids = {
            item["evidence_id"]
            for item in evidence
        }

        cited_ids = set(
            grounding[
                "cited_evidence_ids"
            ]
        )

        assert cited_ids.issubset(
            evidence_ids
        )

        # --------------------------------------------------------------
        # 6. Print useful real result
        # --------------------------------------------------------------

        print(
            "\n=============================="
        )

        print(
            "[Graph-RAG API] ANSWER"
        )

        print(
            "=============================="
        )

        print(
            body["answer"]
        )

        print(
            "\n=============================="
        )

        print(
            "[Graph-RAG API] GROUNDING"
        )

        print(
            "=============================="
        )

        print(
            grounding
        )

        print(
            "\n=============================="
        )

        print(
            "[Graph-RAG API] EVIDENCE"
        )

        print(
            "=============================="
        )

        for item in evidence:
            print(
                f"{item['evidence_id']} | "
                f"{item['source']} | "
                f"{item['evidence_type']} | "
                f"{item['source_reference']}"
            )