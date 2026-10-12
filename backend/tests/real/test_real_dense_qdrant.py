"""
Real Dense + Qdrant retrieval test.

Uses:
    - Real embedding model
    - Real Qdrant instance
    - Existing Qdrant collection
    - Application VectorPipeline
    - DenseRetriever

No mocks.
No fake vectors.
No fake retrieval results.
"""

from __future__ import annotations

import os

import pytest
from sentence_transformers import SentenceTransformer
from app.vector_store.vector_pipeline import VectorPipeline
from app.vector_store.vector_registry import VectorStoreRegistry
from app.retrieval.dense.dense_retriever import DenseRetriever
from app.retrieval.models.retrieval_models import RetrievalRequest
from app.vector_store.qdrant_store import QdrantStore
from app.vector_store.vector_pipeline import vector_pipeline


QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "127.0.0.1",
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        "6333",
    )
)

QDRANT_API_KEY = os.getenv(
    "QDRANT_API_KEY"
) or None

QDRANT_HTTPS = (
    os.getenv(
        "QDRANT_HTTPS",
        "false",
    ).lower()
    == "true"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "supportmind_retrieval_pdf_test",
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-base-en-v1.5",
)


@pytest.fixture(scope="module")
def qdrant_store() -> QdrantStore:
    """
    Connect to the real Qdrant instance.
    """

    store = QdrantStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=QDRANT_HTTPS,
        timeout=30,
    )

    if not store.health_check():
        pytest.fail(
            "Qdrant is not reachable."
        )

    if not store.collection_exists(
        COLLECTION_NAME
    ):
        pytest.fail(
            f"Qdrant collection does not exist: "
            f"{COLLECTION_NAME}"
        )

    return store


@pytest.fixture(scope="module")
def embedding_model() -> SentenceTransformer:
    """
    Load the same embedding model configured for
    the real Qdrant collection.
    """

    try:
        return SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    except Exception as exc:
        pytest.fail(
            f"Unable to load embedding model "
            f"'{EMBEDDING_MODEL_NAME}': {exc}"
        )
@pytest.fixture(scope="module")
def real_vector_pipeline(
    qdrant_store: QdrantStore,
) -> VectorPipeline:
    """
    Build a dedicated VectorPipeline backed by
    the real Qdrant instance.
    """

    registry = VectorStoreRegistry()

    pipeline = VectorPipeline(
        registry=registry,
    )

    pipeline.register_provider(
        qdrant_store,
        make_default=True,
    )

    return pipeline
def test_real_dense_qdrant_retrieval(
    qdrant_store: QdrantStore,
    embedding_model: SentenceTransformer,
    real_vector_pipeline: VectorPipeline,
) -> None:
    """
    Verify real dense retrieval through Qdrant.
    """

    # ------------------------------------------------------
    # Verify Qdrant actually contains vectors.
    # ------------------------------------------------------

    vector_count = qdrant_store.count(
        COLLECTION_NAME
    )

    assert vector_count > 0, (
        f"Qdrant collection '{COLLECTION_NAME}' "
        "contains zero vectors."
    )

    # ------------------------------------------------------
    # Verify embedding dimension.
    # ------------------------------------------------------

    query = (
        "What is the refund policy and "
        "when is a customer eligible for a refund?"
    )

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
    )

    query_vector = [
        float(value)
        for value in query_embedding.tolist()
    ]

    collection_dimension = (
        qdrant_store._get_collection_dimension(
            COLLECTION_NAME
        )
    )

    assert len(query_vector) == (
        collection_dimension
    ), (
        "Embedding dimension does not match "
        f"Qdrant collection. "
        f"Model produced {len(query_vector)}, "
        f"Qdrant expects {collection_dimension}."
    )

    # ------------------------------------------------------
    # Build actual RetrievalRequest.
    # ------------------------------------------------------

    request = RetrievalRequest.model_validate(
        {
            "query": query,
            "top_k": 5,
            "collection_name": COLLECTION_NAME,
            "provider": "qdrant",
        }
    )

    # ------------------------------------------------------
    # Actual application DenseRetriever.
    # ------------------------------------------------------

    retriever = DenseRetriever(
    pipeline=real_vector_pipeline
)

    results = retriever.retrieve(
        request=request,
        query_vector=query_vector,
    )

    # ------------------------------------------------------
    # Assertions.
    # ------------------------------------------------------

    assert results, (
        "DenseRetriever returned no results "
        "from the real Qdrant collection."
    )

    assert len(results) <= 5

    # ------------------------------------------------------
    # Retrieval contract.
    # ------------------------------------------------------

    assert all(
        result.source == "dense"
        for result in results
    )

    assert [
        result.rank
        for result in results
    ] == list(
        range(
            1,
            len(results) + 1,
        )
    )

    assert all(
        isinstance(result.score, float)
        for result in results
    )

    assert all(
        "dense_score"
        in result.retrieval_scores
        for result in results
    )

    assert all(
        result.text.strip()
        for result in results
    )

    assert all(
        result.node_id
        for result in results
    )

    assert all(
        result.document_id
        for result in results
    )

    # ------------------------------------------------------
    # Scores should be descending because Qdrant returns
    # nearest results first.
    # ------------------------------------------------------

    scores = [
        result.score
        for result in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )

    # ------------------------------------------------------
    # Real semantic evidence.
    #
    # At least one returned result should contain refund
    # related terminology.
    # ------------------------------------------------------

    combined_text = " ".join(
        result.text.lower()
        for result in results
    )

    refund_terms = (
        "refund",
        "return",
        "eligible",
        "eligibility",
        "reimbursement",
    )

    assert any(
        term in combined_text
        for term in refund_terms
    ), (
        "Qdrant returned results, but none of the "
        "top results contained recognizable refund "
        "policy terminology."
    )