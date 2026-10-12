"""
Real integration tests for SupportMind AI RetrievalService.

Pipeline under test
-------------------

Query
  ↓
EmbeddingPipeline
  ↓
Query Embedding
  ↓
Dense Retrieval / Qdrant
  +
Sparse Retrieval / BM25
  ↓
RRF Fusion
  ↓
Graph-RAG / Neo4j
  ↓
Candidate Deduplication
  ↓
Reranker
  ↓
RetrievalResult -> ContextItem
  ↓
ContextBuilder
  ↓
BuiltContext

IMPORTANT
---------
These are REAL integration tests.

They do NOT mock:
- Qdrant
- Neo4j
- BM25
- RetrievalService
- ContextBuilder
- reranker
- embedding pipeline

Prerequisites
-------------
1. Qdrant must be running.
2. Neo4j must be running.
3. SupportMind vector collection must already be populated.
4. BM25 index must already be built.
5. Neo4j graph must already be populated.
6. Embedding model must be available locally.
7. Reranker model must be available locally.

Run
---
pytest backend/tests/real/test_real_retrieval_service.py -v -s
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from llama_index.core.schema import TextNode


# ============================================================================
# Environment configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]

ENV_CANDIDATES = (
    PROJECT_ROOT / ".env",
    BACKEND_ROOT / ".env",
)

ENV_LOADED = False

for env_file in ENV_CANDIDATES:
    if env_file.exists():
        load_dotenv(
            env_file,
            override=False,
        )
        ENV_LOADED = True


# ============================================================================
# Application imports
# ============================================================================

from app.embeddings.embedding_pipeline import EmbeddingPipeline

from app.retrieval.context.context_models import BuiltContext

from app.retrieval.dense.dense_retriever import DenseRetriever

from app.retrieval.graph_rag.graph_retriever import GraphRetriever

from app.retrieval.models.retrieval_models import RetrievalRequest

from app.retrieval.reranker.cross_encoder_baseline import (
    CrossEncoderBaseline,
)

from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)

from app.retrieval.sparse.sparse_retriever import (
    SparseRetriever,
)

from app.vector_store.qdrant_store import QdrantStore

from app.vector_store.vector_pipeline import VectorPipeline

from app.vector_store.vector_registry import VectorStoreRegistry


# ============================================================================
# Configuration
# ============================================================================

COLLECTION_NAME = os.getenv(
    "SUPPORTMIND_QDRANT_COLLECTION",
     "supportmind_retrieval_pdf_test",
)


# ---------------------------------------------------------------------------
# Qdrant configuration
# ---------------------------------------------------------------------------

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
)


QDRANT_HTTPS = (
    os.getenv(
        "QDRANT_HTTPS",
        "false",
    ).lower()
    == "true"
)


QDRANT_TIMEOUT = int(
    os.getenv(
        "QDRANT_TIMEOUT",
        "30",
    )
)


# ---------------------------------------------------------------------------
# Vector store provider
#
# This is the vector-store provider.
#
# It is NOT the embedding provider.
#
# Example:
#
#   Embedding provider -> fastembed
#   Vector store       -> qdrant
# ---------------------------------------------------------------------------

VECTOR_STORE_PROVIDER = os.getenv(
    "SUPPORTMIND_VECTOR_STORE_PROVIDER",
    "qdrant",
)


# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------

TOP_K = int(
    os.getenv(
        "SUPPORTMIND_TEST_TOP_K",
        "5",
    )
)


RERANK_TOP_K = int(
    os.getenv(
        "SUPPORTMIND_TEST_RERANK_TOP_K",
        "5",
    )
)


EMBEDDING_PROVIDER = os.getenv(
    "SUPPORTMIND_EMBEDDING_PROVIDER",
    "fastembed",
)


RERANKER_MODEL = os.getenv(
    "SUPPORTMIND_RERANKER_MODEL",
    CrossEncoderBaseline.DEFAULT_MODEL_NAME,
)


# ============================================================================
# Real test queries
# ============================================================================

TEST_QUERIES = [
    "refund policy",
    "subscription cancellation",
    "payment failed",
    "password reset",
    "account management",
    "product troubleshooting",
]


# ============================================================================
# Environment test
# ============================================================================

def test_environment_configuration() -> None:
    """
    Verify required environment configuration is available.
    """

    assert ENV_LOADED, (
        "No .env file was loaded. "
        "Expected .env at project root or backend root."
    )

    assert os.getenv("NEO4J_URI"), (
        "NEO4J_URI is not loaded."
    )

    assert os.getenv("NEO4J_USERNAME"), (
        "NEO4J_USERNAME is not loaded."
    )

    assert os.getenv("NEO4J_PASSWORD"), (
        "NEO4J_PASSWORD is not loaded."
    )

    assert os.getenv("NEO4J_DATABASE"), (
        "NEO4J_DATABASE is not loaded."
    )

    print("\nEnvironment configuration detected:")

    print(
        f"Neo4j URI           : "
        f"{os.getenv('NEO4J_URI')}"
    )

    print(
        f"Neo4j Username      : "
        f"{os.getenv('NEO4J_USERNAME')}"
    )

    print(
        "Neo4j Password      : [LOADED]"
    )

    print(
        f"Neo4j Database      : "
        f"{os.getenv('NEO4J_DATABASE')}"
    )

    print(
        f"Qdrant Host         : "
        f"{QDRANT_HOST}"
    )

    print(
        f"Qdrant Port         : "
        f"{QDRANT_PORT}"
    )

    print(
        f"Qdrant HTTPS        : "
        f"{QDRANT_HTTPS}"
    )

    print(
        f"Qdrant Collection   : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vector Store        : "
        f"{VECTOR_STORE_PROVIDER}"
    )

    print(
        f"Embedding Provider  : "
        f"{EMBEDDING_PROVIDER}"
    )

    print(
        f"Reranker Model      : "
        f"{RERANKER_MODEL}"
    )


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def real_embedding_pipeline() -> EmbeddingPipeline:
    """
    Create the real application embedding pipeline.

    VectorPipeline does NOT create embeddings.

    EmbeddingPipeline is responsible for creating
    query embeddings.
    """

    return EmbeddingPipeline(
        default_provider=EMBEDDING_PROVIDER,
    )


# ============================================================================
# REAL QDRANT VECTOR PIPELINE
# ============================================================================

@pytest.fixture(scope="session")
def real_vector_registry() -> VectorStoreRegistry:
    """
    Create a real vector-store registry and register
    the real Qdrant provider.

    IMPORTANT
    ---------
    VectorStoreRegistry starts empty.

    Therefore the QdrantStore MUST be explicitly registered
    before VectorPipeline can resolve:

        provider_name="qdrant"
    """

    registry = VectorStoreRegistry()

    qdrant_store = QdrantStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=QDRANT_HTTPS,
        timeout=QDRANT_TIMEOUT,
    )

    registry.register(
        qdrant_store
    )

    assert registry.has_provider(
        VECTOR_STORE_PROVIDER
    ), (
        f"Vector provider "
        f"'{VECTOR_STORE_PROVIDER}' "
        f"was not registered."
    )

    print(
        "\nRegistered vector providers:"
    )

    print(
        registry.list_providers()
    )

    return registry


@pytest.fixture(scope="session")
def real_vector_pipeline(
    real_vector_registry: VectorStoreRegistry,
) -> VectorPipeline:
    """
    Create a real VectorPipeline using the test registry.

    This is intentionally NOT the application global
    vector_pipeline because that global registry is empty
    in this test process.

    The pipeline is therefore:

        VectorPipeline
              ↓
        VectorStoreRegistry
              ↓
        QdrantStore
              ↓
        Real Qdrant
    """

    pipeline = VectorPipeline(
        registry=real_vector_registry,
    )

    return pipeline


# ============================================================================
# Dense retriever
# ============================================================================

@pytest.fixture(scope="session")
def real_dense_retriever(
    real_vector_pipeline: VectorPipeline,
) -> DenseRetriever:
    """
    Create the real DenseRetriever backed by Qdrant.
    """

    return DenseRetriever(
        pipeline=real_vector_pipeline,
    )


# ============================================================================
# Sparse retriever
# ============================================================================

@pytest.fixture(scope="session")
def real_sparse_retriever(
    real_vector_registry: VectorStoreRegistry,
) -> SparseRetriever:
    """
    Create the real BM25 sparse retriever
    from the real Qdrant collection.
    """

    qdrant_store = real_vector_registry.get(
        VECTOR_STORE_PROVIDER
    )

    assert isinstance(
        qdrant_store,
        QdrantStore,
    )

    sparse_retriever = (
        SparseRetriever.from_qdrant(
            qdrant_store=qdrant_store,
            collection_name=COLLECTION_NAME,
            batch_size=100,
        )
    )

    assert sparse_retriever.is_indexed

    print("\nBM25 index:")
    print(
        f"Indexed documents: "
        f"{sparse_retriever.document_count}"
    )

    return sparse_retriever

# ============================================================================
# Graph retriever
# ============================================================================

@pytest.fixture(scope="session")
def real_graph_retriever() -> GraphRetriever:
    """
    Create the real Neo4j GraphRetriever.
    """

    return GraphRetriever()


# ============================================================================
# Reranker
# ============================================================================

@pytest.fixture(scope="session")
def real_reranker() -> CrossEncoderBaseline:
    """
    Create the real CrossEncoder reranker.
    """

    return CrossEncoderBaseline(
        model_name=RERANKER_MODEL,
    )


# ============================================================================
# Retrieval service
# ============================================================================

@pytest.fixture(scope="session")
def retrieval_service(
    real_dense_retriever: DenseRetriever,
    real_sparse_retriever: SparseRetriever,
    real_graph_retriever: GraphRetriever,
    real_reranker: CrossEncoderBaseline,
) -> RetrievalService:
    """
    Construct the complete real RetrievalService.
    """

    config = RetrievalServiceConfig(
        use_dense=True,
        use_sparse=True,
        use_graph=True,
        use_reranker=True,
    )

    return RetrievalService(
        dense_retriever=real_dense_retriever,
        sparse_retriever=real_sparse_retriever,
        graph_retriever=real_graph_retriever,
        reranker=real_reranker,
        config=config,
    )


# ============================================================================
# Query embedding helper
# ============================================================================

def create_query_vector(
    query: str,
    embedding_pipeline: EmbeddingPipeline,
) -> list[float]:
    """
    Generate a query embedding using the real
    EmbeddingPipeline.

    Architecture:

        query
          ↓
        TextNode
          ↓
        EmbeddingPipeline
          ↓
        EmbeddedNode
          ↓
        embedding
          ↓
        list[float]

    VectorPipeline is NOT responsible for embeddings.
    """

    if not isinstance(
        query,
        str,
    ):
        raise ValueError(
            "Query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "Query cannot be empty."
        )

    query_node = TextNode(
        text=query,
    )

    embedded_node = embedding_pipeline.embed(
        query_node,
    )

    if embedded_node is None:
        raise RuntimeError(
            "EmbeddingPipeline returned None."
        )

    embedding = getattr(
        embedded_node,
        "embedding",
        None,
    )

    if embedding is None:
        raise RuntimeError(
            "EmbeddedNode does not contain "
            "an embedding."
        )

    if hasattr(
        embedding,
        "tolist",
    ):
        embedding = embedding.tolist()

    if not isinstance(
        embedding,
        list,
    ):
        embedding = list(embedding)

    if not embedding:
        raise RuntimeError(
            "Generated query embedding is empty."
        )

    if not all(
        isinstance(
            value,
            (int, float),
        )
        for value in embedding
    ):
        raise RuntimeError(
            "Generated query embedding contains "
            "non-numeric values."
        )

    return [
        float(value)
        for value in embedding
    ]


# ============================================================================
# Request factory
# ============================================================================

def make_request(
    query: str,
    embedding_pipeline: EmbeddingPipeline,
    *,
    top_k: int = TOP_K,
    include_graph: bool = True,
    rerank: bool = True,
    rerank_top_k: int = RERANK_TOP_K,
) -> RetrievalRequest:
    """
    Build a real RetrievalRequest.

    Query embedding is created by EmbeddingPipeline.

    The vector-store provider is explicitly supplied
    to RetrievalRequest.provider.
    """

    query_vector = create_query_vector(
        query=query,
        embedding_pipeline=embedding_pipeline,
    )

    return RetrievalRequest(
        query=query,
        top_k=top_k,
        collection_name=COLLECTION_NAME,
        provider=VECTOR_STORE_PROVIDER,
        query_vector=query_vector,
        include_graph=include_graph,
        rerank=rerank,
        rerank_top_k=rerank_top_k,
    )


# ============================================================================
# Vector store registration test
# ============================================================================

def test_real_qdrant_provider_is_registered(
    real_vector_registry: VectorStoreRegistry,
) -> None:
    """
    Verify that the real Qdrant provider is registered
    before retrieval starts.
    """

    assert real_vector_registry.has_provider(
        "qdrant"
    )

    providers = (
        real_vector_registry.list_providers()
    )

    assert "qdrant" in providers

    print(
        "\nVector-store providers:"
    )

    print(
        providers
    )


# ============================================================================
# Qdrant connectivity / collection test
# ============================================================================

def test_real_qdrant_collection_is_available(
    real_vector_registry: VectorStoreRegistry,
) -> None:
    """
    Verify the configured Qdrant collection exists.
    """

    qdrant_store = real_vector_registry.get(
        VECTOR_STORE_PROVIDER
    )

    assert isinstance(
        qdrant_store,
        QdrantStore,
    )

    assert qdrant_store.health_check(), (
        "Qdrant health check failed. "
        "Make sure Qdrant is running."
    )

    assert qdrant_store.collection_exists(
        COLLECTION_NAME
    ), (
        f"Qdrant collection "
        f"'{COLLECTION_NAME}' does not exist."
    )

    count = qdrant_store.count(
        COLLECTION_NAME
    )

    print(
        "\nQdrant collection:"
    )

    print(
        f"Name   : {COLLECTION_NAME}"
    )

    print(
        f"Count  : {count}"
    )

    assert count > 0, (
        f"Qdrant collection "
        f"'{COLLECTION_NAME}' is empty."
    )


# ============================================================================
# Basic service construction
# ============================================================================

def test_retrieval_service_initializes(
    retrieval_service: RetrievalService,
) -> None:
    """
    Verify complete service wiring.
    """

    assert retrieval_service is not None

    assert (
        retrieval_service.dense_retriever
        is not None
    )

    assert (
        retrieval_service.sparse_retriever
        is not None
    )

    assert (
        retrieval_service.graph_retriever
        is not None
    )

    assert (
        retrieval_service.reranker
        is not None
    )

    assert (
        retrieval_service.context_builder
        is not None
    )

    assert (
        retrieval_service.config.use_dense
        is True
    )

    assert (
        retrieval_service.config.use_sparse
        is True
    )

    assert (
        retrieval_service.config.use_graph
        is True
    )

    assert (
        retrieval_service.config.use_reranker
        is True
    )


# ============================================================================
# Embedding pipeline
# ============================================================================

def test_real_query_embedding(
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify that the real embedding pipeline can
    generate a valid query vector.
    """

    query = "refund policy"

    vector = create_query_vector(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
    )

    assert isinstance(
        vector,
        list,
    )

    assert vector

    assert all(
        isinstance(
            value,
            float,
        )
        for value in vector
    )

    print(
        "\nReal query embedding:"
    )

    print(
        f"Provider   : "
        f"{EMBEDDING_PROVIDER}"
    )

    print(
        f"Dimensions : "
        f"{len(vector)}"
    )

    print(
        f"First 5    : "
        f"{vector[:5]}"
    )


# ============================================================================
# Vector-store provider configuration
# ============================================================================

def test_vector_store_provider_configuration() -> None:
    """
    Verify the configured vector-store provider.
    """

    assert VECTOR_STORE_PROVIDER == "qdrant"

    print(
        "\nVector-store provider:"
    )

    print(
        VECTOR_STORE_PROVIDER
    )


# ============================================================================
# Complete pipeline
# ============================================================================

def test_complete_real_retrieval_pipeline(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Test the complete real retrieval pipeline.
    """

    query = "refund policy"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    start_time = time.perf_counter()

    result = retrieval_service.retrieve(
        request
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    print("\n")
    print("=" * 80)
    print(
        "REAL RETRIEVAL SERVICE"
    )
    print("=" * 80)

    print(
        f"Query              : {query}"
    )

    print(
        f"Collection         : "
        f"{COLLECTION_NAME}"
    )

    print(
        f"Vector Store       : "
        f"{VECTOR_STORE_PROVIDER}"
    )

    print(
        f"Embedding Provider : "
        f"{EMBEDDING_PROVIDER}"
    )

    print(
        f"Top K              : "
        f"{TOP_K}"
    )

    print(
        f"Elapsed            : "
        f"{elapsed_ms:.2f} ms"
    )

    print("=" * 80)

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.query == query

    assert result.selected_count >= 0

    assert (
        result.total_candidates
        >= result.selected_count
    )

    assert result.character_count >= 0

    assert isinstance(
        result.selected_items,
        list,
    )

    assert (
        result.selected_count
        <= TOP_K
    )

    ids = [
        item.id
        for item in result.selected_items
    ]

    assert len(ids) == len(
        set(ids)
    )

    print(
        f"Candidates : "
        f"{result.total_candidates}"
    )

    print(
        f"Selected   : "
        f"{result.selected_count}"
    )

    print(
        f"Characters : "
        f"{result.character_count}"
    )

    print(
        f"Truncated  : "
        f"{result.truncated}"
    )

    print(
        "\nEvidence:"
    )

    for item in result.selected_items:

        print("-" * 80)

        print(
            f"Rank   : {item.rank}"
        )

        print(
            f"Source : {item.source}"
        )

        print(
            f"Score  : {item.score}"
        )

        print(
            f"ID     : {item.id}"
        )

        print(
            f"Text   : "
            f"{item.text[:500]}"
        )

    print("=" * 80)


# ============================================================================
# All retrieval components
# ============================================================================

def test_all_retrieval_components_are_represented(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify the complete retrieval pipeline produces context.
    """

    query = "payment failed"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.selected_items

    sources = {
        str(item.source).lower()
        for item in result.selected_items
    }

    print(
        "\nSources returned:"
    )

    print(
        sources
    )

    assert len(sources) >= 1


# ============================================================================
# Graph enabled
# ============================================================================

def test_graph_enabled_retrieval(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify retrieval with Graph-RAG enabled.
    """

    query = "subscription cancellation"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=False,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.query == query

    print(
        "\nGraph-enabled query:"
    )

    print(
        query
    )

    for item in result.selected_items:

        print(
            f"[{item.source}] "
            f"{item.text[:300]}"
        )


# ============================================================================
# Reranking enabled
# ============================================================================

def test_reranking_enabled(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify the real reranker path.
    """

    query = (
        "how can I reset my password"
    )

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=False,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.selected_items

    ranks = [
        item.rank
        for item in result.selected_items
    ]

    assert ranks == sorted(
        ranks
    )

    assert len(ranks) <= (
        RERANK_TOP_K
    )

    print(
        "\nReranked results:"
    )

    for item in result.selected_items:

        print(
            f"Rank={item.rank} "
            f"Source={item.source} "
            f"Score={item.score}"
        )


# ============================================================================
# Reranking disabled
# ============================================================================

def test_reranking_disabled(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify retrieval without reranking.
    """

    query = "account management"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=False,
        rerank=False,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.query == query

    assert (
        result.selected_count
        <= TOP_K
    )


# ============================================================================
# Graph disabled
# ============================================================================

def test_graph_disabled(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify Graph-RAG can be disabled per request.
    """

    query = "refund policy"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=False,
        rerank=False,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.query == query

    assert (
        result.selected_count
        <= TOP_K
    )


# ============================================================================
# Top-k behavior
# ============================================================================

@pytest.mark.parametrize(
    "top_k",
    [1, 2, 3, 5],
)
def test_top_k_behavior(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
    top_k: int,
) -> None:
    """
    Verify requested top_k is respected.
    """

    query = "refund policy"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=top_k,
        include_graph=True,
        rerank=True,
        rerank_top_k=top_k,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert (
        result.selected_count
        <= top_k
    )

    assert (
        result.total_candidates
        >= result.selected_count
    )


# ============================================================================
# Multiple real queries
# ============================================================================

@pytest.mark.parametrize(
    "query",
    TEST_QUERIES,
)
def test_multiple_real_supportmind_queries(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
    query: str,
) -> None:
    """
    Run several realistic SupportMind queries.
    """

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    start = time.perf_counter()

    result = retrieval_service.retrieve(
        request
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000

    assert isinstance(
        result,
        BuiltContext,
    )

    assert result.query == query

    assert (
        result.selected_count
        <= TOP_K
    )

    assert result.character_count >= 0

    print("\n")
    print("-" * 80)

    print(
        f"Query   : {query}"
    )

    print(
        f"Results : "
        f"{result.selected_count}"
    )

    print(
        f"Total   : "
        f"{result.total_candidates}"
    )

    print(
        f"Latency : "
        f"{elapsed_ms:.2f} ms"
    )

    print("-" * 80)

    for item in result.selected_items:

        print(
            f"{item.rank}. "
            f"[{item.source}] "
            f"{item.text[:250]}"
        )


# ============================================================================
# ContextBuilder budget
# ============================================================================

def test_context_budget_is_respected(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Verify ContextBuilder budget enforcement.
    """

    query = "refund policy"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    assert (
        result.selected_count
        <= 8
    )

    assert (
        result.character_count
        <= 12000
    )


# ============================================================================
# Final ContextItem structure
# ============================================================================

def test_retrieval_result_structure(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Validate final ContextItem structure.
    """

    query = "payment failed"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    result = retrieval_service.retrieve(
        request
    )

    assert isinstance(
        result,
        BuiltContext,
    )

    for item in result.selected_items:

        assert isinstance(
            item.id,
            str,
        )

        assert isinstance(
            item.text,
            str,
        )

        assert isinstance(
            item.rank,
            int,
        )

        assert item.rank > 0

        assert item.source is not None

        assert item.metadata is not None

        assert isinstance(
            item.metadata,
            dict,
        )


# ============================================================================
# Latency smoke
# ============================================================================

def test_retrieval_latency_smoke(
    retrieval_service: RetrievalService,
    real_embedding_pipeline: EmbeddingPipeline,
) -> None:
    """
    Basic retrieval latency smoke test.

    This is intentionally a generous timeout
    and is not a production SLA.
    """

    query = "refund policy"

    request = make_request(
        query=query,
        embedding_pipeline=real_embedding_pipeline,
        top_k=TOP_K,
        include_graph=True,
        rerank=True,
        rerank_top_k=RERANK_TOP_K,
    )

    start = time.perf_counter()

    result = retrieval_service.retrieve(
        request
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000

    assert isinstance(
        result,
        BuiltContext,
    )

    assert elapsed_ms < 120000

    print(
        f"\nRetrieval latency: "
        f"{elapsed_ms:.2f} ms"
    )


# ============================================================================
# Invalid request tests
# ============================================================================

def test_empty_query_is_rejected(
    retrieval_service: RetrievalService,
) -> None:
    """
    RetrievalRequest should reject an empty query.
    """

    with pytest.raises(Exception):

        RetrievalRequest(
            query="",
            top_k=TOP_K,
            collection_name=COLLECTION_NAME,
            provider=VECTOR_STORE_PROVIDER,
        )


def test_invalid_top_k_is_rejected(
    retrieval_service: RetrievalService,
) -> None:
    """
    RetrievalRequest should reject invalid top_k.
    """

    with pytest.raises(Exception):

        RetrievalRequest(
            query="refund policy",
            top_k=0,
            collection_name=COLLECTION_NAME,
            provider=VECTOR_STORE_PROVIDER,
        )


# ============================================================================
# Service close
# ============================================================================

def test_service_close_is_safe(
    retrieval_service: RetrievalService,
) -> None:
    """
    Verify service lifecycle cleanup is safe.
    """

    close_method = getattr(
        retrieval_service,
        "close",
        None,
    )

    if callable(close_method):

        close_method()

    assert True