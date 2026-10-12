"""
SupportMind Retrieval Evaluation Benchmark.

Evaluates the real retrieval pipeline against the
golden evaluation dataset.

Configurations
--------------
1. Dense
2. BM25
3. Hybrid (Dense + BM25)
4. Full Retrieval (Dense + BM25 + Graph + Reranker)

Metrics
-------
- Recall@K
- Precision@K
- MRR@K
- Latency

Important
---------
The current golden dataset uses document-level ground truth.

Therefore a retrieved result is considered relevant when
its source/document matches one of the expected sources
in the evaluation dataset.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from llama_index.core.schema import TextNode


# ============================================================================
# Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DATASET = (
    PROJECT_ROOT
    / "rawsupportminddata"
    / "evaluvation"
    / "supportmind_retrieval_evaluation_dataset_v1.json"
)


# ============================================================================
# Environment
# ============================================================================

ENV_CANDIDATES = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / "backend" / ".env",
)

for env_file in ENV_CANDIDATES:
    if env_file.exists():
        load_dotenv(
            env_file,
            override=False,
        )


# ============================================================================
# Application imports
# ============================================================================

from app.embeddings.embedding_pipeline import (
    EmbeddingPipeline,
)

from app.retrieval.context.context_models import (
    BuiltContext,
)

from app.retrieval.dense.dense_retriever import (
    DenseRetriever,
)

from app.retrieval.graph_rag.graph_retriever import (
    GraphRetriever,
)

from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
)

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

from app.vector_store.qdrant_store import (
    QdrantStore,
)

from app.vector_store.vector_pipeline import (
    VectorPipeline,
)

from app.vector_store.vector_registry import (
    VectorStoreRegistry,
)


# ============================================================================
# Configuration
# ============================================================================

COLLECTION_NAME = os.getenv(
    "SUPPORTMIND_QDRANT_COLLECTION",
    "supportmind_retrieval_pdf_test",
)

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

VECTOR_STORE_PROVIDER = os.getenv(
    "SUPPORTMIND_VECTOR_STORE_PROVIDER",
    "qdrant",
)

EMBEDDING_PROVIDER = os.getenv(
    "SUPPORTMIND_EMBEDDING_PROVIDER",
    "fastembed",
)

RERANKER_MODEL = os.getenv(
    "SUPPORTMIND_RERANKER_MODEL",
    CrossEncoderBaseline.DEFAULT_MODEL_NAME,
)

TOP_K = 5


# ============================================================================
# Evaluation dataset
# ============================================================================

def load_evaluation_dataset() -> dict[str, Any]:
    """
    Load the golden retrieval evaluation dataset.
    """

    if not EVALUATION_DATASET.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n"
            f"{EVALUATION_DATASET}"
        )

    with EVALUATION_DATASET.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError(
            "Evaluation dataset must contain a JSON object."
        )

    queries = dataset.get("queries")

    if not isinstance(queries, list):
        raise ValueError(
            "Evaluation dataset must contain a 'queries' list."
        )

    if not queries:
        raise ValueError(
            "Evaluation dataset contains no queries."
        )

    return dataset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def evaluation_dataset() -> dict[str, Any]:
    """
    Load the golden evaluation dataset once.
    """

    dataset = load_evaluation_dataset()

    print("\n")
    print("=" * 80)
    print("SUPPORTMIND RETRIEVAL EVALUATION")
    print("=" * 80)

    print(
        f"Dataset    : {EVALUATION_DATASET}"
    )

    print(
        f"Queries    : "
        f"{len(dataset['queries'])}"
    )

    print(
        f"Top K      : "
        f"{TOP_K}"
    )

    print("=" * 80)

    return dataset


@pytest.fixture(scope="session")
def embedding_pipeline() -> EmbeddingPipeline:
    """
    Real query embedding pipeline.
    """

    return EmbeddingPipeline(
        default_provider=EMBEDDING_PROVIDER,
    )


@pytest.fixture(scope="session")
def vector_registry() -> VectorStoreRegistry:
    """
    Real Qdrant vector-store registry.
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
    )

    return registry


@pytest.fixture(scope="session")
def vector_pipeline(
    vector_registry: VectorStoreRegistry,
) -> VectorPipeline:
    """
    Real VectorPipeline backed by Qdrant.
    """

    return VectorPipeline(
        registry=vector_registry,
    )


@pytest.fixture(scope="session")
def dense_retriever(
    vector_pipeline: VectorPipeline,
) -> DenseRetriever:
    """
    Real dense retriever.
    """

    return DenseRetriever(
        pipeline=vector_pipeline,
    )


@pytest.fixture(scope="session")
def sparse_retriever(
    vector_registry: VectorStoreRegistry,
) -> SparseRetriever:
    """
    Build real BM25 index from Qdrant documents.
    """

    qdrant_store = vector_registry.get(
        VECTOR_STORE_PROVIDER
    )

    assert isinstance(
        qdrant_store,
        QdrantStore,
    )

    retriever = SparseRetriever.from_qdrant(
        qdrant_store=qdrant_store,
        collection_name=COLLECTION_NAME,
        batch_size=100,
    )

    assert retriever.is_indexed

    print("\nBM25 index:")
    print(
        f"Indexed documents: "
        f"{retriever.document_count}"
    )

    return retriever


@pytest.fixture(scope="session")
def graph_retriever() -> GraphRetriever:
    """
    Real Neo4j graph retriever.
    """

    return GraphRetriever()


@pytest.fixture(scope="session")
def reranker() -> CrossEncoderBaseline:
    """
    Real CrossEncoder reranker.
    """

    return CrossEncoderBaseline(
        model_name=RERANKER_MODEL,
    )


# ============================================================================
# Query embedding
# ============================================================================

def create_query_vector(
    query: str,
    embedding_pipeline: EmbeddingPipeline,
) -> list[float]:
    """
    Create a real query embedding.
    """

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

    embedding = getattr(
        embedded_node,
        "embedding",
        None,
    )

    if embedding is None:
        raise RuntimeError(
            "Query embedding was not generated."
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

    return [
        float(value)
        for value in embedding
    ]


# ============================================================================
# Retrieval service factory
# ============================================================================

def create_retrieval_service(
    *,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    graph_retriever: GraphRetriever,
    reranker: CrossEncoderBaseline,
    use_dense: bool,
    use_sparse: bool,
    use_graph: bool,
    use_reranker: bool,
) -> RetrievalService:
    """
    Create a RetrievalService with a specific
    retrieval configuration.
    """

    config = RetrievalServiceConfig(
        use_dense=use_dense,
        use_sparse=use_sparse,
        use_graph=use_graph,
        use_reranker=use_reranker,
    )

    return RetrievalService(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
        config=config,
    )


# ============================================================================
# Ground truth helpers
# ============================================================================

def expected_sources(
    evaluation_item: dict[str, Any],
) -> set[str]:
    """
    Return normalized expected source names.
    """

    sources = set()

    for item in evaluation_item.get(
        "relevant_sources",
        [],
    ):

        source = item.get(
            "source"
        )

        if isinstance(
            source,
            str,
        ):
            sources.add(
                source.strip().lower()
            )

    return sources


def normalize_source(
    source: Any,
) -> str:
    """
    Normalize a retrieved source/document name.
    """

    if source is None:
        return ""

    value = str(source).strip().lower()

    return value


def retrieved_source_names(
    result: BuiltContext,
) -> list[str]:
    """
    Extract source/document names from retrieved ContextItems.

    The retrieval pipeline's ContextItem.source describes
    the retrieval mechanism (dense, graph, etc.), so we first
    inspect metadata for the actual source document.
    """

    sources: list[str] = []

    for item in result.selected_items:

        metadata = item.metadata or {}

        possible_keys = (
            "source",
            "source_file",
            "file_name",
            "filename",
            "document_name",
            "document",
            "document_id",
        )

        found_source = None

        for key in possible_keys:

            value = metadata.get(key)

            if isinstance(
                value,
                str,
            ) and value.strip():

                found_source = value
                break

        if found_source is None:

            # Fall back to the retrieval source.
            found_source = str(
                item.source
            )

        sources.append(
            normalize_source(
                found_source
            )
        )

    return sources


# ============================================================================
# Metric calculation
# ============================================================================

def recall_at_k(
    retrieved_sources: list[str],
    expected: set[str],
) -> float:
    """
    Document-level Recall@K.

    Returns 1 when at least one expected source
    appears in the retrieved results.
    """

    if not expected:
        return 0.0

    retrieved = set(
        retrieved_sources
    )

    return (
        1.0
        if retrieved.intersection(expected)
        else 0.0
    )


def precision_at_k(
    retrieved_sources: list[str],
    expected: set[str],
) -> float:
    """
    Document-level Precision@K.
    """

    if not retrieved_sources:
        return 0.0

    relevant = sum(
        1
        for source in retrieved_sources
        if source in expected
    )

    return (
        relevant
        / len(retrieved_sources)
    )


def reciprocal_rank(
    retrieved_sources: list[str],
    expected: set[str],
) -> float:
    """
    Reciprocal rank of the first relevant result.
    """

    for rank, source in enumerate(
        retrieved_sources,
        start=1,
    ):

        if source in expected:

            return 1.0 / rank

    return 0.0

def test_inspect_qdrant_metadata(
    vector_registry: VectorStoreRegistry,
) -> None:

    qdrant_store = vector_registry.get(
        VECTOR_STORE_PROVIDER
    )

    assert isinstance(
        qdrant_store,
        QdrantStore,
    )

    client = qdrant_store.client

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    print("\n")
    print("=" * 80)
    print("QDRANT METADATA INSPECTION")
    print("=" * 80)

    for index, point in enumerate(
        points,
        start=1,
    ):

        print(f"\nPOINT {index}")
        print("-" * 80)
        print("ID:")
        print(point.id)

        print("\nPAYLOAD:")

        payload = point.payload or {}

        print(
            json.dumps(
                payload,
                indent=2,
                default=str,
            )
        )

    print("=" * 80)


# ============================================================================
# Benchmark runner
# ============================================================================

def run_benchmark(
    *,
    service: RetrievalService,
    dataset: dict[str, Any],
    embedding_pipeline: EmbeddingPipeline,
    include_graph: bool,
    rerank: bool,
) -> dict[str, Any]:
    """
    Run all golden queries against one retrieval configuration.
    """

    query_results = []

    recall_values = []
    precision_values = []
    reciprocal_ranks = []
    latencies = []

    for evaluation_item in dataset["queries"]:

        query = evaluation_item["query"]

        expected = expected_sources(
            evaluation_item
        )

        query_vector = create_query_vector(
            query=query,
            embedding_pipeline=embedding_pipeline,
        )

        request = RetrievalRequest(
            query=query,
            top_k=TOP_K,
            collection_name=COLLECTION_NAME,
            provider=VECTOR_STORE_PROVIDER,
            query_vector=query_vector,
            include_graph=include_graph,
            rerank=rerank,
            rerank_top_k=TOP_K if rerank else None,
        )

        start = time.perf_counter()

        result = service.retrieve(
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

        retrieved = retrieved_source_names(
            result
        )

        recall = recall_at_k(
            retrieved,
            expected,
        )

        precision = precision_at_k(
            retrieved,
            expected,
        )

        rr = reciprocal_rank(
            retrieved,
            expected,
        )

        recall_values.append(
            recall
        )

        precision_values.append(
            precision
        )

        reciprocal_ranks.append(
            rr
        )

        latencies.append(
            elapsed_ms
        )

        query_results.append(
            {
                "id": evaluation_item["id"],
                "query": query,
                "expected_sources": sorted(
                    expected
                ),
                "retrieved_sources": retrieved,
                "recall": recall,
                "precision": precision,
                "reciprocal_rank": rr,
                "latency_ms": elapsed_ms,
            }
        )

    count = len(
        query_results
    )

    return {
        "queries": count,
        "recall_at_5": (
            sum(recall_values)
            / count
        ),
        "precision_at_5": (
            sum(precision_values)
            / count
        ),
        "mrr_at_5": (
            sum(reciprocal_ranks)
            / count
        ),
        "average_latency_ms": (
            sum(latencies)
            / count
        ),
        "query_results": query_results,
    }


# ============================================================================
# Evaluation
# ============================================================================

def test_retrieval_evaluation(
    evaluation_dataset: dict[str, Any],
    embedding_pipeline: EmbeddingPipeline,
    dense_retriever: DenseRetriever,
    sparse_retriever: SparseRetriever,
    graph_retriever: GraphRetriever,
    reranker: CrossEncoderBaseline,
) -> None:
    """
    Compare Dense, BM25, Hybrid and Full Retrieval.
    """

    configurations = {
        "dense": {
            "use_dense": True,
            "use_sparse": False,
            "use_graph": False,
            "use_reranker": False,
            "include_graph": False,
            "rerank": False,
        },
        "bm25": {
            "use_dense": False,
            "use_sparse": True,
            "use_graph": False,
            "use_reranker": False,
            "include_graph": False,
            "rerank": False,
        },
        "hybrid": {
            "use_dense": True,
            "use_sparse": True,
            "use_graph": False,
            "use_reranker": False,
            "include_graph": False,
            "rerank": False,
        },
        "full": {
            "use_dense": True,
            "use_sparse": True,
            "use_graph": True,
            "use_reranker": True,
            "include_graph": True,
            "rerank": True,
        },
    }

    reports = {}

    for name, settings in configurations.items():

        print("\n")
        print("=" * 80)
        print(
            f"EVALUATING: {name.upper()}"
        )
        print("=" * 80)

        service = create_retrieval_service(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            graph_retriever=graph_retriever,
            reranker=reranker,
            use_dense=settings["use_dense"],
            use_sparse=settings["use_sparse"],
            use_graph=settings["use_graph"],
            use_reranker=settings["use_reranker"],
        )

        report = run_benchmark(
            service=service,
            dataset=evaluation_dataset,
            embedding_pipeline=embedding_pipeline,
            include_graph=settings["include_graph"],
            rerank=settings["rerank"],
        )

        reports[name] = report

        print(
            f"Queries        : "
            f"{report['queries']}"
        )

        print(
            f"Recall@5       : "
            f"{report['recall_at_5']:.4f}"
        )

        print(
            f"Precision@5    : "
            f"{report['precision_at_5']:.4f}"
        )

        print(
            f"MRR@5          : "
            f"{report['mrr_at_5']:.4f}"
        )

        print(
            f"Avg Latency    : "
            f"{report['average_latency_ms']:.2f} ms"
        )

    # ------------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RETRIEVAL EVALUATION COMPARISON")
    print("=" * 80)

    print(
        f"{'Configuration':<15}"
        f"{'Recall@5':>12}"
        f"{'Precision@5':>15}"
        f"{'MRR@5':>12}"
        f"{'Latency':>15}"
    )

    print("-" * 80)

    for name, report in reports.items():

        print(
            f"{name:<15}"
            f"{report['recall_at_5']:>12.4f}"
            f"{report['precision_at_5']:>15.4f}"
            f"{report['mrr_at_5']:>12.4f}"
            f"{report['average_latency_ms']:>12.2f} ms"
        )

    print("=" * 80)

    # ------------------------------------------------------------------------
    # Detailed full-pipeline failures
    # ------------------------------------------------------------------------

    full_report = reports["full"]

    failures = [
        item
        for item in full_report["query_results"]
        if item["reciprocal_rank"] == 0
    ]

    print("\n")
    print("=" * 80)
    print("FULL PIPELINE MISSED QUERIES")
    print("=" * 80)

    if not failures:

        print(
            "No queries were missed."
        )

    else:

        for item in failures:

            print("-" * 80)

            print(
                f"ID       : "
                f"{item['id']}"
            )

            print(
                f"Query    : "
                f"{item['query']}"
            )

            print(
                f"Expected : "
                f"{item['expected_sources']}"
            )

            print(
                f"Retrieved: "
                f"{item['retrieved_sources']}"
            )

    print("=" * 80)

    # ------------------------------------------------------------------------
    # Basic sanity checks
    # ------------------------------------------------------------------------

    for name, report in reports.items():

        assert 0.0 <= (
            report["recall_at_5"]
        ) <= 1.0

        assert 0.0 <= (
            report["precision_at_5"]
        ) <= 1.0

        assert 0.0 <= (
            report["mrr_at_5"]
        ) <= 1.0

        assert (
            report["average_latency_ms"]
            >= 0
        )