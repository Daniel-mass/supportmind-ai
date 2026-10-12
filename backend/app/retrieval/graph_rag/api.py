
"""
SupportMind AI - Graph-RAG API.

HTTP boundary for the existing Graph-RAG pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from llama_index.core.schema import TextNode
from pydantic import BaseModel, ConfigDict, Field, field_validator
from qdrant_client import QdrantClient

from app.embeddings.fastembed_embedder import FastEmbedder
from app.intelligence.graph.operational_graph import (
    create_neo4j_operational_graph,
)
from app.retrieval.graph_rag.evidence_fusion import EvidenceFusion
from app.retrieval.graph_rag.generator import GraphRAGGenerator
from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGRequest,
    GraphRAGResponse,
)
from app.retrieval.graph_rag.graph_rag_service import GraphRAGService
from app.retrieval.graph_rag.graph_retriever import (
    create_graph_retriever,
)
from app.retrieval.graph_rag.groundedness import GroundednessValidator
from app.retrieval.graph_rag.intelligence_adapter import (
    IntelligenceAdapter,
)
from app.retrieval.graph_rag.providers.ollama import (
    OllamaGenerationConfig,
    OllamaProvider,
)
from app.retrieval.reranker.cross_encoder_baseline import (
    CrossEncoderBaseline,
)
from app.retrieval.retrieval_service import (
    RetrievalService,
    RetrievalServiceConfig,
)
from app.retrieval.sparse.sparse_retriever import SparseRetriever
from app.retrieval.models.retrieval_models import RetrievalRequest

# NEW: DENSE VECTOR PIPELINE
from app.vector_store.qdrant_store import QdrantStore
from app.vector_store.vector_pipeline import VectorPipeline
from app.vector_store.vector_registry import VectorStoreRegistry


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_QDRANT_HOST = "127.0.0.1"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"

DEFAULT_TOP_K = 5
DEFAULT_MAX_EVIDENCE_ITEMS = 20
DEFAULT_MAX_EVIDENCE_CHARACTERS = 16000
DEFAULT_CONTEXT_MAX_ITEMS = 8
DEFAULT_CONTEXT_MAX_CHARACTERS = 12000

ROUTER_PREFIX = "/api/graph-rag"


# ============================================================================
# Qdrant adapter
# ============================================================================

class QdrantStoreAdapter:
    """
    Minimal adapter required by SparseRetriever.from_qdrant().

    The existing SparseRetriever expects an object exposing
    a .client attribute.
    """

    def __init__(self, client: QdrantClient) -> None:
        self.client = client


# ============================================================================
# Environment helpers
# ============================================================================

def _env_bool(
    name: str,
    default: bool,
) -> bool:
    """Read a boolean environment variable safely."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_int(
    name: str,
    default: int,
) -> int:
    """Read an integer environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {name!r} must be an integer."
        ) from exc


def _resolve_qdrant_url() -> str:
    """Build the Qdrant URL from existing project environment variables."""

    host = os.getenv(
        "QDRANT_HOST",
        DEFAULT_QDRANT_HOST,
    )

    port = _env_int(
        "QDRANT_PORT",
        DEFAULT_QDRANT_PORT,
    )

    https = _env_bool(
        "QDRANT_HTTPS",
        False,
    )

    scheme = "https" if https else "http"

    return f"{scheme}://{host}:{port}"


def _resolve_qdrant_collection(
    client: QdrantClient,
) -> str:
    """
    Resolve the SupportMind collection.

    Preferred order:
        1. SUPPORTMIND_QDRANT_COLLECTION
        2. QDRANT_COLLECTION_NAME
        3. Exactly one available collection
        4. Exactly one collection containing "supportmind"
    """

    configured_collection = (
        os.getenv("SUPPORTMIND_QDRANT_COLLECTION")
        or os.getenv("QDRANT_COLLECTION_NAME")
    )

    collections = {
        collection.name
        for collection in client.get_collections().collections
    }

    if not collections:
        raise RuntimeError(
            "Qdrant is reachable but contains no collections."
        )

    if configured_collection:
        if configured_collection not in collections:
            raise RuntimeError(
                "Configured Qdrant collection does not exist: "
                f"{configured_collection!r}. "
                f"Available collections: {sorted(collections)}"
            )

        return configured_collection

    if len(collections) == 1:
        return next(iter(collections))

    supportmind_candidates = [
        name
        for name in collections
        if "supportmind" in name.lower()
    ]

    if len(supportmind_candidates) == 1:
        return supportmind_candidates[0]

    raise RuntimeError(
        "Unable to resolve the SupportMind Qdrant collection "
        "automatically. Set either SUPPORTMIND_QDRANT_COLLECTION "
        "or QDRANT_COLLECTION_NAME. "
        f"Available collections: {sorted(collections)}"
    )


# ============================================================================
# Public API request
# ============================================================================

class GraphRAGAPIRequest(BaseModel):
    """
    HTTP request contract for Graph-RAG.

    This is deliberately separate from the internal GraphRAGRequest
    so the API contract remains stable if internal orchestration changes.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=50,
    )

    collection_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    query_vector: list[float] | None = None

    include_graph: bool = True
    rerank: bool = False

    rerank_top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
    )

    ticket_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    include_intelligence: bool = True
    include_workflow: bool = True

    max_evidence_items: int = Field(
        default=DEFAULT_MAX_EVIDENCE_ITEMS,
        ge=1,
        le=100,
    )

    max_evidence_characters: int = Field(
        default=DEFAULT_MAX_EVIDENCE_CHARACTERS,
        ge=1000,
        le=50000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("query")
    @classmethod
    def normalize_query(
        cls,
        value: str,
    ) -> str:
        """Trim whitespace and reject an empty query."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("query cannot be empty.")

        return normalized

    @field_validator("query_vector")
    @classmethod
    def validate_query_vector(
        cls,
        value: list[float] | None,
    ) -> list[float] | None:
        """Reject empty vectors and non-finite dimensions."""

        if value is None:
            return None

        if not value:
            raise ValueError("query_vector cannot be empty.")

        for dimension in value:
            if not isfinite(float(dimension)):
                raise ValueError(
                    "query_vector must contain only finite numbers."
                )

        return value


# ============================================================================
# Runtime
# ============================================================================

@dataclass
class GraphRAGRuntime:
    """
    Process-level Graph-RAG runtime.

    Holds the dependencies used by GraphRAGService and reuses one
    query embedder when dense retrieval is enabled.
    """

    service: GraphRAGService
    qdrant_client: QdrantClient
    retrieval_graph: Any
    operational_graph: Any
    ollama_provider: OllamaProvider
    collection_name: str
    use_dense: bool
    use_reranker: bool
    ollama_model: str
    query_embedder: FastEmbedder | None

    def health(self) -> dict[str, Any]:
        """
        Check infrastructure dependencies.

        Generation itself is not executed here because a health endpoint
        should not spend tokens on a model response.
        """

        qdrant_ok = False
        qdrant_collection_count = 0

        try:
            collections = self.qdrant_client.get_collections()

            qdrant_collection_count = len(
                collections.collections
            )

            qdrant_ok = (
                self.collection_name
                in {
                    item.name
                    for item in collections.collections
                }
            )

        except Exception:
            LOGGER.exception("Qdrant health check failed.")

        retrieval_graph_ok = False

        try:
            retrieval_graph_ok = bool(
                self.retrieval_graph.verify_connection()
            )
        except Exception:
            LOGGER.exception(
                "Existing retrieval graph health check failed."
            )

        try:
            intelligence_health = dict(
                self.operational_graph.health_check()
            )
        except Exception:
            LOGGER.exception(
                "Intelligence graph health check failed."
            )
            intelligence_health = {"healthy": False}

        intelligence_graph_ok = bool(
            intelligence_health.get("healthy", False)
        )

        overall_ok = (
            qdrant_ok
            and retrieval_graph_ok
            and intelligence_graph_ok
        )

        return {
            "status": "ok" if overall_ok else "degraded",
            "qdrant": {
                "healthy": qdrant_ok,
                "collection": self.collection_name,
                "collection_count": qdrant_collection_count,
            },
            "retrieval_graph": {
                "healthy": retrieval_graph_ok,
            },
            "intelligence_graph": intelligence_health,
            "generation": {
                "provider": "ollama",
                "model": self.ollama_model,
                "configured": True,
            },
            "embedding": {
                "provider": (
                    self.query_embedder.provider
                    if self.query_embedder is not None
                    else None
                ),
                "model": (
                    self.query_embedder.model_name
                    if self.query_embedder is not None
                    else None
                ),
                "configured": (
                    self.query_embedder is not None
                    if self.use_dense
                    else False
                ),
            },
            "pipeline": {
                "dense_enabled": self.use_dense,
                "sparse_enabled": True,
                "graph_enabled": True,
                "reranker_enabled": self.use_reranker,
            },
        }

    def close(self) -> None:
        """Close runtime resources when possible."""

        components = [
            self.service,
            self.retrieval_graph,
            self.operational_graph,
            self.qdrant_client,
        ]

        seen: set[int] = set()

        for component in components:
            if component is None:
                continue

            component_id = id(component)

            if component_id in seen:
                continue

            seen.add(component_id)

            close_method = getattr(
                component,
                "close",
                None,
            )

            if callable(close_method):
                try:
                    close_method()
                except Exception:
                    LOGGER.exception(
                        "Failed to close runtime component."
                    )


# ============================================================================
# Real Graph-RAG runtime factory
# ============================================================================

def create_graph_rag_runtime() -> GraphRAGRuntime:
    """
    Build the complete real Graph-RAG runtime.
    """

    qdrant_client: QdrantClient | None = None
    retrieval_graph = None
    operational_graph = None

    try:
        # ------------------------------------------------------------------
        # 1. Qdrant
        # ------------------------------------------------------------------

        qdrant_url = _resolve_qdrant_url()

        qdrant_api_key = (
            os.getenv("QDRANT_API_KEY") or None
        )

        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        collections = qdrant_client.get_collections()

        if not collections.collections:
            raise RuntimeError(
                "Qdrant is reachable but contains no collections."
            )

        collection_name = _resolve_qdrant_collection(
            qdrant_client
        )

        LOGGER.info(
            "Graph-RAG Qdrant collection: %s",
            collection_name,
        )

        # ------------------------------------------------------------------
        # 2. Sparse retrieval
        # ------------------------------------------------------------------

        sparse_retriever = SparseRetriever.from_qdrant(
            qdrant_store=QdrantStoreAdapter(qdrant_client),
            collection_name=collection_name,
            batch_size=100,
        )

        if not sparse_retriever.is_indexed:
            raise RuntimeError(
                "BM25 sparse retriever index was not created."
            )

        LOGGER.info(
            "Graph-RAG BM25 documents: %d",
            sparse_retriever.document_count,
        )

        # ------------------------------------------------------------------
        # 3. Existing SupportMind retrieval graph
        # ------------------------------------------------------------------

        retrieval_graph = create_graph_retriever()

        if not retrieval_graph.verify_connection():
            raise RuntimeError(
                "Existing SupportMind retrieval graph "
                "could not connect."
            )

        # ------------------------------------------------------------------
        # 4. Optional dense retrieval and query embedding
        # ------------------------------------------------------------------

        use_dense = _env_bool(
            "SUPPORTMIND_GRAPH_RAG_USE_DENSE",
            False,
        )

        dense_retriever = None
        query_embedder = None

        if use_dense:
            from app.retrieval.dense.dense_retriever import (
                DenseRetriever,
            )

            query_embedder = FastEmbedder(
                model_name=os.getenv(
                    "SUPPORTMIND_EMBEDDING_MODEL",
                    "BAAI/bge-base-en-v1.5",
                )
            )

            # --------------------------------------------------------------
            # NEW: DENSE VECTOR PIPELINE
            # --------------------------------------------------------------
            # Use an isolated registry so Graph-RAG does not depend on
            # the application's global vector registry configuration.

            graph_rag_registry = VectorStoreRegistry()

            # Reuse the Qdrant client already created above.
            graph_rag_qdrant_store = QdrantStore(
                client=qdrant_client,
            )

            # Configure a dedicated pipeline for this Graph-RAG runtime.
            graph_rag_pipeline = VectorPipeline(
                registry=graph_rag_registry,
            )

            # Register Qdrant and make it the default provider.
            graph_rag_pipeline.register_provider(
                graph_rag_qdrant_store,
                make_default=True,
            )

            # Inject the configured pipeline into DenseRetriever.
            dense_retriever = DenseRetriever(
                pipeline=graph_rag_pipeline,
            )

            LOGGER.info(
                "Dense retrieval enabled; query embedder: %s",
                query_embedder.model_name,
            )

            LOGGER.info(
                "Graph-RAG dense vector providers: %s; default: %s",
                graph_rag_pipeline.list_providers(),
                graph_rag_pipeline.default_provider,
            )

        # ------------------------------------------------------------------
        # 5. Optional reranker
        # ------------------------------------------------------------------

        use_reranker = _env_bool(
            "SUPPORTMIND_GRAPH_RAG_USE_RERANKER",
            False,
        )

        reranker = None

        if use_reranker:
            reranker = CrossEncoderBaseline()

        # ------------------------------------------------------------------
        # 6. Existing RetrievalService
        # ------------------------------------------------------------------

        retrieval_service = RetrievalService(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            graph_retriever=retrieval_graph,
            reranker=reranker,
            config=RetrievalServiceConfig(
                use_dense=use_dense,
                use_sparse=True,
                use_graph=True,
                use_reranker=use_reranker,
                context_max_items=DEFAULT_CONTEXT_MAX_ITEMS,
                context_max_characters=(
                    DEFAULT_CONTEXT_MAX_CHARACTERS
                ),
            ),
        )

        # ------------------------------------------------------------------
        # 7. Help Desk Intelligence Graph
        # ------------------------------------------------------------------

        operational_graph = create_neo4j_operational_graph()

        intelligence_health = operational_graph.health_check()

        if not intelligence_health.get("healthy", False):
            raise RuntimeError(
                "Help Desk Intelligence graph is unhealthy: "
                f"{intelligence_health}"
            )

        intelligence_adapter = IntelligenceAdapter(
            operational_graph
        )

        # ------------------------------------------------------------------
        # 8. Ollama
        # ------------------------------------------------------------------

        ollama_host = os.getenv(
            "OLLAMA_HOST",
            DEFAULT_OLLAMA_HOST,
        )

        ollama_model = os.getenv(
            "SUPPORTMIND_GRAPH_RAG_MODEL",
            DEFAULT_OLLAMA_MODEL,
        )

        ollama_provider = OllamaProvider(
            config=OllamaGenerationConfig(
                host=ollama_host,
                model_name=ollama_model,
                temperature=0.1,
                max_tokens=1024,
                timeout=180.0,
            )
        )

        # ------------------------------------------------------------------
        # 9. Generator
        # ------------------------------------------------------------------

        generator = GraphRAGGenerator(
            provider=ollama_provider,
            provider_name="ollama",
            model_name=ollama_model,
        )

        # ------------------------------------------------------------------
        # 10. Evidence fusion
        # ------------------------------------------------------------------

        evidence_fusion = EvidenceFusion()

        # ------------------------------------------------------------------
        # 11. Grounding
        # ------------------------------------------------------------------

        groundedness_validator = GroundednessValidator()

        # ------------------------------------------------------------------
        # 12. Complete Graph-RAG orchestration
        # ------------------------------------------------------------------

        service = GraphRAGService(
            retrieval_service=retrieval_service,
            intelligence_adapter=intelligence_adapter,
            evidence_fusion=evidence_fusion,
            generator=generator,
            groundedness_validator=groundedness_validator,
        )

        return GraphRAGRuntime(
            service=service,
            qdrant_client=qdrant_client,
            retrieval_graph=retrieval_graph,
            operational_graph=operational_graph,
            ollama_provider=ollama_provider,
            collection_name=collection_name,
            use_dense=use_dense,
            use_reranker=use_reranker,
            ollama_model=ollama_model,
            query_embedder=query_embedder,
        )

    except Exception:
        LOGGER.exception(
            "Failed to construct Graph-RAG runtime."
        )

        if operational_graph is not None:
            try:
                operational_graph.close()
            except Exception:
                LOGGER.exception(
                    "Failed to close operational graph "
                    "after startup failure."
                )

        if retrieval_graph is not None:
            try:
                retrieval_graph.close()
            except Exception:
                LOGGER.exception(
                    "Failed to close retrieval graph "
                    "after startup failure."
                )

        if qdrant_client is not None:
            try:
                qdrant_client.close()
            except Exception:
                LOGGER.exception(
                    "Failed to close Qdrant after startup failure."
                )

        raise


# ============================================================================
# Cached runtime dependency
# ============================================================================

@lru_cache(maxsize=1)
def get_graph_rag_runtime() -> GraphRAGRuntime:
    """Build the Graph-RAG runtime once per application process."""

    return create_graph_rag_runtime()


def clear_graph_rag_runtime_cache() -> None:
    """Close and clear the cached runtime for controlled shutdown tests."""

    try:
        runtime = get_graph_rag_runtime()
    except Exception:
        get_graph_rag_runtime.cache_clear()
        return

    try:
        runtime.close()
    finally:
        get_graph_rag_runtime.cache_clear()


# ============================================================================
# Router
# ============================================================================

router = APIRouter(
    prefix=ROUTER_PREFIX,
    tags=["Graph-RAG"],
)


@router.get("/health")
def graph_rag_health(
    runtime: GraphRAGRuntime = Depends(
        get_graph_rag_runtime
    ),
) -> dict[str, Any]:
    """Check Graph-RAG infrastructure health."""

    try:
        return runtime.health()

    except Exception as exc:
        LOGGER.exception(
            "Graph-RAG health check failed."
        )

        raise HTTPException(
            status_code=503,
            detail="Graph-RAG health check failed.",
        ) from exc


@router.post(
    "/query",
    response_model=GraphRAGResponse,
)
def graph_rag_query(
    payload: GraphRAGAPIRequest,
    runtime: GraphRAGRuntime = Depends(
        get_graph_rag_runtime
    ),
) -> GraphRAGResponse:
    """Execute the complete Graph-RAG pipeline."""

    # ------------------------------------------------------------------
    # 1. Validate reranker configuration
    # ------------------------------------------------------------------

    if payload.rerank and not runtime.use_reranker:
        raise HTTPException(
            status_code=400,
            detail=(
                "Reranking is disabled in the Graph-RAG runtime."
            ),
        )

    # ------------------------------------------------------------------
    # 2. Validate the collection
    # ------------------------------------------------------------------

    collection_name = (
        payload.collection_name
        or runtime.collection_name
    )

    if collection_name != runtime.collection_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "The requested collection does not match "
                "the configured Graph-RAG collection."
            ),
        )

    # ------------------------------------------------------------------
    # 3. Generate the query embedding when needed
    # ------------------------------------------------------------------

    query_vector = payload.query_vector

    if runtime.use_dense and query_vector is None:
        if runtime.query_embedder is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Dense retrieval is enabled, but the "
                    "query embedder is unavailable."
                ),
            )

        try:
            embedded_query = runtime.query_embedder.embed(
                TextNode(text=payload.query)
            )

            query_vector = embedded_query.embedding

            LOGGER.debug(
                "Generated query embedding with %d dimensions.",
                len(query_vector),
            )

        except Exception as exc:
            LOGGER.exception(
                "Failed to generate the Graph-RAG query embedding."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Failed to generate the query embedding."
                ),
            ) from exc

    # ------------------------------------------------------------------
    # 4. Build the existing retrieval request
    # ------------------------------------------------------------------

    retrieval_request = RetrievalRequest(
        query=payload.query,
        top_k=payload.top_k,
        collection_name=collection_name,
        query_vector=query_vector,
        include_graph=payload.include_graph,
        rerank=payload.rerank,
        rerank_top_k=payload.rerank_top_k,
        metadata=payload.metadata,
    )

    # ------------------------------------------------------------------
    # 5. Build the Graph-RAG orchestration request
    # ------------------------------------------------------------------

    graph_rag_request = GraphRAGRequest(
        query=payload.query,
        retrieval_request=retrieval_request,
        ticket_id=payload.ticket_id,
        include_intelligence=payload.include_intelligence,
        include_workflow=payload.include_workflow,
        max_evidence_items=payload.max_evidence_items,
        max_evidence_characters=(
            payload.max_evidence_characters
        ),
        metadata=payload.metadata,
    )

    # ------------------------------------------------------------------
    # 6. Execute the existing service
    # ------------------------------------------------------------------

    try:
        response = runtime.service.answer(
            graph_rag_request
        )

        return response

    except ValueError as exc:
        LOGGER.warning(
            "Invalid Graph-RAG request: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        LOGGER.exception(
            "Graph-RAG runtime failure."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Graph-RAG dependency or generation service failed."
            ),
        ) from exc

    except Exception as exc:
        LOGGER.exception(
            "Unexpected Graph-RAG API failure."
        )

        raise HTTPException(
            status_code=500,
            detail="Graph-RAG request failed.",
        ) from exc


# ============================================================================
# Optional standalone FastAPI application
# ============================================================================

def create_graph_rag_app() -> FastAPI:
    """
    Create a standalone FastAPI application.

    The production application's main.py can instead include this router.
    """

    app = FastAPI(
        title="SupportMind Graph-RAG API",
        version="1.0.0",
        description=(
            "Grounded Graph-RAG API for SupportMind AI."
        ),
    )

    app.include_router(router)

    return app
