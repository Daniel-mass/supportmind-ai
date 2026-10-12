"""
retrieval_models.py

Provider-agnostic contracts for the SupportMind AI retrieval layer.

These models are shared by:

    Query Router
        ↓
    Dense Retriever
        ↓
    Sparse Retriever
        ↓
    Hybrid / RRF
        ↓
    Graph-RAG
        ↓
    Rerankers
        ↓
    Context Builder
        ↓
    Retrieval Service

The models intentionally contain no provider-specific logic.

SupportMind AI
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ==============================================================
# Retrieval Source
# ==============================================================


class RetrievalSource(str, Enum):
    """
    Identifies the retrieval mechanism that produced a result.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH = "graph"
    RERANKED = "reranked"


# ==============================================================
# Retrieval Request
# ==============================================================


class RetrievalRequest(BaseModel):
    """
    Standard request passed into the retrieval layer.

    The request deliberately contains only information that
    retrieval components need.

    Query embedding generation happens outside this contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Original user retrieval query.",
    )

    top_k: int = Field(
        default=5,
        gt=0,
        description="Maximum number of results requested.",
    )

    collection_name: str = Field(
        ...,
        min_length=1,
        description="Qdrant collection used for vector retrieval.",
    )
    provider: str | None = Field(
      default=None,
      min_length=1,
      description=(
        "Optional vector-store provider override. "
        "When omitted, VectorPipeline uses its configured default provider."),
)
    query_vector: list[float] | None = Field(
        default=None,
        description=(
            "Optional query embedding. Required by dense/hybrid "
            "retrieval when vector search is performed."
        ),
    )

    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-neutral metadata filters.",
    )

    include_graph: bool = Field(
        default=False,
        description="Whether Graph-RAG retrieval should be included.",
    )

    rerank: bool = Field(
        default=False,
        description="Whether retrieved candidates should be reranked.",
    )

    rerank_top_k: int | None = Field(
        default=None,
        gt=0,
        description="Optional number of candidates retained after reranking.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata.",
    )


# ==============================================================
# Retrieval Result
# ==============================================================


class RetrievalResult(BaseModel):
    """
    Standard representation of one retrieved candidate.

    Every retrieval mechanism must convert its native result
    into this schema.

    This is intentionally independent of Qdrant, BM25, Neo4j,
    ColBERT, or Cross-Encoder implementations.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(
        ...,
        min_length=1,
        description="Unique result identifier.",
    )

    node_id: str = Field(
        ...,
        min_length=1,
        description="Original document/chunk node identifier.",
    )

    document_id: str = Field(
        ...,
        min_length=1,
        description="Parent document identifier.",
    )

    text: str = Field(
        ...,
        description="Retrieved chunk text.",
    )

    score: float = Field(
        ...,
        description="Primary retrieval score.",
    )

    rank: int | None = Field(
        default=None,
        gt=0,
        description="Rank assigned by the producing retriever.",
    )

    source: RetrievalSource = Field(
        ...,
        description="Retrieval mechanism producing this result.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk/document metadata.",
    )

    retrieval_scores: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Additional scores such as dense_score, bm25_score, "
            "rrf_score, cross_encoder_score, or colbert_score."
        ),
    )

    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC retrieval timestamp.",
    )


# ==============================================================
# Retrieval Response
# ==============================================================


class RetrievalResponse(BaseModel):
    """
    Standard response returned by the retrieval layer.

    Contains the final ordered candidates plus lightweight
    execution metadata.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Original retrieval query.",
    )

    results: list[RetrievalResult] = Field(
        default_factory=list,
        description="Ordered retrieval results.",
    )

    total_results: int = Field(
        default=0,
        ge=0,
        description="Number of results returned.",
    )

    retrieval_strategy: RetrievalSource = Field(
        ...,
        description="Primary retrieval strategy used.",
    )

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Retrieval execution latency in milliseconds.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC response creation timestamp.",
    )

    def model_post_init(self, __context: Any) -> None:
        """
        Keep total_results consistent with the actual result list.

        The result list is the source of truth.
        """

        object.__setattr__(
            self,
            "total_results",
            len(self.results),
        )