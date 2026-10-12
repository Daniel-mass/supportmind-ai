"""
retrieval_service.py

Production retrieval orchestration for SupportMind AI.

Pipeline
--------

RetrievalRequest
      |
      +----------------------+
      |                      |
      v                      v
   Dense                  Sparse
   Qdrant                  BM25
      |                      |
      +----------+-----------+
                 |
                 v
              RRF Fusion
                 |
                 v
          Optional Graph-RAG
                 |
                 v
        Candidate Deduplication
                 |
                 v
        Optional Reranking
                 |
                 v
          ContextItem mapping
                 |
                 v
          ContextBuilder
                 |
                 v
            BuiltContext


Responsibilities
----------------
- Validate RetrievalRequest.
- Execute dense retrieval.
- Execute sparse retrieval.
- Fuse dense/sparse results using RRF.
- Optionally execute Graph-RAG.
- Deduplicate candidates.
- Optionally rerank candidates.
- Convert RetrievalResult -> ContextItem.
- Build bounded generation-ready context.
- Return BuiltContext.

Non-responsibilities
--------------------
- No query routing.
- No embedding generation.
- No Qdrant implementation.
- No BM25 implementation.
- No graph traversal.
- No reranking algorithm.
- No LLM generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.retrieval.context.context_builder import ContextBuilder
from app.retrieval.context.context_models import (
    BuiltContext,
    ContextBudget,
    ContextItem,
    ContextRequest,
    ContextSource,
)
from app.retrieval.dense.dense_retriever import DenseRetriever
from app.retrieval.graph_rag.graph_retriever import GraphRetriever
from app.retrieval.hybrid.rrf_fusion import RRFFusion
from app.retrieval.models.retrieval_models import (
    RetrievalRequest,
    RetrievalResult,
    RetrievalSource,
)
from app.retrieval.reranker.colbert_reranker import ColBERTReranker
from app.retrieval.sparse.sparse_retriever import SparseRetriever


# ============================================================================
# Configuration
# ============================================================================


@dataclass(frozen=True)
class RetrievalServiceConfig:
    """
    Configuration for the complete retrieval pipeline.
    """

    use_dense: bool = True
    use_sparse: bool = True
    use_graph: bool = True
    use_reranker: bool = True

    # Candidate pool retrieved before final ranking.
    candidate_multiplier: int = 4

    # Maximum candidate pool size.
    max_candidate_k: int = 40

    # RRF smoothing constant.
    rrf_k: int = RRFFusion.DEFAULT_K

    # Final context budget.
    context_max_items: int = 8
    context_max_characters: int = 12000

    include_metadata: bool = True
    include_provenance: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.use_dense, bool):
            raise TypeError("use_dense must be a boolean")

        if not isinstance(self.use_sparse, bool):
            raise TypeError("use_sparse must be a boolean")

        if not isinstance(self.use_graph, bool):
            raise TypeError("use_graph must be a boolean")

        if not isinstance(self.use_reranker, bool):
            raise TypeError("use_reranker must be a boolean")

        if (
            isinstance(self.candidate_multiplier, bool)
            or not isinstance(self.candidate_multiplier, int)
        ):
            raise TypeError(
                "candidate_multiplier must be an integer"
            )

        if self.candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than zero"
            )

        if (
            isinstance(self.max_candidate_k, bool)
            or not isinstance(self.max_candidate_k, int)
        ):
            raise TypeError(
                "max_candidate_k must be an integer"
            )

        if self.max_candidate_k <= 0:
            raise ValueError(
                "max_candidate_k must be greater than zero"
            )

        if (
            isinstance(self.rrf_k, bool)
            or not isinstance(self.rrf_k, int)
        ):
            raise TypeError(
                "rrf_k must be an integer"
            )

        if self.rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero"
            )

        if (
            isinstance(self.context_max_items, bool)
            or not isinstance(self.context_max_items, int)
        ):
            raise TypeError(
                "context_max_items must be an integer"
            )

        if self.context_max_items <= 0:
            raise ValueError(
                "context_max_items must be greater than zero"
            )

        if (
            isinstance(self.context_max_characters, bool)
            or not isinstance(self.context_max_characters, int)
        ):
            raise TypeError(
                "context_max_characters must be an integer"
            )

        if self.context_max_characters <= 0:
            raise ValueError(
                "context_max_characters must be greater than zero"
            )

        if not isinstance(self.include_metadata, bool):
            raise TypeError(
                "include_metadata must be a boolean"
            )

        if not isinstance(self.include_provenance, bool):
            raise TypeError(
                "include_provenance must be a boolean"
            )


# ============================================================================
# Retrieval Service
# ============================================================================


class RetrievalService:
    """
    Complete SupportMind retrieval pipeline.

    The service is intentionally thin.

    Each retrieval/reranking/context component owns its own algorithm.
    This class only coordinates them.
    """

    def __init__(
        self,
        *,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        graph_retriever: GraphRetriever | None = None,
        reranker: Any | None = None,
        rrf_fusion: RRFFusion | None = None,
        context_builder: ContextBuilder | None = None,
        config: RetrievalServiceConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else RetrievalServiceConfig()
        )

        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.graph_retriever = graph_retriever
        self.reranker = reranker

        self.rrf_fusion = (
            rrf_fusion
            if rrf_fusion is not None
            else RRFFusion(
                k=self.config.rrf_k
            )
        )

        self.context_builder = (
            context_builder
            if context_builder is not None
            else ContextBuilder()
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> BuiltContext:
        """
        Execute the complete retrieval pipeline.

        Parameters
        ----------
        request:
            Validated RetrievalRequest.

        Returns
        -------
        BuiltContext
            Final bounded generation-ready context.
        """

        self._validate_request(request)

        start = perf_counter()

        candidate_k = self._candidate_k(
            request.top_k
        )

        candidate_request = request.model_copy(
            update={
                "top_k": candidate_k,
            }
        )

        # ------------------------------------------------------------------
        # 1. Dense retrieval
        # ------------------------------------------------------------------

        dense_results: list[RetrievalResult] = []

        if self.config.use_dense:
            if self.dense_retriever is None:
                raise RuntimeError(
                    "Dense retrieval is enabled but "
                    "dense_retriever is not configured."
                )

            if request.query_vector is None:
                raise ValueError(
                    "query_vector is required when dense retrieval "
                    "is enabled."
                )

            dense_results = self.dense_retriever.retrieve(
                request=candidate_request,
                query_vector=request.query_vector,
            )

        # ------------------------------------------------------------------
        # 2. Sparse retrieval
        # ------------------------------------------------------------------

        sparse_results: list[RetrievalResult] = []

        if self.config.use_sparse:
            if self.sparse_retriever is None:
                raise RuntimeError(
                    "Sparse retrieval is enabled but "
                    "sparse_retriever is not configured."
                )

            sparse_results = self.sparse_retriever.retrieve(
                request=candidate_request
            )

        # ------------------------------------------------------------------
        # 3. Dense + sparse fusion
        # ------------------------------------------------------------------

        fused_results = self._fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=candidate_k,
        )

        # ------------------------------------------------------------------
        # 4. Graph-RAG
        # ------------------------------------------------------------------

        graph_results: list[RetrievalResult] = []

        if (
            self.config.use_graph
            and request.include_graph
        ):
            if self.graph_retriever is None:
                raise RuntimeError(
                    "Graph retrieval is enabled for this request but "
                    "graph_retriever is not configured."
                )

            graph_results = self.graph_retriever.retrieve(
                request=candidate_request
            )

        # ------------------------------------------------------------------
        # 5. Combine + deterministic deduplication
        # ------------------------------------------------------------------

        candidates = self._merge_candidates(
            fused_results,
            graph_results,
        )

        # ------------------------------------------------------------------
        # 6. Reranking
        # ------------------------------------------------------------------

        reranked = False

        if (
            self.config.use_reranker
            and request.rerank
            and self.reranker is not None
            and candidates
        ):
            rerank_top_k = (
                request.rerank_top_k
                if request.rerank_top_k is not None
                else request.top_k
            )

            candidates = self._rerank(
                query=request.query,
                candidates=candidates,
                top_k=rerank_top_k,
                reranker=self.reranker,
            )

            reranked = True

        # ------------------------------------------------------------------
        # 7. Final top-k
        # ------------------------------------------------------------------

        candidates = candidates[
            :request.top_k
        ]

        candidates = self._assign_final_ranks(
            candidates
        )

        # ------------------------------------------------------------------
        # 8. RetrievalResult -> ContextItem
        # ------------------------------------------------------------------

        context_items = [
            self._to_context_item(
                result
            )
            for result in candidates
        ]

        # ------------------------------------------------------------------
        # 9. ContextRequest
        # ------------------------------------------------------------------

        context_request = ContextRequest(
            query=request.query,
            results=context_items,
            budget=ContextBudget(
                max_items=self.config.context_max_items,
                max_characters=self.config.context_max_characters,
            ),
            include_metadata=self.config.include_metadata,
            include_provenance=self.config.include_provenance,
        )

        # ------------------------------------------------------------------
        # 10. ContextBuilder
        # ------------------------------------------------------------------

        built_context = self.context_builder.build(
            context_request
        )

        # ------------------------------------------------------------------
        # 11. Attach execution diagnostics
        # ------------------------------------------------------------------

        latency_ms = (
            perf_counter() - start
        ) * 1000.0

        # BuiltContext intentionally remains clean and focused on context.
        # Store diagnostics in request metadata only if downstream code
        # needs them. The canonical retrieval diagnostics are exposed through
        # the service method below.

        self._last_execution_metadata = {
            "latency_ms": latency_ms,
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused_results),
            "graph_count": len(graph_results),
            "candidate_count": len(candidates),
            "reranked": reranked,
            "final_count": len(candidates),
        }

        return built_context

    # =========================================================================
    # Fusion
    # =========================================================================

    def _fuse(
        self,
        *,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Fuse dense and sparse candidates using RRF.

        If only one retrieval source is enabled, preserve its native results
        rather than artificially applying RRF to a single list.
        """

        if dense_results and sparse_results:
            return self.rrf_fusion.fuse(
                dense_results=dense_results,
                sparse_results=sparse_results,
                top_k=top_k,
            )

        if dense_results:
            return list(
                dense_results[:top_k]
            )

        if sparse_results:
            return list(
                sparse_results[:top_k]
            )

        return []

    # =========================================================================
    # Candidate merge
    # =========================================================================

    @staticmethod
    def _merge_candidates(
        fused_results: list[RetrievalResult],
        graph_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Merge hybrid and graph candidates.

        Hybrid/RRF results are kept first because they are the primary
        retrieval candidates.

        Graph results are supplementary structural evidence.

        IMPORTANT
        ---------
        Deduplication is performed at retrieval-result/chunk level.

        A single document can contain many chunks. Therefore,
        document_id must NOT be used as the primary deduplication key.

        Preferred identity:

            1. node_id
            2. result.id
            3. document_id as final fallback
        """

        merged: list[RetrievalResult] = []
        seen: set[str] = set()

        for result in (
            list(fused_results)
            + list(graph_results)
        ):
            node_id = getattr(
                result,
                "node_id",
                None,
            )

            result_id = getattr(
                result,
                "id",
                None,
            )

            document_id = getattr(
                result,
                "document_id",
                None,
            )

            if (
                isinstance(node_id, str)
                and node_id.strip()
            ):
                key = (
                    f"node:{node_id.strip()}"
                )

            elif (
                isinstance(result_id, str)
                and result_id.strip()
            ):
                key = (
                    f"id:{result_id.strip()}"
                )

            elif (
                isinstance(document_id, str)
                and document_id.strip()
            ):
                key = (
                    f"document:{document_id.strip()}"
                )

            else:
                key = (
                    f"object:{id(result)}"
                )

            if key in seen:
                continue

            seen.add(key)
            merged.append(result)

        return merged

    # =========================================================================
    # Reranking
    # =========================================================================

    @staticmethod
    def _rerank(
        *,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int,
        reranker: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        Execute configured reranker.

        Supports:
            reranker.rerank(query, candidates, top_k=...)
            reranker.rerank(query, candidates)
            callable(query, candidates)
        """

        if reranker is None:
            raise RuntimeError(
                "reranker is required for reranking."
            )

        rerank_method = getattr(
            reranker,
            "rerank",
            None,
        )

        if callable(rerank_method):
            try:
                results = rerank_method(
                    query,
                    candidates,
                    top_k=top_k,
                )
            except TypeError:
                results = rerank_method(
                    query,
                    candidates,
                )
        elif callable(reranker):
            results = reranker(
                query,
                candidates,
            )
        else:
            raise TypeError(
                "Reranker must expose rerank() or be callable."
            )

        if results is None:
            return []

        if isinstance(
            results,
            (str, bytes),
        ):
            raise TypeError(
                "Reranker must return retrieval results."
            )

        return list(results)

    # =========================================================================
    # Context conversion
    # =========================================================================

    @staticmethod
    def _to_context_item(result: RetrievalResult) -> ContextItem:
       source_map = {
        RetrievalSource.DENSE: ContextSource.DENSE,
        RetrievalSource.SPARSE: ContextSource.BM25,
        RetrievalSource.HYBRID: ContextSource.HYBRID,
        RetrievalSource.GRAPH: ContextSource.GRAPH,
        RetrievalSource.RERANKED: ContextSource.RERANKED,
    }

       source = source_map.get(
        result.source,
        ContextSource.HYBRID,
    )

       raw_metadata = dict(result.metadata or {})

    # Only pass retrieval/context-relevant metadata
    # to ContextBuilder.
       allowed_metadata_keys = {
        "source",
        "source_file",
        "document_id",
        "chunk_index",
        "total_chunks",
        "word_count",
        "chunking_strategy",
        "original_text",
        "window",
    }

       metadata = {
        key: raw_metadata[key]
        for key in allowed_metadata_keys
        if key in raw_metadata
    }

       metadata["node_id"] = result.node_id
       metadata["document_id"] = result.document_id

       if result.retrieval_scores:
        metadata["retrieval_scores"] = dict(
            result.retrieval_scores
        )

        return ContextItem(
        id=result.id,
        text=result.text,
        source=source,
        rank=result.rank or 1,
        score=float(result.score),
        metadata=metadata,
    )

    # =========================================================================
    # Rank normalization
    # =========================================================================

    @staticmethod
    def _assign_final_ranks(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Assign deterministic 1-based final ranks.

        RetrievalResult is copied instead of mutating the original object.
        """

        return [
            result.model_copy(
                update={
                    "rank": rank,
                }
            )
            for rank, result in enumerate(
                results,
                start=1,
            )
        ]

    # =========================================================================
    # Candidate sizing
    # =========================================================================

    def _candidate_k(
        self,
        top_k: int,
    ) -> int:
        """
        Calculate candidate pool size before reranking.
        """

        candidate_k = (
            top_k
            * self.config.candidate_multiplier
        )

        return min(
            candidate_k,
            self.config.max_candidate_k,
        )

    # =========================================================================
    # Validation
    # =========================================================================

    @staticmethod
    def _validate_request(
        request: RetrievalRequest,
    ) -> None:
        """
        Validate RetrievalRequest.
        """

        if request is None:
            raise ValueError(
                "request cannot be None."
            )

        if not isinstance(
            request,
            RetrievalRequest,
        ):
            raise TypeError(
                "request must be a RetrievalRequest instance."
            )

        if not request.query.strip():
            raise ValueError(
                "request query cannot be empty."
            )

        if request.top_k <= 0:
            raise ValueError(
                "request top_k must be greater than zero."
            )

    # =========================================================================
    # Diagnostics
    # =========================================================================

    @property
    def last_execution_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return diagnostics from the most recent retrieval execution.
        """

        return dict(
            getattr(
                self,
                "_last_execution_metadata",
                {},
            )
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """
        Close components that expose close().
        """

        components = (
            self.dense_retriever,
            self.sparse_retriever,
            self.graph_retriever,
            self.reranker,
            self.context_builder,
        )

        closed: set[int] = set()

        for component in components:
            if component is None:
                continue

            component_id = id(component)

            if component_id in closed:
                continue

            closed.add(component_id)

            close_method = getattr(
                component,
                "close",
                None,
            )

            if callable(close_method):
                close_method()

    def __enter__(
        self,
    ) -> "RetrievalService":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()


# ============================================================================
# Application Factory
# ============================================================================


def create_retrieval_service(
    *,
    dense_retriever: DenseRetriever | None = None,
    sparse_retriever: SparseRetriever | None = None,
    graph_retriever: GraphRetriever | None = None,
    reranker: Any | None = None,
    rrf_fusion: RRFFusion | None = None,
    context_builder: ContextBuilder | None = None,
    config: RetrievalServiceConfig | None = None,
) -> RetrievalService:
    """
    Create the application-level retrieval service.
    """

    return RetrievalService(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        graph_retriever=graph_retriever,
        reranker=reranker,
        rrf_fusion=rrf_fusion,
        context_builder=context_builder,
        config=config,
    )