"""
rrf_fusion.py

Reciprocal Rank Fusion (RRF) for SupportMind AI.

Responsibilities
----------------
1. Fuse dense and sparse retrieval result lists.
2. Rank chunks using Reciprocal Rank Fusion.
3. Deduplicate the same retrieval chunk appearing in multiple retrievers.
4. Preserve RetrievalResult metadata and source information.
5. Store the calculated RRF score without destroying the
   original retrieval score.
6. Assign deterministic final ranks.
7. Remain independent of Qdrant, Neo4j, embeddings, LLMs,
   and rerankers.

Important
---------
RRF deduplication MUST happen at retrieval-result/chunk level.

A document can contain many chunks. Therefore document_id
must NOT be used as the primary identity for RRF deduplication.

Preferred identity:

    node_id
        ↓
    id
        ↓
    document_id (fallback only)

Locked Technology Stack
-----------------------
Fusion Strategy : Reciprocal Rank Fusion (RRF)
Contracts       : Pydantic RetrievalResult
"""

from __future__ import annotations

from typing import Sequence

from app.retrieval.models.retrieval_models import RetrievalResult


class RRFFusion:
    """
    Reciprocal Rank Fusion implementation.

    RRF combines ranked result lists without requiring the
    underlying retrieval scores to be comparable.

    Formula
    -------

        RRF(x) = Σ 1 / (k + rank(x))

    where:

        x    = retrieval result / chunk
        k    = RRF smoothing constant
        rank = 1-based rank in a retrieval list

    Important:
        RRF operates on retrieval chunks/results, not entire
        source documents.
    """

    DEFAULT_K = 60

    def __init__(
        self,
        k: int = DEFAULT_K,
    ) -> None:
        """
        Initialize the RRF fusion engine.
        """

        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(
                "RRF k must be a positive integer."
            )

        if k <= 0:
            raise ValueError(
                "RRF k must be greater than zero."
            )

        self.k = k

    # ==========================================================
    # Public API
    # ==========================================================

    def fuse(
        self,
        dense_results: Sequence[RetrievalResult],
        sparse_results: Sequence[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Fuse dense and sparse retrieval results using RRF.

        Deduplication happens at chunk/result level.

        A single document may therefore contribute multiple
        chunks to the fused candidate pool.
        """

        self._validate_results(
            dense_results,
            "dense_results",
        )

        self._validate_results(
            sparse_results,
            "sparse_results",
        )

        self._validate_top_k(top_k)

        if not dense_results and not sparse_results:
            return []

        contributions: dict[str, float] = {}
        representatives: dict[str, RetrievalResult] = {}
        first_seen_order: dict[str, int] = {}

        order_counter = 0

        # ------------------------------------------------------
        # Dense contribution
        # ------------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            key = self._document_key(result)

            if key not in contributions:
                contributions[key] = 0.0
                representatives[key] = result
                first_seen_order[key] = order_counter
                order_counter += 1

            contributions[key] += self._rrf_score(rank)

        # ------------------------------------------------------
        # Sparse contribution
        # ------------------------------------------------------

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            key = self._document_key(result)

            if key not in contributions:
                contributions[key] = 0.0
                representatives[key] = result
                first_seen_order[key] = order_counter
                order_counter += 1

            contributions[key] += self._rrf_score(rank)

        # ------------------------------------------------------
        # Sort
        #
        # 1. Higher RRF score
        # 2. Earlier first appearance
        # ------------------------------------------------------

        ranked_keys = sorted(
            contributions.keys(),
            key=lambda key: (
                -contributions[key],
                first_seen_order[key],
            ),
        )

        fused_results: list[RetrievalResult] = []

        for rank, key in enumerate(
            ranked_keys[:top_k],
            start=1,
        ):
            original = representatives[key]

            rrf_score = contributions[key]

            fused_results.append(
                self._build_fused_result(
                    result=original,
                    rrf_score=rrf_score,
                    rank=rank,
                )
            )

        return fused_results

    # ==========================================================
    # RRF Calculation
    # ==========================================================

    def _rrf_score(
        self,
        rank: int,
    ) -> float:
        """
        Calculate RRF contribution for one retrieval rank.
        """

        if rank <= 0:
            raise ValueError(
                "RRF rank must be greater than zero."
            )

        return 1.0 / (
            self.k + rank
        )

    # ==========================================================
    # Result Construction
    # ==========================================================

    @staticmethod
    def _build_fused_result(
        result: RetrievalResult,
        rrf_score: float,
        rank: int,
    ) -> RetrievalResult:
        """
        Create a new RetrievalResult containing the RRF score.

        The original result is never mutated.
        """

        retrieval_scores = dict(
            result.retrieval_scores or {}
        )

        retrieval_scores["rrf_score"] = rrf_score

        return result.model_copy(
            update={
                "rank": rank,
                "score": rrf_score,
                "retrieval_scores": retrieval_scores,
            }
        )

    # ==========================================================
    # Retrieval Result Identity
    # ==========================================================

    @staticmethod
    def _document_key(
        result: RetrievalResult,
    ) -> str:
        """
        Return the stable retrieval-result identity.

        IMPORTANT
        ---------
        This method intentionally does NOT prioritize document_id.

        One document can contain many chunks. If document_id were
        used first, all chunks from that document would collapse
        into one candidate.

        Preferred identity:

            1. node_id
            2. result.id
            3. document_id as final fallback

        node_id is preferred because it normally identifies the
        actual indexed chunk/node.

        result.id is the next safest fallback because it identifies
        the RetrievalResult itself.

        document_id is only used when no chunk/result identity
        exists.
        """

        # ------------------------------------------------------
        # 1. Chunk/node identity
        # ------------------------------------------------------

        node_id = getattr(
            result,
            "node_id",
            None,
        )

        if isinstance(
            node_id,
            str,
        ) and node_id.strip():

            return (
                f"node:{node_id.strip()}"
            )

        # ------------------------------------------------------
        # 2. Retrieval-result identity
        # ------------------------------------------------------

        result_id = getattr(
            result,
            "id",
            None,
        )

        if isinstance(
            result_id,
            str,
        ) and result_id.strip():

            return (
                f"id:{result_id.strip()}"
            )

        # ------------------------------------------------------
        # 3. Document identity
        #
        # ONLY fallback.
        # ------------------------------------------------------

        document_id = getattr(
            result,
            "document_id",
            None,
        )

        if isinstance(
            document_id,
            str,
        ) and document_id.strip():

            return (
                f"document:{document_id.strip()}"
            )

        # ------------------------------------------------------
        # 4. Last-resort deterministic representation
        # ------------------------------------------------------

        return (
            f"object:{id(result)}"
        )

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_results(
        results: Sequence[RetrievalResult],
        name: str,
    ) -> None:
        """
        Validate a retrieval result collection.
        """

        if results is None:
            raise ValueError(
                f"{name} cannot be None."
            )

        if isinstance(
            results,
            (str, bytes),
        ):
            raise ValueError(
                f"{name} must be a sequence of RetrievalResult objects."
            )

        try:
            result_list = list(results)
        except TypeError as exc:
            raise ValueError(
                f"{name} must be an iterable collection."
            ) from exc

        for result in result_list:

            if not isinstance(
                result,
                RetrievalResult,
            ):
                raise ValueError(
                    f"All items in {name} must be "
                    "RetrievalResult instances."
                )

    @staticmethod
    def _validate_top_k(
        top_k: int,
    ) -> None:
        """
        Validate requested fused result count.
        """

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )


# ==============================================================
# Application Factory
# ==============================================================


def create_rrf_fusion(
    k: int = RRFFusion.DEFAULT_K,
) -> RRFFusion:
    """
    Create the application-level RRF fusion component.
    """

    return RRFFusion(
        k=k
    )