"""
colbert_reranker.py

Production ColBERT / late-interaction reranker for SupportMind AI.

Responsibilities
----------------
1. Accept an already retrieved candidate set.
2. Encode the query into token-level vectors.
3. Encode candidate documents into token-level vectors.
4. Score candidates using ColBERT MaxSim.
5. Reorder candidates by late-interaction relevance.
6. Preserve existing RetrievalResult information.
7. Add the ColBERT score to retrieval_scores.
8. Assign deterministic final ranks.
9. Respect top_k.
10. Avoid mutating original RetrievalResult objects.

Architecture
------------

Dense Retriever ─┐
                  │
Sparse Retriever ─┼──> RRF ──> Candidates
                  │                │
Graph Retriever ──┘                ▼
                              ColBERT
                         MultiVectorEncoder
                                  │
                                MaxSim
                                  │
                                  ▼
                           Reranked Results

Technology
----------
Model:
    colbert-ir/colbertv2.0

Framework:
    sentence-transformers MultiVectorEncoder

Scoring:
    ColBERT late interaction / MaxSim
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any, Sequence

from sentence_transformers import MultiVectorEncoder

from app.retrieval.models.retrieval_models import RetrievalResult


class ColBERTReranker:
    """
    ColBERT-based late-interaction reranker.

    The reranker does not retrieve from Qdrant, Neo4j, or BM25.

    It receives a relatively small candidate set from the
    upstream retrieval/fusion layer and performs token-level
    late-interaction scoring.
    """

    DEFAULT_MODEL_NAME = "colbert-ir/colbertv2.0"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: MultiVectorEncoder | None = None,
    ) -> None:
        """
        Initialize the ColBERT reranker.

        Parameters
        ----------
        model_name:
            Hugging Face model identifier.

        model:
            Optional injected MultiVectorEncoder.

            Dependency injection keeps unit tests independent
            of model downloads and GPU availability.
        """

        self._validate_model_name(model_name)

        self.model_name = model_name

        if model is not None:
            if not isinstance(
                model,
                MultiVectorEncoder,
            ):
                raise ValueError(
                    "model must be a MultiVectorEncoder instance."
                )

            self._model = model

        else:
            self._model = MultiVectorEncoder(
                model_name
            )

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def model(self) -> MultiVectorEncoder:
        """
        Return the underlying ColBERT model.
        """

        return self._model

    # ==========================================================
    # Reranking
    # ==========================================================

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank retrieved candidates using ColBERT MaxSim.

        Parameters
        ----------
        query:
            User query.

        results:
            Candidate RetrievalResult objects.

        top_k:
            Maximum number of results to return.

            If None, all candidates are returned.

        Returns
        -------
        list[RetrievalResult]
            Reranked candidates.
        """

        self._validate_query(query)
        self._validate_results(results)

        if top_k is not None:
            self._validate_top_k(top_k)

        if not results:
            return []

        effective_top_k = (
            len(results)
            if top_k is None
            else min(
                top_k,
                len(results),
            )
        )

        document_texts = [
            result.text
            for result in results
        ]

        scores = self._score_candidates(
            query=query,
            documents=document_texts,
        )

        if len(scores) != len(results):
            raise ValueError(
                "ColBERT returned a score count that does not "
                "match the candidate count."
            )

        scored_results: list[
            tuple[
                float,
                int,
                RetrievalResult,
            ]
        ] = []

        for original_index, (
            result,
            score,
        ) in enumerate(
            zip(results, scores)
        ):
            copied_result = deepcopy(
                result
            )

            colbert_score = (
                self._validate_score(score)
            )

            copied_result.retrieval_scores[
                "colbert_score"
            ] = colbert_score

            scored_results.append(
                (
                    colbert_score,
                    original_index,
                    copied_result,
                )
            )

        # Deterministic ordering:
        #
        # 1. highest ColBERT score
        # 2. original candidate position
        #
        # The second key makes ties deterministic.
        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        reranked_results: list[
            RetrievalResult
        ] = []

        for rank, (
            _score,
            _original_index,
            result,
        ) in enumerate(
            scored_results[:effective_top_k],
            start=1,
        ):
            result.rank = rank

            reranked_results.append(
                result
            )

        return reranked_results

    # ==========================================================
    # ColBERT Scoring
    # ==========================================================

    def _score_candidates(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """
        Encode query/documents and calculate MaxSim scores.

        Sentence Transformers MultiVectorEncoder exposes
        ColBERT-style late interaction through model.similarity().
        """

        if not documents:
            return []

        query_embeddings = (
            self._model.encode_query(
                [query]
            )
        )

        document_embeddings = (
            self._model.encode_document(
                list(documents)
            )
        )

        similarity_scores = (
            self._model.similarity(
                query_embeddings,
                document_embeddings,
            )
        )

        if hasattr(
            similarity_scores,
            "detach",
        ):
            similarity_scores = (
                similarity_scores.detach()
            )

        if hasattr(
            similarity_scores,
            "cpu",
        ):
            similarity_scores = (
                similarity_scores.cpu()
            )

        if hasattr(
            similarity_scores,
            "tolist",
        ):
            similarity_scores = (
                similarity_scores.tolist()
            )

        # model.similarity() returns a matrix:
        #
        # [[score_doc_1, score_doc_2, ...]]
        #
        # because we supplied one query.
        if (
            isinstance(
                similarity_scores,
                list,
            )
            and len(similarity_scores) == 1
        ):
            similarity_scores = (
                similarity_scores[0]
            )

        if not isinstance(
            similarity_scores,
            (list, tuple),
        ):
            raise ValueError(
                "ColBERT similarity output must be a sequence."
            )

        return [
            self._validate_score(score)
            for score in similarity_scores
        ]

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_model_name(
        model_name: str,
    ) -> None:
        """
        Validate model identifier.
        """

        if (
            not isinstance(
                model_name,
                str,
            )
            or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-empty string."
            )

    @staticmethod
    def _validate_query(
        query: str,
    ) -> None:
        """
        Validate reranking query.
        """

        if not isinstance(
            query,
            str,
        ):
            raise ValueError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

    @staticmethod
    def _validate_results(
        results: Sequence[RetrievalResult],
    ) -> None:
        """
        Validate candidate results.
        """

        if results is None:
            raise ValueError(
                "results cannot be None."
            )

        if isinstance(
            results,
            (str, bytes),
        ):
            raise ValueError(
                "results must be a sequence of RetrievalResult objects."
            )

        try:
            result_list = list(results)
        except TypeError as exc:
            raise ValueError(
                "results must be an iterable collection."
            ) from exc

        for result in result_list:
            if not isinstance(
                result,
                RetrievalResult,
            ):
                raise ValueError(
                    "All results must be RetrievalResult instances."
                )

    @staticmethod
    def _validate_top_k(
        top_k: int,
    ) -> None:
        """
        Validate top_k.
        """

        if isinstance(
            top_k,
            bool,
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if (
            not isinstance(
                top_k,
                int,
            )
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

    @staticmethod
    def _validate_score(
        score: Any,
    ) -> float:
        """
        Validate and normalize a ColBERT score.
        """

        if isinstance(
            score,
            bool,
        ):
            raise ValueError(
                "ColBERT score must be numeric."
            )

        if not isinstance(
            score,
            Real,
        ):
            raise ValueError(
                "ColBERT score must be numeric."
            )

        return float(score)


# ==============================================================
# Application Factory
# ==============================================================


def create_colbert_reranker(
    model_name: str = (
        ColBERTReranker.DEFAULT_MODEL_NAME
    ),
) -> ColBERTReranker:
    """
    Create the application-level ColBERT reranker.

    Model loading remains behind a factory so startup wiring
    stays explicit.
    """

    return ColBERTReranker(
        model_name=model_name
    )