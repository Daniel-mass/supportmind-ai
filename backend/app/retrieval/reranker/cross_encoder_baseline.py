"""
cross_encoder_baseline.py

Production CrossEncoder reranker for SupportMind AI.

Responsibilities
----------------
1. Accept retrieved candidate documents.
2. Score query/document pairs using a Sentence Transformers
   CrossEncoder.
3. Reorder candidates by CrossEncoder relevance.
4. Preserve RetrievalResult metadata and retrieval scores.
5. Assign deterministic final ranks.
6. Respect top_k.
7. Avoid mutation of original retrieval results.

Architecture
------------

Dense Retriever ─┐
                 ├── RRF ──> Candidates ──> CrossEncoder
Sparse Retriever ┘                              │
                                                ▼
                                         Reranked Results

Technology
----------
Reranker : sentence-transformers CrossEncoder
Model    : cross-encoder/ms-marco-MiniLM-L-6-v2
Testing  : pytest
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any, Sequence

from sentence_transformers import CrossEncoder

from app.retrieval.models.retrieval_models import RetrievalResult


class CrossEncoderBaseline:
    """
    Baseline CrossEncoder reranker.

    The reranker operates only on an already retrieved candidate
    set. It does not perform retrieval itself.
    """

    DEFAULT_MODEL_NAME = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    DEFAULT_MAX_LENGTH = 512

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        model: CrossEncoder | None = None,
    ) -> None:
        """
        Initialize the CrossEncoder reranker.

        Parameters
        ----------
        model_name:
            Hugging Face/Sentence Transformers CrossEncoder model.

        max_length:
            Maximum token length supplied to the model.

        model:
            Optional injected CrossEncoder.

            Dependency injection is supported primarily for testing
            and controlled application startup.
        """

        self._validate_model_name(model_name)
        self._validate_max_length(max_length)

        self.model_name = model_name
        self.max_length = max_length

        if model is not None:
            if not isinstance(model, CrossEncoder):
                raise ValueError(
                    "model must be a CrossEncoder instance."
                )

            self._model = model

        else:
            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
            )

    # ==========================================================
    # Properties
    # ==========================================================

    @property
    def model(self) -> CrossEncoder:
        """
        Return the underlying CrossEncoder model.
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
        Rerank retrieved candidates using the CrossEncoder.

        Parameters
        ----------
        query:
            User search query.

        results:
            Candidate RetrievalResult objects.

        top_k:
            Maximum number of results to return.

            If omitted, all candidates are returned.

        Returns
        -------
        list[RetrievalResult]
            Reranked retrieval results.

        Notes
        -----
        Original RetrievalResult objects are never mutated.
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
            else min(top_k, len(results))
        )

        pairs = [
            (
                query,
                result.text,
            )
            for result in results
        ]

        scores = self._predict_scores(pairs)

        scored_results: list[tuple[
            float,
            int,
            RetrievalResult,
        ]] = []

        for original_index, (
            result,
            score,
        ) in enumerate(
            zip(results, scores)
        ):
            copied_result = deepcopy(result)

            reranker_score = float(score)

            copied_result.retrieval_scores[
                "cross_encoder_score"
            ] = reranker_score

            scored_results.append(
                (
                    reranker_score,
                    original_index,
                    copied_result,
                )
            )

        # Deterministic ordering:
        #
        # 1. highest CrossEncoder score
        # 2. original candidate position
        #
        # The second key prevents unstable ordering when
        # CrossEncoder scores are identical.
        scored_results.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        reranked_results: list[RetrievalResult] = []

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
    # Model Prediction
    # ==========================================================

    def _predict_scores(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        """
        Generate CrossEncoder relevance scores.

        This method is intentionally isolated so inference
        behavior can be controlled and tested independently.
        """

        if not pairs:
            return []

        scores = self._model.predict(
            list(pairs),
            show_progress_bar=False,
        )

        return [
            self._validate_score(score)
            for score in scores
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
            not isinstance(model_name, str)
            or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-empty string."
            )

    @staticmethod
    def _validate_max_length(
        max_length: int,
    ) -> None:
        """
        Validate CrossEncoder maximum sequence length.
        """

        if isinstance(max_length, bool):
            raise ValueError(
                "max_length must be a positive integer."
            )

        if (
            not isinstance(max_length, int)
            or max_length <= 0
        ):
            raise ValueError(
                "max_length must be a positive integer."
            )

    @staticmethod
    def _validate_query(
        query: str,
    ) -> None:
        """
        Validate reranking query.
        """

        if not isinstance(query, str):
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
        Validate candidate retrieval results.
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
        Validate reranking top_k.
        """

        if isinstance(top_k, bool):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if (
            not isinstance(top_k, int)
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
        Validate and normalize a CrossEncoder score.
        """

        if isinstance(score, bool):
            raise ValueError(
                "CrossEncoder score must be numeric."
            )

        if not isinstance(score, Real):
            raise ValueError(
                "CrossEncoder score must be numeric."
            )

        return float(score)


# ==============================================================
# Application Factory
# ==============================================================


def create_cross_encoder_reranker(
    model_name: str = CrossEncoderBaseline.DEFAULT_MODEL_NAME,
    max_length: int = CrossEncoderBaseline.DEFAULT_MAX_LENGTH,
) -> CrossEncoderBaseline:
    """
    Create the application-level CrossEncoder reranker.

    Model loading is kept behind a factory so application startup
    wiring remains explicit.
    """

    return CrossEncoderBaseline(
        model_name=model_name,
        max_length=max_length,
    )