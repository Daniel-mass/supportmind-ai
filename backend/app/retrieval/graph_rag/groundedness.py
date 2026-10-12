"""
groundedness.py

Deterministic groundedness validation for SupportMind Graph-RAG.

This module validates the structural grounding contract between a generated
answer and the evidence supplied to the LLM.

It verifies:

    - the answer is non-empty
    - citation syntax is valid
    - cited evidence IDs exist
    - at least one citation is present when evidence exists
    - evidence is available when the answer is expected to be grounded

It does NOT attempt full semantic claim verification.

Semantic evaluation will later be handled by:
    - RAGAS
    - DeepEval
    - optional LLM-based evaluators
"""

from __future__ import annotations

import re

from app.retrieval.graph_rag.graph_rag_models import (
    GraphRAGContext,
    GroundingResult,
    GroundingStatus,
)


# ============================================================================
# Citation patterns
# ============================================================================

# Valid citation token:
#     [E1]
#     [E12]
_VALID_CITATION_PATTERN = re.compile(
    r"\[(E\d+)\]"
)

# Any bracketed token that starts with E is considered an attempted
# evidence citation and must either match the valid format or be reported
# as malformed.
_ATTEMPTED_CITATION_PATTERN = re.compile(
    r"\[(E[^\[\]]*)\]"
)


# ============================================================================
# Groundedness Validator
# ============================================================================


class GroundednessValidator:
    """
    Validate the structural grounding contract of a Graph-RAG response.

    The validator is deterministic and does not call an LLM.
    """

    def validate(
        self,
        *,
        answer: str,
        context: GraphRAGContext,
    ) -> GroundingResult:
        """
        Validate generated answer against supplied evidence.
        """

        if not isinstance(
            answer,
            str,
        ):
            raise TypeError(
                "answer must be a string."
            )

        if not isinstance(
            context,
            GraphRAGContext,
        ):
            raise TypeError(
                "context must be a GraphRAGContext instance."
            )

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "answer cannot be empty."
            )

        # ------------------------------------------------------------------
        # No evidence available.
        # ------------------------------------------------------------------

        if not context.evidence:
            return GroundingResult(
                status=(
                    GroundingStatus
                    .INSUFFICIENT_EVIDENCE
                ),
                valid=False,
                cited_evidence_ids=[],
                invalid_citation_ids=[],
                unsupported_claims=[],
                issues=[
                    (
                        "No evidence was supplied to the "
                        "generation layer."
                    )
                ],
                metadata={
                    "evidence_count": 0,
                    "citation_count": 0,
                },
            )

        # ------------------------------------------------------------------
        # Evidence IDs available to the answer.
        # ------------------------------------------------------------------

        available_ids = {
            item.evidence_id
            for item in context.evidence
        }

        # ------------------------------------------------------------------
        # Extract valid citation IDs.
        # ------------------------------------------------------------------

        cited_evidence_ids = (
            self.extract_citation_ids(
                answer
            )
        )

        # ------------------------------------------------------------------
        # Detect malformed attempted evidence citations.
        # ------------------------------------------------------------------

        attempted_tokens = (
            _ATTEMPTED_CITATION_PATTERN.findall(
                answer
            )
        )

        invalid_citation_ids = [
            token
            for token in attempted_tokens
            if not re.fullmatch(
                r"E\d+",
                token,
            )
        ]

        # ------------------------------------------------------------------
        # Detect citations that do not exist.
        # ------------------------------------------------------------------

        invalid_citation_ids.extend(
            citation_id
            for citation_id in cited_evidence_ids
            if citation_id not in available_ids
        )

        invalid_citation_ids = (
            self._unique_preserving_order(
                invalid_citation_ids
            )
        )

        # ------------------------------------------------------------------
        # Invalid or malformed citations always take precedence over
        # the "no valid citations" condition.
        # ------------------------------------------------------------------

        if invalid_citation_ids:
            return GroundingResult(
                status=GroundingStatus.INVALID,
                valid=False,
                cited_evidence_ids=(
                    self._unique_preserving_order(
                        cited_evidence_ids
                    )
                ),
                invalid_citation_ids=(
                    invalid_citation_ids
                ),
                unsupported_claims=[],
                issues=[
                    (
                        "One or more citation IDs are "
                        "malformed or do not match the "
                        "supplied evidence."
                    )
                ],
                metadata={
                    "evidence_count": len(
                        context.evidence
                    ),
                    "citation_count": len(
                        cited_evidence_ids
                    ),
                },
            )

        # ------------------------------------------------------------------
        # No citations.
        # ------------------------------------------------------------------

        if not cited_evidence_ids:
            return GroundingResult(
                status=GroundingStatus.PARTIALLY_GROUNDED,
                valid=False,
                cited_evidence_ids=[],
                invalid_citation_ids=[],
                unsupported_claims=[],
                issues=[
                    (
                        "The answer contains no valid "
                        "evidence citations."
                    )
                ],
                metadata={
                    "evidence_count": len(
                        context.evidence
                    ),
                    "citation_count": 0,
                },
            )

        # ------------------------------------------------------------------
        # All citations are structurally valid.
        # ------------------------------------------------------------------

        return GroundingResult(
            status=GroundingStatus.GROUNDED,
            valid=True,
            cited_evidence_ids=(
                self._unique_preserving_order(
                    cited_evidence_ids
                )
            ),
            invalid_citation_ids=[],
            unsupported_claims=[],
            issues=[],
            metadata={
                "evidence_count": len(
                    context.evidence
                ),
                "citation_count": len(
                    cited_evidence_ids
                ),
            },
        )

    # ========================================================================
    # Citation extraction
    # ========================================================================

    @staticmethod
    def extract_citation_ids(
        answer: str,
    ) -> list[str]:
        """
        Extract valid [E<number>] citation IDs from an answer.

        Duplicate citation IDs are preserved during extraction so the caller
        can inspect the raw answer if needed.
        """

        if not isinstance(
            answer,
            str,
        ):
            raise TypeError(
                "answer must be a string."
            )

        return _VALID_CITATION_PATTERN.findall(
            answer
        )

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _unique_preserving_order(
        values: list[str],
    ) -> list[str]:
        """
        Return unique values while preserving first-seen order.
        """

        seen: set[str] = set()
        result: list[str] = []

        for value in values:
            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result