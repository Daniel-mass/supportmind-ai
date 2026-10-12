#output_parser.py
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agents.resolution_agent.contracts.state import (
    ResolutionRecommendation,
)


class ResolutionOutputParser:
    """
    Parses and validates structured Resolution Agent output.

    Raw LLM output is treated as untrusted until it has:

    1. Been parsed as JSON.
    2. Passed the ResolutionRecommendation Pydantic contract.
    3. Passed semantic evidence validation.

    The resolution_type field is intentionally strict.

    The LLM must output one of the exact ResolutionType enum values.
    Semantic aliases such as "policy", "policy_guidance", or
    "technical_issue" are NOT silently normalized.

    If the LLM cannot confidently classify the resolution, it must
    explicitly return:

        "resolution_type": "unknown"
    """

    @staticmethod
    def parse(
        content: str,
        *,
        valid_evidence_ids: set[str],
    ) -> ResolutionRecommendation:
        """
        Parse and validate raw Resolution Agent output.

        Parameters
        ----------
        content:
            Raw textual response produced by the Resolution Agent.

        valid_evidence_ids:
            Evidence IDs that are actually available in the current
            Resolution Agent state.

        Returns
        -------
        ResolutionRecommendation
            Fully validated internal recommendation.

        Raises
        ------
        TypeError
            If input types are invalid.

        ValueError
            If the LLM output is empty, invalid JSON, violates the
            recommendation schema, or references unknown evidence IDs.
        """

        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string."
            )

        if not content.strip():
            raise ValueError(
                "Resolution Agent output must not be empty."
            )

        if not isinstance(valid_evidence_ids, set):
            raise TypeError(
                "valid_evidence_ids must be a set."
            )

        # ------------------------------------------------------------------
        # JSON parsing
        # ------------------------------------------------------------------

        try:
            payload: Any = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Resolution Agent output must be valid JSON."
            ) from exc

        # ------------------------------------------------------------------
        # Root object validation
        # ------------------------------------------------------------------

        if not isinstance(payload, dict):
            raise ValueError(
                "Resolution Agent output must be a JSON object."
            )

        # ------------------------------------------------------------------
        # Pydantic schema validation
        # ------------------------------------------------------------------

        try:
            recommendation = (
                ResolutionRecommendation.model_validate(
                    payload
                )
            )

        except ValidationError as exc:
            # Preserve the exact Pydantic diagnostics so that a production
            # LLM/schema mismatch can be diagnosed without rerunning the
            # expensive Resolution Agent call.

            validation_errors = exc.errors()

            diagnostics = {
                "validation_error": str(exc),
                "validation_errors": validation_errors,
                "payload_keys": sorted(
                    payload.keys()
                ),
                "payload": payload,
            }

            # Keep the diagnostic object intentionally constructed here.
            # ResolutionAgent extracts the structured validation details
            # from the exception message and stores them in execution
            # metadata.

            _ = diagnostics

            raise ValueError(
                "Resolution Agent output failed schema validation. "
                f"Validation errors: "
                f"{json.dumps(validation_errors, default=str)}"
            ) from exc

        # ------------------------------------------------------------------
        # Evidence ID validation
        # ------------------------------------------------------------------

        invalid_evidence_ids = (
            set(recommendation.evidence_ids)
            - valid_evidence_ids
        )

        if invalid_evidence_ids:
            invalid_ids = ", ".join(
                sorted(invalid_evidence_ids)
            )

            raise ValueError(
                "Resolution Agent referenced unknown evidence IDs: "
                f"{invalid_ids}"
            )

        # ------------------------------------------------------------------
        # Trusted internal recommendation
        # ------------------------------------------------------------------

        return recommendation


__all__ = [
    "ResolutionOutputParser",
]