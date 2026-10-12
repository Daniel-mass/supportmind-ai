from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agents.escalation_agent.contracts.state import (
    EscalationRecommendation,
)


class EscalationOutputParser:
    """
    Parse and validate the structured output produced by the
    Escalation Agent.

    The parser is intentionally strict.

    Responsibilities:
    - Parse JSON.
    - Validate the JSON against EscalationRecommendation.
    - Validate that every cited evidence ID exists in the supplied state.
    - Reject unsupported or malformed output.

    The parser does NOT:
    - infer missing fields,
    - convert Markdown into structured data,
    - invent evidence IDs,
    - repair arbitrary LLM prose,
    - decide whether escalation is correct.
    """

    @staticmethod
    def parse(
        content: str,
        *,
        valid_evidence_ids: set[str],
    ) -> EscalationRecommendation:
        """
        Parse an LLM response into EscalationRecommendation.

        Parameters
        ----------
        content:
            Raw textual response produced by the LLM.

        valid_evidence_ids:
            Canonical evidence IDs available in the current state.

        Returns
        -------
        EscalationRecommendation
            Fully validated structured recommendation.

        Raises
        ------
        TypeError
            If the input types are invalid.

        ValueError
            If JSON parsing, schema validation, or evidence validation
            fails.
        """

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string."
            )

        if not isinstance(valid_evidence_ids, set):
            raise TypeError(
                "valid_evidence_ids must be a set[str]."
            )

        normalized_content = content.strip()

        if not normalized_content:
            raise ValueError(
                "Escalation Agent response is empty."
            )

        # --------------------------------------------------------------
        # JSON parsing
        # --------------------------------------------------------------

        try:
            payload: Any = json.loads(normalized_content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Escalation Agent response is not valid JSON. "
                "The agent must return only the required JSON object "
                "without Markdown, commentary, or code fences."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                "Escalation Agent response must be a JSON object."
            )

        # --------------------------------------------------------------
        # Pydantic schema validation
        # --------------------------------------------------------------

        try:
            recommendation = (
                EscalationRecommendation.model_validate(payload)
            )

        except ValidationError as exc:
            raise ValueError(
                "Escalation Agent response failed "
                "EscalationRecommendation validation: "
                f"{exc}"
            ) from exc

        # --------------------------------------------------------------
        # Evidence grounding validation
        # --------------------------------------------------------------

        invalid_evidence_ids = (
            set(recommendation.evidence_ids)
            - valid_evidence_ids
        )

        if invalid_evidence_ids:
            raise ValueError(
                "Escalation Agent cited unknown evidence IDs: "
                f"{sorted(invalid_evidence_ids)}. "
                f"Valid evidence IDs: "
                f"{sorted(valid_evidence_ids)}"
            )

        return recommendation