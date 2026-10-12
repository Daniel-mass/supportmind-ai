import pytest

from app.agents.resolution_agent.contracts.state import (
    ResolutionType,
)
from app.agents.resolution_agent.output_parser import (
    ResolutionOutputParser,
)


def build_valid_output() -> str:
    return """
    {
        "resolution_type": "policy_based",
        "recommendation": "Apply the subscription refund policy.",
        "rationale": "The retrieved policy evidence supports the refund process.",
        "evidence_ids": ["E1", "E2"],
        "confidence": 0.92,
        "requires_human_review": false,
        "proposed_actions": [
            "Verify the subscription eligibility",
            "Process the refund according to policy"
        ]
    }
    """


def test_parse_valid_resolution_output() -> None:
    recommendation = ResolutionOutputParser.parse(
        build_valid_output(),
        valid_evidence_ids={"E1", "E2", "E3"},
    )

    assert recommendation.resolution_type == ResolutionType.POLICY_BASED
    assert recommendation.recommendation == (
        "Apply the subscription refund policy."
    )
    assert recommendation.evidence_ids == ["E1", "E2"]
    assert recommendation.confidence == 0.92
    assert recommendation.requires_human_review is False


def test_parse_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        ResolutionOutputParser.parse(
            "This is not JSON.",
            valid_evidence_ids={"E1"},
        )


def test_parse_rejects_invalid_schema() -> None:
    invalid_output = """
    {
        "resolution_type": "policy_based",
        "recommendation": "",
        "rationale": "Policy evidence exists.",
        "evidence_ids": ["E1"],
        "confidence": 0.9,
        "requires_human_review": false
    }
    """

    with pytest.raises(
        ValueError,
        match="schema validation",
    ):
        ResolutionOutputParser.parse(
            invalid_output,
            valid_evidence_ids={"E1"},
        )


def test_parse_rejects_unknown_evidence_id() -> None:
    output = """
    {
        "resolution_type": "policy_based",
        "recommendation": "Apply the refund policy.",
        "rationale": "The policy supports this resolution.",
        "evidence_ids": ["E1", "E99"],
        "confidence": 0.9,
        "requires_human_review": false,
        "proposed_actions": []
    }
    """

    with pytest.raises(
        ValueError,
        match="unknown evidence IDs",
    ):
        ResolutionOutputParser.parse(
            output,
            valid_evidence_ids={"E1", "E2"},
        )


def test_parse_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ResolutionOutputParser.parse(
            "",
            valid_evidence_ids={"E1"},
        )


def test_parse_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        ResolutionOutputParser.parse(
            '["E1", "E2"]',
            valid_evidence_ids={"E1", "E2"},
        )