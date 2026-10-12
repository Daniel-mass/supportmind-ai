import pytest
from pydantic import ValidationError

from backend.app.intelligence.schemas.evidence import (
    EvidenceScope,
    EvidenceType,
    IntelligenceEvidence,
)


def make_valid_evidence() -> IntelligenceEvidence:
    return IntelligenceEvidence(
        evidence_id="EV-1001",
        scope=EvidenceScope.TICKET,
        ticket_id="C06ggn039:1923",
        evidence_type=EvidenceType.WORKFLOW_METRIC,
        metric_name="waiting_time",
        value=18.4,
        unit="hours",
        source_dataset="issues_change_history.csv",
        source_fields=[
            "timestamp",
            "status",
        ],
        metadata={
            "status": "waiting",
        },
    )


def test_valid_evidence_creation():
    evidence = make_valid_evidence()

    assert evidence.evidence_id == "EV-1001"
    assert evidence.scope == EvidenceScope.TICKET
    assert evidence.ticket_id == "C06ggn039:1923"
    assert evidence.evidence_type == EvidenceType.WORKFLOW_METRIC
    assert evidence.metric_name == "waiting_time"
    assert evidence.value == 18.4


def test_agent_level_evidence():
    evidence = IntelligenceEvidence(
        evidence_id="EV-1002",
        scope=EvidenceScope.AGENT,
        agent_id="agent-123",
        evidence_type=EvidenceType.TICKET_METRIC,
        metric_name="tickets_handled",
        value=47,
        unit="tickets",
        source_dataset="issues.csv",
        source_fields=[
            "issue_num",
            "issue_assignee",
        ],
    )

    assert evidence.scope == EvidenceScope.AGENT
    assert evidence.agent_id == "agent-123"
    assert evidence.value == 47


def test_organization_level_trend():
    evidence = IntelligenceEvidence(
        evidence_id="EV-1003",
        scope=EvidenceScope.ORGANIZATION,
        evidence_type=EvidenceType.TREND,
        metric_name="ticket_volume_change",
        value=23.5,
        unit="percent",
        source_dataset="issues.csv",
        source_fields=[
            "issue_created",
            "issue_priority",
        ],
    )

    assert evidence.scope == EvidenceScope.ORGANIZATION
    assert evidence.evidence_type == EvidenceType.TREND
    assert evidence.value == 23.5


def test_multiple_source_fields_are_supported():
    evidence = make_valid_evidence()

    assert evidence.source_fields == [
        "timestamp",
        "status",
    ]


def test_empty_evidence_id_is_rejected():
    with pytest.raises(ValidationError):
        IntelligenceEvidence(
            evidence_id="",
            scope=EvidenceScope.TICKET,
            ticket_id="C01:1",
            evidence_type=EvidenceType.TICKET_METRIC,
            metric_name="comments_count",
            value=5,
            source_dataset="issues.csv",
            source_fields=["issue_comments_count"],
        )


def test_empty_metric_name_is_rejected():
    with pytest.raises(ValidationError):
        IntelligenceEvidence(
            evidence_id="EV-1004",
            scope=EvidenceScope.TICKET,
            ticket_id="C01:1",
            evidence_type=EvidenceType.TICKET_METRIC,
            metric_name="",
            value=5,
            source_dataset="issues.csv",
            source_fields=["issue_comments_count"],
        )


def test_empty_source_fields_are_rejected():
    with pytest.raises(ValidationError):
        IntelligenceEvidence(
            evidence_id="EV-1005",
            scope=EvidenceScope.TICKET,
            ticket_id="C01:1",
            evidence_type=EvidenceType.TICKET_METRIC,
            metric_name="comments_count",
            value=5,
            source_dataset="issues.csv",
            source_fields=[],
        )


def test_invalid_evidence_type_is_rejected():
    with pytest.raises(ValidationError):
        IntelligenceEvidence(
            evidence_id="EV-1006",
            scope=EvidenceScope.TICKET,
            ticket_id="C01:1",
            evidence_type="llm_reasoning",
            metric_name="answer_quality",
            value=0.8,
            source_dataset="issues.csv",
            source_fields=["issue_comments_count"],
        )


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        IntelligenceEvidence(
            evidence_id="EV-1007",
            scope=EvidenceScope.TICKET,
            ticket_id="C01:1",
            evidence_type=EvidenceType.TICKET_METRIC,
            metric_name="comments_count",
            value=5,
            source_dataset="issues.csv",
            source_fields=["issue_comments_count"],
            unexpected_field="should_fail",
        )