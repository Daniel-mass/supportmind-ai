from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.intelligence.schemas.ticket import (
    PriorityLevel,
    Ticket,
)


def make_valid_ticket() -> Ticket:
    return Ticket(
        ticket_id="C06ggn039:1923",
        project="C06ggn039",
        issue_num=1923,
        issue_type="Ticket",
        priority=PriorityLevel.HIGH,
        status="closed",
        created_at=datetime(
            2023,
            3,
            5,
            19,
            34,
            56,
            tzinfo=timezone.utc,
        ),
        started_at=datetime(
            2023,
            3,
            5,
            19,
            34,
            56,
            tzinfo=timezone.utc,
        ),
        resolution_date=datetime(
            2023,
            6,
            13,
            6,
            25,
            26,
            tzinfo=timezone.utc,
        ),
        resolution="Done",
        reporter="user@example.com",
        assignee="agent-123",
        contributor_count=2,
        comments_count=5,
        source="helpdesk",
    )


def test_valid_ticket_creation():
    ticket = make_valid_ticket()

    assert ticket.ticket_id == "C06ggn039:1923"
    assert ticket.project == "C06ggn039"
    assert ticket.issue_num == 1923
    assert ticket.priority == PriorityLevel.HIGH
    assert ticket.status == "closed"


def test_default_priority_is_unknown():
    ticket = make_valid_ticket()

    ticket.priority = PriorityLevel.UNKNOWN

    assert ticket.priority == PriorityLevel.UNKNOWN


def test_optional_fields_can_be_missing():
    ticket = Ticket(
        ticket_id="C01:1",
        project="C01",
        issue_num=1,
        issue_type="Ticket",
        status="open",
        created_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert ticket.started_at is None
    assert ticket.ended_at is None
    assert ticket.resolution_date is None
    assert ticket.resolution is None
    assert ticket.assignee is None


def test_negative_counts_are_rejected():
    with pytest.raises(ValidationError):
        Ticket(
            ticket_id="C01:1",
            project="C01",
            issue_num=1,
            issue_type="Ticket",
            status="open",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            comments_count=-1,
        )


def test_empty_ticket_id_is_rejected():
    with pytest.raises(ValidationError):
        Ticket(
            ticket_id="",
            project="C01",
            issue_num=1,
            issue_type="Ticket",
            status="open",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_unknown_priority_is_supported():
    ticket = make_valid_ticket()

    ticket.priority = "unknown"

    assert ticket.priority == PriorityLevel.UNKNOWN


def test_invalid_priority_is_rejected():
    with pytest.raises(ValidationError):
        ticket = make_valid_ticket()
        ticket.priority = "Critical"


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        Ticket(
            ticket_id="C01:1",
            project="C01",
            issue_num=1,
            issue_type="Ticket",
            status="open",
            created_at=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            unexpected_field="should fail",
        )