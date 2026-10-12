from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from backend.app.intelligence.schemas.ticket import PriorityLevel
from backend.app.intelligence.schemas.workflow import WorkflowEvent


def make_valid_event() -> WorkflowEvent:
    return WorkflowEvent(
        event_id="EVT-1001",
        ticket_id="C06ggn039:1923",
        timestamp=datetime(
            2023,
            3,
            5,
            19,
            34,
            56,
            tzinfo=timezone.utc,
        ),
        activity="status_change",
        status="in_progress",
        assignee="agent-123",
        team="support",
        priority=PriorityLevel.HIGH,
        category="Ticket",
        source="helpdesk",
        metadata={
            "raw_status": "in_progress",
        },
    )


def test_valid_workflow_event_creation():
    event = make_valid_event()

    assert event.event_id == "EVT-1001"
    assert event.ticket_id == "C06ggn039:1923"
    assert event.activity == "status_change"
    assert event.status == "in_progress"
    assert event.priority == PriorityLevel.HIGH


def test_optional_workflow_fields_can_be_missing():
    event = WorkflowEvent(
        event_id="EVT-1002",
        ticket_id="C01:1",
        timestamp=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        activity="ticket_created",
    )

    assert event.status is None
    assert event.assignee is None
    assert event.team is None
    assert event.category is None
    assert event.priority == PriorityLevel.UNKNOWN
    assert event.metadata == {}


def test_default_source_is_helpdesk():
    event = WorkflowEvent(
        event_id="EVT-1003",
        ticket_id="C01:1",
        timestamp=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        activity="ticket_created",
    )

    assert event.source == "helpdesk"


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        WorkflowEvent(
            event_id="EVT-1004",
            ticket_id="C01:1",
            timestamp=datetime(
                2026,
                1,
                1,
            ),
            activity="ticket_created",
        )


def test_empty_event_id_is_rejected():
    with pytest.raises(ValidationError):
        WorkflowEvent(
            event_id="",
            ticket_id="C01:1",
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            activity="ticket_created",
        )


def test_empty_activity_is_rejected():
    with pytest.raises(ValidationError):
        WorkflowEvent(
            event_id="EVT-1005",
            ticket_id="C01:1",
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            activity="",
        )


def test_invalid_priority_assignment_is_rejected():
    event = make_valid_event()

    with pytest.raises(ValidationError):
        event.priority = "Critical"


def test_unexpected_fields_are_rejected():
    with pytest.raises(ValidationError):
        WorkflowEvent(
            event_id="EVT-1006",
            ticket_id="C01:1",
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            activity="ticket_created",
            unexpected_field="should_fail",
        )