from datetime import datetime, timedelta, timezone

from backend.app.intelligence.schemas.ticket import PriorityLevel
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.event_builder import (
    WorkflowInterval,
    build_workflow_intervals,
)


BASE_TIME = datetime(
    2026,
    1,
    1,
    9,
    0,
    tzinfo=timezone.utc,
)


def make_event(
    event_id: str,
    ticket_id: str,
    minutes: int,
    *,
    activity: str = "status_change",
    status: str = "open",
    assignee: str | None = "agent-1",
) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        ticket_id=ticket_id,
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        activity=activity,
        status=status,
        assignee=assignee,
        priority=PriorityLevel.MEDIUM,
        source="helpdesk",
    )


def test_two_events_create_one_interval():
    events = [
        make_event("E1", "T1", 0),
        make_event("E2", "T1", 30, status="waiting"),
    ]

    intervals = build_workflow_intervals(events)

    assert len(intervals) == 1

    interval = intervals[0]

    assert interval.ticket_id == "T1"
    assert interval.start_event_id == "E1"
    assert interval.end_event_id == "E2"
    assert interval.duration_seconds == 1800.0
    assert interval.status == "open"


def test_three_events_create_two_intervals():
    events = [
        make_event("E1", "T1", 0),
        make_event("E2", "T1", 30, status="waiting"),
        make_event("E3", "T1", 90, status="resolved"),
    ]

    intervals = build_workflow_intervals(events)

    assert len(intervals) == 2

    assert intervals[0].duration_seconds == 1800.0
    assert intervals[1].duration_seconds == 3600.0

    assert intervals[0].status == "open"
    assert intervals[1].status == "waiting"


def test_unsorted_events_are_sorted_deterministically():
    events = [
        make_event("E3", "T1", 90, status="resolved"),
        make_event("E1", "T1", 0),
        make_event("E2", "T1", 30, status="waiting"),
    ]

    intervals = build_workflow_intervals(events)

    assert [i.start_event_id for i in intervals] == [
        "E1",
        "E2",
    ]

    assert [i.end_event_id for i in intervals] == [
        "E2",
        "E3",
    ]


def test_different_tickets_are_not_connected():
    events = [
        make_event("E1", "T1", 0),
        make_event("E2", "T2", 30),
    ]

    intervals = build_workflow_intervals(events)

    assert intervals == []


def test_last_event_has_no_interval():
    events = [
        make_event("E1", "T1", 0),
        make_event("E2", "T1", 30),
        make_event("E3", "T1", 90),
    ]

    intervals = build_workflow_intervals(events)

    assert all(
        interval.end_event_id != "E3"
        or interval.start_event_id == "E2"
        for interval in intervals
    )

    assert len(intervals) == 2


def test_same_timestamp_produces_zero_duration():
    events = [
        make_event("E1", "T1", 0),
        make_event("E2", "T1", 0),
    ]

    intervals = build_workflow_intervals(events)

    assert len(intervals) == 1
    assert intervals[0].duration_seconds == 0.0


def test_empty_event_list_returns_empty_intervals():
    assert build_workflow_intervals([]) == []


def test_interval_contains_starting_event_context():
    events = [
        make_event(
            "E1",
            "T1",
            0,
            activity="assignment",
            status="in_progress",
            assignee="agent-123",
        ),
        make_event(
            "E2",
            "T1",
            60,
            activity="status_change",
            status="waiting",
            assignee="agent-456",
        ),
    ]

    intervals = build_workflow_intervals(events)

    interval = intervals[0]

    assert interval.activity == "assignment"
    assert interval.status == "in_progress"
    assert interval.assignee == "agent-123"
    assert interval.metadata["end_status"] == "waiting"


def test_interval_id_is_deterministic():
    events = [
        make_event("E100", "T1", 0),
        make_event("E200", "T1", 30),
    ]

    intervals = build_workflow_intervals(events)

    assert intervals[0].interval_id == "E100__E200"