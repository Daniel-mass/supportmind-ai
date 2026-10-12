#test_workflow_metrics.py
from datetime import datetime, timedelta, timezone

from backend.app.intelligence.schemas.ticket import PriorityLevel
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.event_builder import (
    build_workflow_intervals,
)
from backend.app.intelligence.workflow.metrics import (
    calculate_workflow_metrics,
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
    minutes: int,
    status: str,
    assignee: str | None = "agent-1",
) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        ticket_id="T1",
        timestamp=BASE_TIME + timedelta(minutes=minutes),
        activity="status_change",
        status=status,
        assignee=assignee,
        priority=PriorityLevel.MEDIUM,
        source="helpdesk",
    )


def build_intervals():
    events = [
        make_event("E1", 0, "open", "agent-1"),
        make_event("E2", 30, "waiting", "agent-1"),
        make_event("E3", 90, "in_progress", "agent-2"),
        make_event("E4", 150, "resolved", "agent-2"),
    ]

    return build_workflow_intervals(events)


def test_metrics_are_calculated():
    intervals = build_intervals()

    metrics = calculate_workflow_metrics(intervals)

    assert len(metrics) == 1

    result = metrics[0]

    assert result.ticket_id == "T1"
    assert result.lifecycle_seconds == 9000.0
    assert result.waiting_seconds == 3600.0
    assert result.active_seconds == 5400.0


def test_status_durations_are_preserved():
    intervals = build_intervals()

    result = calculate_workflow_metrics(intervals)[0]

    assert result.status_durations["open"] == 1800.0
    assert result.status_durations["waiting"] == 3600.0
    assert result.status_durations["in_progress"] == 3600.0


def test_interval_and_transition_counts():
    intervals = build_intervals()

    result = calculate_workflow_metrics(intervals)[0]

    assert result.interval_count == 3
    assert result.transition_count == 3


def test_unique_status_count():
    intervals = build_intervals()

    result = calculate_workflow_metrics(intervals)[0]

    assert result.unique_status_count == 3


def test_unique_assignee_count():
    intervals = build_intervals()

    result = calculate_workflow_metrics(intervals)[0]

    assert result.unique_assignee_count == 2


def test_empty_intervals_return_empty_metrics():
    assert calculate_workflow_metrics([]) == []


def test_only_waiting_time_is_zero_active_time():
    events = [
        make_event("E1", 0, "waiting"),
        make_event("E2", 60, "waiting"),
    ]

    intervals = build_workflow_intervals(events)

    result = calculate_workflow_metrics(intervals)[0]

    assert result.lifecycle_seconds == 3600.0
    assert result.waiting_seconds == 3600.0
    assert result.active_seconds == 0.0