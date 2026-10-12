from datetime import datetime, timedelta, timezone

from backend.app.intelligence.schemas.ticket import PriorityLevel
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.event_builder import (
    build_workflow_intervals,
)
from backend.app.intelligence.workflow.handoffs import (
    identify_handoffs,
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
    assignee: str | None,
    status: str = "in_progress",
    team: str | None = "support",
) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        ticket_id=ticket_id,
        timestamp=BASE_TIME + timedelta(
            minutes=minutes
        ),
        activity="assignment",
        status=status,
        assignee=assignee,
        team=team,
        priority=PriorityLevel.MEDIUM,
        source="helpdesk",
    )


def build_intervals():
    events = [
        make_event(
            "E1",
            "T1",
            0,
            "agent-a",
            "in_progress",
        ),
        make_event(
            "E2",
            "T1",
            60,
            "agent-b",
            "waiting",
        ),
        make_event(
            "E3",
            "T1",
            120,
            "agent-b",
            "resolved",
        ),
    ]

    return build_workflow_intervals(events)


def test_one_agent_handoff_is_detected():
    intervals = build_intervals()

    handoffs = identify_handoffs(intervals)

    assert len(handoffs) == 1

    handoff = handoffs[0]

    assert handoff.ticket_id == "T1"
    assert handoff.from_agent == "agent-a"
    assert handoff.to_agent == "agent-b"


def test_handoff_timestamp_is_correct():
    intervals = build_intervals()

    handoff = identify_handoffs(intervals)[0]

    assert handoff.handoff_timestamp == (
        BASE_TIME + timedelta(minutes=60)
    )


def test_time_before_and_after_handoff():
    intervals = build_intervals()

    handoff = identify_handoffs(intervals)[0]

    assert handoff.time_before_handoff_seconds == 3600.0
    assert handoff.time_after_handoff_seconds == 3600.0


def test_same_agent_is_not_a_handoff():
    events = [
        make_event("E1", "T1", 0, "agent-a"),
        make_event("E2", "T1", 60, "agent-a"),
        make_event("E3", "T1", 120, "agent-a"),
    ]

    intervals = build_workflow_intervals(events)

    assert identify_handoffs(intervals) == []


def test_missing_source_agent_is_ignored():
    events = [
        make_event("E1", "T1", 0, None),
        make_event("E2", "T1", 60, "agent-b"),
    ]

    intervals = build_workflow_intervals(events)

    assert identify_handoffs(intervals) == []


def test_missing_destination_agent_is_ignored():
    events = [
        make_event("E1", "T1", 0, "agent-a"),
        make_event("E2", "T1", 60, None),
    ]

    intervals = build_workflow_intervals(events)

    assert identify_handoffs(intervals) == []


def test_multiple_handoffs_are_detected():
    events = [
        make_event("E1", "T1", 0, "agent-a"),
        make_event("E2", "T1", 30, "agent-b"),
        make_event("E3", "T1", 60, "agent-c"),
        make_event("E4", "T1", 90, "agent-c"),
    ]

    intervals = build_workflow_intervals(events)

    handoffs = identify_handoffs(intervals)

    assert len(handoffs) == 2

    assert handoffs[0].from_agent == "agent-a"
    assert handoffs[0].to_agent == "agent-b"

    assert handoffs[1].from_agent == "agent-b"
    assert handoffs[1].to_agent == "agent-c"


def test_different_tickets_do_not_create_handoffs():
    events = [
        make_event("E1", "T1", 0, "agent-a"),
        make_event("E2", "T2", 60, "agent-b"),
    ]

    intervals = build_workflow_intervals(events)

    assert identify_handoffs(intervals) == []


def test_handoff_preserves_status_and_team_context():
    events = [
        make_event(
            "E1",
            "T1",
            0,
            "agent-a",
            status="in_progress",
            team="team-a",
        ),
        make_event(
            "E2",
            "T1",
            60,
            "agent-b",
            status="waiting",
            team="team-b",
        ),
        make_event(
            "E3",
            "T1",
            120,
            "agent-b",
            status="resolved",
            team="team-b",
        ),
    ]

    intervals = build_workflow_intervals(events)

    handoff = identify_handoffs(intervals)[0]

    assert handoff.from_team == "team-a"
    assert handoff.to_team == "team-b"
    assert handoff.status_before == "in_progress"
    assert handoff.status_after == "waiting"