from datetime import datetime, timezone

from backend.app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
    build_ticket_analytics,
)
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.workflow.handoffs import HandoffRecord
from backend.app.intelligence.workflow.metrics import WorkflowMetrics


def make_ticket(
    *,
    ticket_id: str = "TEST:100",
    priority: str = "High",
    status: str = "in_progress",
) -> Ticket:
    timestamp = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)

    return Ticket(
        ticket_id=ticket_id,
        project="TEST",
        issue_num=100,
        issue_type="Bug",
        priority=priority,
        status=status,
        created_at=timestamp,
        started_at=timestamp,
        ended_at=None,
        resolution_date=None,
        resolution=None,
        reporter="reporter@example.com",
        assignee="agent-001",
        contributor_count=1,
        comments_count=2,
        source="helpdesk",
        metadata={},
    )


def make_workflow_metrics(
    *,
    lifecycle_seconds: float = 1000,
    active_seconds: float = 700,
    waiting_seconds: float = 300,
    transition_count: int = 4,
    unique_status_count: int = 3,
    unique_assignee_count: int = 2,
) -> WorkflowMetrics:
    return WorkflowMetrics(
        ticket_id="TEST:100",
        lifecycle_seconds=lifecycle_seconds,
        active_seconds=active_seconds,
        waiting_seconds=waiting_seconds,
        status_durations={
            "in_progress": active_seconds,
            "waiting": waiting_seconds,
        },
        interval_count=5,
        transition_count=transition_count,
        unique_status_count=unique_status_count,
        unique_assignee_count=unique_assignee_count,
    )


def make_handoff(
    *,
    ticket_id: str = "TEST:100",
    handoff_id: str = "handoff-001",
) -> HandoffRecord:
    timestamp = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    return HandoffRecord(
        handoff_id=handoff_id,
        ticket_id=ticket_id,
        from_agent="agent-001",
        to_agent="agent-002",
        handoff_timestamp=timestamp,
        from_team="Team-A",
        to_team="Team-B",
        status_before="in_progress",
        status_after="waiting",
        time_before_handoff_seconds=200,
        time_after_handoff_seconds=100,
    )


def test_build_ticket_analytics_returns_expected_model():
    ticket = make_ticket()

    metrics = make_workflow_metrics()

    result = build_ticket_analytics(
        ticket=ticket,
        workflow_metrics=metrics,
        handoffs=[make_handoff()],
    )

    assert isinstance(result, TicketAnalytics)


def test_ticket_context_is_preserved():
    ticket = make_ticket(
        ticket_id="PROJ:42",
        priority="Medium",
        status="closed",
    )

    metrics = make_workflow_metrics()

    result = build_ticket_analytics(ticket, metrics)

    assert result.ticket_id == "PROJ:42"
    assert result.project == "TEST"
    assert result.issue_num == 100
    assert result.issue_type == "Bug"
    assert result.priority == "Medium"
    assert result.status == "closed"


def test_time_metrics_and_ratios_are_calculated():
    ticket = make_ticket()

    metrics = make_workflow_metrics(
        lifecycle_seconds=1000,
        active_seconds=700,
        waiting_seconds=300,
    )

    result = build_ticket_analytics(ticket, metrics)

    assert result.lifecycle_seconds == 1000
    assert result.active_seconds == 700
    assert result.waiting_seconds == 300

    assert result.active_ratio == 0.7
    assert result.waiting_ratio == 0.3


def test_zero_lifecycle_does_not_divide_by_zero():
    ticket = make_ticket()

    metrics = make_workflow_metrics(
        lifecycle_seconds=0,
        active_seconds=0,
        waiting_seconds=0,
    )

    result = build_ticket_analytics(ticket, metrics)

    assert result.active_ratio == 0.0
    assert result.waiting_ratio == 0.0


def test_workflow_metrics_are_preserved():
    ticket = make_ticket()

    metrics = make_workflow_metrics(
        transition_count=8,
        unique_status_count=5,
        unique_assignee_count=4,
    )

    result = build_ticket_analytics(ticket, metrics)

    assert result.transition_count == 8
    assert result.unique_status_count == 5
    assert result.unique_assignee_count == 4


def test_only_matching_handoffs_are_counted():
    ticket = make_ticket(ticket_id="TEST:100")

    metrics = make_workflow_metrics()

    matching_handoff = make_handoff(
        ticket_id="TEST:100",
        handoff_id="handoff-001",
    )

    unrelated_handoff = make_handoff(
        ticket_id="TEST:200",
        handoff_id="handoff-002",
    )

    result = build_ticket_analytics(
        ticket=ticket,
        workflow_metrics=metrics,
        handoffs=[
            matching_handoff,
            unrelated_handoff,
        ],
    )

    assert result.handoff_count == 1


def test_high_priority_flag():
    metrics = make_workflow_metrics()

    for priority in ["High", "Highest", "Blocker"]:
        ticket = make_ticket(priority=priority)

        result = build_ticket_analytics(ticket, metrics)

        assert result.is_high_priority is True


def test_non_high_priority_flag():
    ticket = make_ticket(priority="Medium")

    metrics = make_workflow_metrics()

    result = build_ticket_analytics(ticket, metrics)

    assert result.is_high_priority is False


def test_waiting_status_flag():
    ticket = make_ticket(status="waiting")

    metrics = make_workflow_metrics()

    result = build_ticket_analytics(ticket, metrics)

    assert result.is_waiting is True


def test_non_waiting_status_flag():
    ticket = make_ticket(status="in_progress")

    metrics = make_workflow_metrics()

    result = build_ticket_analytics(ticket, metrics)

    assert result.is_waiting is False


def test_handoff_count_is_zero_when_not_provided():
    ticket = make_ticket()

    metrics = make_workflow_metrics()

    result = build_ticket_analytics(
        ticket=ticket,
        workflow_metrics=metrics,
        handoffs=None,
    )

    assert result.handoff_count == 0