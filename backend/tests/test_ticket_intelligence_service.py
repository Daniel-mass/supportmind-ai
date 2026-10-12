from datetime import datetime, timezone

from backend.app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
)
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.schemas.workflow import (
    WorkflowEvent,
)
from backend.app.intelligence.services.ticket_intelligence import (
    TicketIntelligence,
    build_ticket_intelligence,
)
from backend.app.intelligence.workflow.handoffs import (
    HandoffRecord,
)
from backend.app.intelligence.workflow.metrics import (
    WorkflowMetrics,
)


TIMESTAMP = datetime(
    2026,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)


def make_ticket() -> Ticket:
    return Ticket(
        ticket_id="PROJ:100",
        project="PROJ",
        issue_num=100,
        issue_type="Bug",
        priority="High",
        status="in_progress",
        created_at=TIMESTAMP,
        started_at=TIMESTAMP,
        ended_at=None,
        resolution_date=None,
        resolution=None,
        reporter="user@example.com",
        assignee="agent-001",
        contributor_count=1,
        comments_count=2,
        source="helpdesk",
        metadata={},
    )


def make_analytics() -> TicketAnalytics:
    return TicketAnalytics(
        ticket_id="PROJ:100",
        project="PROJ",
        issue_num=100,
        issue_type="Bug",
        priority="High",
        status="in_progress",
        lifecycle_seconds=1000,
        active_seconds=700,
        waiting_seconds=300,
        active_ratio=0.7,
        waiting_ratio=0.3,
        transition_count=4,
        unique_status_count=3,
        unique_assignee_count=2,
        handoff_count=1,
        is_high_priority=True,
        is_waiting=False,
    )


def make_workflow_metrics() -> WorkflowMetrics:
    return WorkflowMetrics(
        ticket_id="PROJ:100",
        lifecycle_seconds=1000,
        active_seconds=700,
        waiting_seconds=300,
        status_durations={
            "in_progress": 700,
            "waiting": 300,
        },
        interval_count=5,
        transition_count=4,
        unique_status_count=3,
        unique_assignee_count=2,
    )


def make_event() -> WorkflowEvent:
    return WorkflowEvent(
        event_id="event-001",
        ticket_id="PROJ:100",
        timestamp=TIMESTAMP,
        activity="status_change",
        status="in_progress",
        assignee="agent-001",
        team="Team-A",
        priority="High",
        category="Bug",
        source="helpdesk_change_history",
        metadata={},
    )


def make_handoff() -> HandoffRecord:
    return HandoffRecord(
        handoff_id="handoff-001",
        ticket_id="PROJ:100",
        from_agent="agent-001",
        to_agent="agent-002",
        handoff_timestamp=TIMESTAMP,
        from_team="Team-A",
        to_team="Team-B",
        status_before="in_progress",
        status_after="waiting",
        time_before_handoff_seconds=100,
        time_after_handoff_seconds=200,
    )


def test_build_ticket_intelligence_returns_expected_model():
    result = build_ticket_intelligence(
        ticket=make_ticket(),
        analytics=make_analytics(),
    )

    assert isinstance(
        result,
        TicketIntelligence,
    )


def test_ticket_and_analytics_are_preserved():
    ticket = make_ticket()
    analytics = make_analytics()

    result = build_ticket_intelligence(
        ticket=ticket,
        analytics=analytics,
    )

    assert result.ticket == ticket
    assert result.analytics == analytics


def test_workflow_metrics_are_preserved():
    workflow_metrics = (
        make_workflow_metrics()
    )

    result = build_ticket_intelligence(
        ticket=make_ticket(),
        analytics=make_analytics(),
        workflow_metrics=workflow_metrics,
    )

    assert (
        result.workflow_metrics
        == workflow_metrics
    )


def test_workflow_events_are_preserved():
    event = make_event()

    result = build_ticket_intelligence(
        ticket=make_ticket(),
        analytics=make_analytics(),
        workflow_events=[event],
    )

    assert result.workflow_events == [
        event
    ]


def test_handoffs_are_preserved():
    handoff = make_handoff()

    result = build_ticket_intelligence(
        ticket=make_ticket(),
        analytics=make_analytics(),
        handoffs=[handoff],
    )

    assert result.handoffs == [
        handoff
    ]


def test_missing_optional_data_returns_empty_lists():
    result = build_ticket_intelligence(
        ticket=make_ticket(),
        analytics=make_analytics(),
    )

    assert result.workflow_metrics is None
    assert result.workflow_events == []
    assert result.handoffs == []


def test_full_ticket_intelligence_composition():
    ticket = make_ticket()
    analytics = make_analytics()
    workflow_metrics = (
        make_workflow_metrics()
    )
    event = make_event()
    handoff = make_handoff()

    result = build_ticket_intelligence(
        ticket=ticket,
        analytics=analytics,
        workflow_metrics=workflow_metrics,
        workflow_events=[event],
        handoffs=[handoff],
    )

    assert result.ticket.ticket_id == "PROJ:100"
    assert (
        result.analytics.ticket_id
        == "PROJ:100"
    )
    assert (
        result.workflow_metrics.ticket_id
        == "PROJ:100"
    )
    assert len(
        result.workflow_events
    ) == 1
    assert len(
        result.handoffs
    ) == 1