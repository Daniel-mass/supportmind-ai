from datetime import datetime, timezone

from backend.app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
    build_agent_analytics,
)
from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.workflow.handoffs import HandoffRecord
from backend.app.intelligence.workflow.metrics import WorkflowMetrics


def make_agent(
    *,
    agent_id: str = "agent-001",
    name: str = "Agent One",
    team: str = "Team-A",
    active: bool = True,
) -> Agent:
    return Agent(
        agent_id=agent_id,
        name=name,
        team=team,
        active=active,
        source="helpdesk",
        metadata={},
    )


def make_ticket(
    *,
    ticket_id: str,
    issue_num: int,
    project: str = "PROJ",
    issue_type: str = "Bug",
    priority: str = "Medium",
    status: str = "in_progress",
    assignee: str = "agent-001",
) -> Ticket:
    timestamp = datetime(
        2026,
        1,
        1,
        10,
        0,
        tzinfo=timezone.utc,
    )

    return Ticket(
        ticket_id=ticket_id,
        project=project,
        issue_num=issue_num,
        issue_type=issue_type,
        priority=priority,
        status=status,
        created_at=timestamp,
        started_at=timestamp,
        ended_at=None,
        resolution_date=None,
        resolution=None,
        reporter="reporter@example.com",
        assignee=assignee,
        contributor_count=1,
        comments_count=2,
        source="helpdesk",
        metadata={},
    )


def make_metrics(
    *,
    ticket_id: str,
    lifecycle_seconds: float,
    waiting_seconds: float,
    transition_count: int,
) -> WorkflowMetrics:
    active_seconds = lifecycle_seconds - waiting_seconds

    return WorkflowMetrics(
        ticket_id=ticket_id,
        lifecycle_seconds=lifecycle_seconds,
        active_seconds=active_seconds,
        waiting_seconds=waiting_seconds,
        status_durations={
            "in_progress": active_seconds,
            "waiting": waiting_seconds,
        },
        interval_count=4,
        transition_count=transition_count,
        unique_status_count=2,
        unique_assignee_count=2,
    )


def make_handoff(
    *,
    handoff_id: str,
    ticket_id: str,
    from_agent: str,
    to_agent: str,
) -> HandoffRecord:
    timestamp = datetime(
        2026,
        1,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    return HandoffRecord(
        handoff_id=handoff_id,
        ticket_id=ticket_id,
        from_agent=from_agent,
        to_agent=to_agent,
        handoff_timestamp=timestamp,
        from_team="Team-A",
        to_team="Team-B",
        status_before="in_progress",
        status_after="waiting",
        time_before_handoff_seconds=200,
        time_after_handoff_seconds=100,
    )


def test_build_agent_analytics_returns_expected_model():
    agent = make_agent()

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
        )
    ]

    metrics = {
        "PROJ:1": make_metrics(
            ticket_id="PROJ:1",
            lifecycle_seconds=1000,
            waiting_seconds=200,
            transition_count=3,
        )
    }

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket=metrics,
    )

    assert isinstance(result, AgentAnalytics)


def test_only_assigned_tickets_are_counted():
    agent = make_agent(agent_id="agent-001")

    assigned_ticket = make_ticket(
        ticket_id="PROJ:1",
        issue_num=1,
        assignee="agent-001",
    )

    other_ticket = make_ticket(
        ticket_id="PROJ:2",
        issue_num=2,
        assignee="agent-002",
    )

    result = build_agent_analytics(
        agent=agent,
        tickets=[
            assigned_ticket,
            other_ticket,
        ],
        workflow_metrics_by_ticket={
            "PROJ:1": make_metrics(
                ticket_id="PROJ:1",
                lifecycle_seconds=1000,
                waiting_seconds=200,
                transition_count=3,
            ),
            "PROJ:2": make_metrics(
                ticket_id="PROJ:2",
                lifecycle_seconds=5000,
                waiting_seconds=1000,
                transition_count=8,
            ),
        },
    )

    assert result.assigned_ticket_count == 1
    assert result.total_lifecycle_seconds == 1000
    assert result.total_transition_count == 3


def test_closed_and_open_ticket_counts():
    agent = make_agent()

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
            status="closed",
        ),
        make_ticket(
            ticket_id="PROJ:2",
            issue_num=2,
            status="in_progress",
        ),
        make_ticket(
            ticket_id="PROJ:3",
            issue_num=3,
            status="waiting",
        ),
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={},
    )

    assert result.assigned_ticket_count == 3
    assert result.closed_ticket_count == 1
    assert result.open_ticket_count == 2


def test_high_priority_and_waiting_counts():
    agent = make_agent()

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
            priority="High",
            status="waiting",
        ),
        make_ticket(
            ticket_id="PROJ:2",
            issue_num=2,
            priority="Blocker",
            status="in_progress",
        ),
        make_ticket(
            ticket_id="PROJ:3",
            issue_num=3,
            priority="Medium",
            status="waiting",
        ),
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={},
    )

    assert result.high_priority_ticket_count == 2
    assert result.waiting_ticket_count == 2


def test_workflow_time_metrics_are_aggregated():
    agent = make_agent()

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
        ),
        make_ticket(
            ticket_id="PROJ:2",
            issue_num=2,
        ),
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={
            "PROJ:1": make_metrics(
                ticket_id="PROJ:1",
                lifecycle_seconds=1000,
                waiting_seconds=200,
                transition_count=3,
            ),
            "PROJ:2": make_metrics(
                ticket_id="PROJ:2",
                lifecycle_seconds=2000,
                waiting_seconds=600,
                transition_count=5,
            ),
        },
    )

    assert result.total_lifecycle_seconds == 3000
    assert result.average_lifecycle_seconds == 1500

    assert result.total_waiting_seconds == 800
    assert result.average_waiting_seconds == 400

    assert result.total_transition_count == 8


def test_missing_workflow_metrics_are_ignored_safely():
    agent = make_agent()

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
        ),
        make_ticket(
            ticket_id="PROJ:2",
            issue_num=2,
        ),
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={
            "PROJ:1": make_metrics(
                ticket_id="PROJ:1",
                lifecycle_seconds=1000,
                waiting_seconds=200,
                transition_count=3,
            )
        },
    )

    assert result.assigned_ticket_count == 2
    assert result.total_lifecycle_seconds == 1000
    assert result.total_waiting_seconds == 200
    assert result.total_transition_count == 3


def test_handoffs_are_split_into_incoming_and_outgoing():
    agent = make_agent(agent_id="agent-001")

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
        )
    ]

    handoffs = [
        make_handoff(
            handoff_id="h1",
            ticket_id="PROJ:1",
            from_agent="agent-002",
            to_agent="agent-001",
        ),
        make_handoff(
            handoff_id="h2",
            ticket_id="PROJ:1",
            from_agent="agent-001",
            to_agent="agent-003",
        ),
        make_handoff(
            handoff_id="h3",
            ticket_id="PROJ:1",
            from_agent="agent-001",
            to_agent="agent-004",
        ),
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={},
        handoffs=handoffs,
    )

    assert result.incoming_handoff_count == 1
    assert result.outgoing_handoff_count == 2


def test_unrelated_handoffs_are_not_counted_as_assigned_work():
    agent = make_agent(agent_id="agent-001")

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
        )
    ]

    handoffs = [
        make_handoff(
            handoff_id="h1",
            ticket_id="PROJ:999",
            from_agent="agent-002",
            to_agent="agent-001",
        )
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={},
        handoffs=handoffs,
    )

    assert result.incoming_handoff_count == 1


def test_unique_project_and_issue_type_counts():
    agent = make_agent()

    tickets = [
        make_ticket(
            ticket_id="PROJ:1",
            issue_num=1,
            project="PROJ",
            issue_type="Bug",
        ),
        make_ticket(
            ticket_id="PROJ:2",
            issue_num=2,
            project="PROJ",
            issue_type="Ticket",
        ),
        make_ticket(
            ticket_id="OTHER:3",
            issue_num=3,
            project="OTHER",
            issue_type="Bug",
        ),
    ]

    result = build_agent_analytics(
        agent=agent,
        tickets=tickets,
        workflow_metrics_by_ticket={},
    )

    assert result.unique_project_count == 2
    assert result.unique_issue_type_count == 2


def test_zero_tickets_produce_zero_aggregates():
    agent = make_agent()

    result = build_agent_analytics(
        agent=agent,
        tickets=[],
        workflow_metrics_by_ticket={},
        handoffs=[],
    )

    assert result.assigned_ticket_count == 0
    assert result.open_ticket_count == 0
    assert result.closed_ticket_count == 0
    assert result.total_lifecycle_seconds == 0
    assert result.average_lifecycle_seconds == 0
    assert result.total_waiting_seconds == 0
    assert result.average_waiting_seconds == 0
    assert result.total_transition_count == 0
    assert result.unique_project_count == 0
    assert result.unique_issue_type_count == 0