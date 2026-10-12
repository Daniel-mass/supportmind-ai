from backend.app.intelligence.analytics.agent_analytics import AgentAnalytics
from backend.app.intelligence.analytics.organization import (
    OrganizationAnalytics,
    build_organization_analytics,
)
from backend.app.intelligence.analytics.team_analytics import TeamAnalytics
from backend.app.intelligence.analytics.ticket_analytics import TicketAnalytics
from backend.app.intelligence.schemas.agent import Agent
from backend.app.intelligence.schemas.overview import DistributionItem
from backend.app.intelligence.workflow.bottlenecks import BottleneckCandidate
from backend.app.intelligence.workflow.variants import WorkflowVariant


def make_agent(
    *,
    agent_id: str,
    team: str,
    active: bool = True,
) -> Agent:
    return Agent(
        agent_id=agent_id,
        name=agent_id,
        team=team,
        active=active,
        source="helpdesk",
        metadata={},
    )


def make_ticket_analytics(
    *,
    ticket_id: str,
    priority: str,
    status: str,
    lifecycle: float,
    waiting: float,
    transitions: int,
    handoffs: int,
    high_priority: bool,
    is_waiting: bool,
) -> TicketAnalytics:
    return TicketAnalytics(
        ticket_id=ticket_id,
        project="PROJ",
        issue_num=1,
        issue_type="Bug",
        priority=priority,
        status=status,
        lifecycle_seconds=lifecycle,
        active_seconds=lifecycle - waiting,
        waiting_seconds=waiting,
        active_ratio=(
            (lifecycle - waiting) / lifecycle
            if lifecycle
            else 0.0
        ),
        waiting_ratio=(
            waiting / lifecycle
            if lifecycle
            else 0.0
        ),
        transition_count=transitions,
        unique_status_count=2,
        unique_assignee_count=1,
        handoff_count=handoffs,
        is_high_priority=high_priority,
        is_waiting=is_waiting,
    )


def make_agent_analytics(
    *,
    agent_id: str,
    team: str,
    assigned: int,
) -> AgentAnalytics:
    return AgentAnalytics(
        agent_id=agent_id,
        name=agent_id,
        team=team,
        active=True,
        assigned_ticket_count=assigned,
        open_ticket_count=assigned,
        closed_ticket_count=0,
        high_priority_ticket_count=0,
        waiting_ticket_count=0,
        total_lifecycle_seconds=assigned * 1000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=assigned * 100,
        average_waiting_seconds=100,
        total_transition_count=assigned * 2,
        incoming_handoff_count=1,
        outgoing_handoff_count=1,
        unique_project_count=1,
        unique_issue_type_count=1,
    )


def make_team_analytics(
    *,
    team: str,
    assigned: int,
) -> TeamAnalytics:
    return TeamAnalytics(
        team=team,
        agent_count=1,
        active_agent_count=1,
        assigned_ticket_count=assigned,
        open_ticket_count=assigned,
        closed_ticket_count=0,
        high_priority_ticket_count=0,
        waiting_ticket_count=0,
        total_lifecycle_seconds=assigned * 1000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=assigned * 100,
        average_waiting_seconds=100,
        total_transition_count=assigned * 2,
        incoming_handoff_count=1,
        outgoing_handoff_count=1,
    )


def make_bottleneck(
    *,
    status: str,
    score: float,
    total_seconds: float,
) -> BottleneckCandidate:
    return BottleneckCandidate(
        status=status,
        interval_count=10,
        ticket_count=5,
        total_seconds=total_seconds,
        average_seconds=total_seconds / 5,
        volume_share=0.5,
        duration_share=0.5,
        recurrence_rate=0.5,
        bottleneck_score=score,
    )


def make_variant(
    *,
    variant_id: str,
    ticket_count: int,
    percentage: float,
) -> WorkflowVariant:
    return WorkflowVariant(
        variant_id=variant_id,
        path=["in_progress", "waiting", "closed"],
        path_string="in_progress -> waiting -> closed",
        ticket_count=ticket_count,
        percentage=percentage,
    )


def test_build_organization_analytics_returns_expected_model():
    result = build_organization_analytics(
        agents=[],
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=[],
    )

    assert isinstance(result, OrganizationAnalytics)


def test_agent_and_team_counts():
    agents = [
        make_agent(
            agent_id="agent-001",
            team="Team-A",
            active=True,
        ),
        make_agent(
            agent_id="agent-002",
            team="Team-A",
            active=False,
        ),
        make_agent(
            agent_id="agent-003",
            team="Team-B",
            active=True,
        ),
    ]

    result = build_organization_analytics(
        agents=agents,
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=[],
    )

    assert result.agent_count == 3
    assert result.active_agent_count == 2
    assert result.team_count == 2


def test_ticket_totals_and_time_metrics():
    tickets = [
        make_ticket_analytics(
            ticket_id="PROJ:1",
            priority="High",
            status="in_progress",
            lifecycle=1000,
            waiting=200,
            transitions=3,
            handoffs=1,
            high_priority=True,
            is_waiting=False,
        ),
        make_ticket_analytics(
            ticket_id="PROJ:2",
            priority="Medium",
            status="closed",
            lifecycle=2000,
            waiting=600,
            transitions=5,
            handoffs=2,
            high_priority=False,
            is_waiting=False,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=tickets,
        agent_analytics=[],
        team_analytics=[],
    )

    assert result.total_ticket_count == 2
    assert result.open_ticket_count == 1
    assert result.closed_ticket_count == 1

    assert result.high_priority_ticket_count == 1
    assert result.waiting_ticket_count == 0

    assert result.total_lifecycle_seconds == 3000
    assert result.average_lifecycle_seconds == 1500

    assert result.total_waiting_seconds == 800
    assert result.average_waiting_seconds == 400

    assert result.total_transition_count == 8
    assert result.total_handoff_count == 3


def test_priority_distribution_is_counted_and_sorted():
    tickets = [
        make_ticket_analytics(
            ticket_id="PROJ:1",
            priority="Medium",
            status="open",
            lifecycle=100,
            waiting=10,
            transitions=1,
            handoffs=0,
            high_priority=False,
            is_waiting=False,
        ),
        make_ticket_analytics(
            ticket_id="PROJ:2",
            priority="Medium",
            status="open",
            lifecycle=100,
            waiting=10,
            transitions=1,
            handoffs=0,
            high_priority=False,
            is_waiting=False,
        ),
        make_ticket_analytics(
            ticket_id="PROJ:3",
            priority="High",
            status="open",
            lifecycle=100,
            waiting=10,
            transitions=1,
            handoffs=0,
            high_priority=True,
            is_waiting=False,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=tickets,
        agent_analytics=[],
        team_analytics=[],
    )

    assert [
        (item.label, item.count)
        for item in result.priority_distribution
    ] == [
        ("Medium", 2),
        ("High", 1),
    ]


def test_status_distribution_is_counted():
    tickets = [
        make_ticket_analytics(
            ticket_id="PROJ:1",
            priority="Medium",
            status="open",
            lifecycle=100,
            waiting=10,
            transitions=1,
            handoffs=0,
            high_priority=False,
            is_waiting=False,
        ),
        make_ticket_analytics(
            ticket_id="PROJ:2",
            priority="High",
            status="closed",
            lifecycle=100,
            waiting=10,
            transitions=1,
            handoffs=0,
            high_priority=True,
            is_waiting=False,
        ),
        make_ticket_analytics(
            ticket_id="PROJ:3",
            priority="Medium",
            status="open",
            lifecycle=100,
            waiting=10,
            transitions=1,
            handoffs=0,
            high_priority=False,
            is_waiting=False,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=tickets,
        agent_analytics=[],
        team_analytics=[],
    )

    assert [
        (item.label, item.count)
        for item in result.status_distribution
    ] == [
        ("open", 2),
        ("closed", 1),
    ]


def test_team_workload_is_sorted_by_assigned_tickets():
    team_analytics = [
        make_team_analytics(
            team="Team-A",
            assigned=5,
        ),
        make_team_analytics(
            team="Team-B",
            assigned=12,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=team_analytics,
    )

    assert [
        (item.label, item.count)
        for item in result.team_workload
    ] == [
        ("Team-B", 12),
        ("Team-A", 5),
    ]


def test_bottlenecks_are_sorted_by_score():
    bottlenecks = [
        make_bottleneck(
            status="waiting",
            score=0.20,
            total_seconds=2000,
        ),
        make_bottleneck(
            status="in_progress",
            score=0.60,
            total_seconds=5000,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=[],
        bottlenecks=bottlenecks,
    )

    assert [
        item.status
        for item in result.bottlenecks
    ] == [
        "in_progress",
        "waiting",
    ]


def test_workflow_variants_are_sorted_by_ticket_count():
    variants = [
        make_variant(
            variant_id="variant-a",
            ticket_count=5,
            percentage=10,
        ),
        make_variant(
            variant_id="variant-b",
            ticket_count=15,
            percentage=30,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=[],
        workflow_variants=variants,
    )

    assert [
        item.variant_id
        for item in result.workflow_variants
    ] == [
        "variant-b",
        "variant-a",
    ]


def test_emerging_issues_are_preserved():
    issues = [
        DistributionItem(
            label="Login failures",
            count=24,
        ),
        DistributionItem(
            label="Payment issues",
            count=15,
        ),
    ]

    result = build_organization_analytics(
        agents=[],
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=[],
        emerging_issues=issues,
    )

    assert result.emerging_issues == issues


def test_empty_inputs_produce_zero_safe_output():
    result = build_organization_analytics(
        agents=[],
        ticket_analytics=[],
        agent_analytics=[],
        team_analytics=[],
    )

    assert result.agent_count == 0
    assert result.active_agent_count == 0
    assert result.team_count == 0

    assert result.total_ticket_count == 0
    assert result.open_ticket_count == 0
    assert result.closed_ticket_count == 0

    assert result.total_lifecycle_seconds == 0
    assert result.average_lifecycle_seconds == 0

    assert result.total_waiting_seconds == 0
    assert result.average_waiting_seconds == 0

    assert result.total_transition_count == 0
    assert result.total_handoff_count == 0

    assert result.priority_distribution == []
    assert result.status_distribution == []
    assert result.team_workload == []
    assert result.emerging_issues == []
    assert result.bottlenecks == []
    assert result.workflow_variants == []