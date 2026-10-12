from backend.app.intelligence.analytics.agent_analytics import AgentAnalytics
from backend.app.intelligence.analytics.team_analytics import (
    TeamAnalytics,
    build_team_analytics,
)
from backend.app.intelligence.schemas.agent import Agent


def make_agent(
    *,
    agent_id: str,
    name: str,
    team: str,
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


def make_analytics(
    *,
    agent_id: str,
    team: str,
    assigned: int,
    open_tickets: int,
    closed_tickets: int,
    high_priority: int,
    waiting: int,
    lifecycle: float,
    waiting_seconds: float,
    transitions: int,
    incoming: int,
    outgoing: int,
) -> AgentAnalytics:
    average_lifecycle = (
        lifecycle / assigned if assigned else 0.0
    )

    average_waiting = (
        waiting_seconds / assigned if assigned else 0.0
    )

    return AgentAnalytics(
        agent_id=agent_id,
        name=agent_id,
        team=team,
        active=True,
        assigned_ticket_count=assigned,
        open_ticket_count=open_tickets,
        closed_ticket_count=closed_tickets,
        high_priority_ticket_count=high_priority,
        waiting_ticket_count=waiting,
        total_lifecycle_seconds=lifecycle,
        average_lifecycle_seconds=average_lifecycle,
        total_waiting_seconds=waiting_seconds,
        average_waiting_seconds=average_waiting,
        total_transition_count=transitions,
        incoming_handoff_count=incoming,
        outgoing_handoff_count=outgoing,
        unique_project_count=1,
        unique_issue_type_count=2,
    )


def test_build_team_analytics_returns_expected_model():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        )
    ]

    analytics = [
        make_analytics(
            agent_id="agent-001",
            team="Team-A",
            assigned=5,
            open_tickets=3,
            closed_tickets=2,
            high_priority=1,
            waiting=1,
            lifecycle=5000,
            waiting_seconds=1000,
            transitions=8,
            incoming=2,
            outgoing=1,
        )
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=analytics,
    )

    assert isinstance(result, TeamAnalytics)


def test_only_requested_team_agents_are_counted():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        ),
        make_agent(
            agent_id="agent-002",
            name="Agent Two",
            team="Team-B",
        ),
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=[],
    )

    assert result.agent_count == 1


def test_active_agent_count():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
            active=True,
        ),
        make_agent(
            agent_id="agent-002",
            name="Agent Two",
            team="Team-A",
            active=False,
        ),
        make_agent(
            agent_id="agent-003",
            name="Agent Three",
            team="Team-A",
            active=True,
        ),
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=[],
    )

    assert result.agent_count == 3
    assert result.active_agent_count == 2


def test_ticket_counts_are_aggregated():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        ),
        make_agent(
            agent_id="agent-002",
            name="Agent Two",
            team="Team-A",
        ),
    ]

    analytics = [
        make_analytics(
            agent_id="agent-001",
            team="Team-A",
            assigned=10,
            open_tickets=6,
            closed_tickets=4,
            high_priority=2,
            waiting=3,
            lifecycle=10000,
            waiting_seconds=2000,
            transitions=10,
            incoming=3,
            outgoing=2,
        ),
        make_analytics(
            agent_id="agent-002",
            team="Team-A",
            assigned=5,
            open_tickets=2,
            closed_tickets=3,
            high_priority=1,
            waiting=1,
            lifecycle=5000,
            waiting_seconds=1000,
            transitions=6,
            incoming=1,
            outgoing=4,
        ),
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=analytics,
    )

    assert result.assigned_ticket_count == 15
    assert result.open_ticket_count == 8
    assert result.closed_ticket_count == 7
    assert result.high_priority_ticket_count == 3
    assert result.waiting_ticket_count == 4


def test_time_aggregates_use_team_totals():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        ),
        make_agent(
            agent_id="agent-002",
            name="Agent Two",
            team="Team-A",
        ),
    ]

    analytics = [
        make_analytics(
            agent_id="agent-001",
            team="Team-A",
            assigned=10,
            open_tickets=5,
            closed_tickets=5,
            high_priority=1,
            waiting=2,
            lifecycle=10000,
            waiting_seconds=2000,
            transitions=5,
            incoming=1,
            outgoing=1,
        ),
        make_analytics(
            agent_id="agent-002",
            team="Team-A",
            assigned=5,
            open_tickets=3,
            closed_tickets=2,
            high_priority=1,
            waiting=1,
            lifecycle=5000,
            waiting_seconds=1000,
            transitions=3,
            incoming=0,
            outgoing=2,
        ),
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=analytics,
    )

    assert result.total_lifecycle_seconds == 15000
    assert result.average_lifecycle_seconds == 1000

    assert result.total_waiting_seconds == 3000
    assert result.average_waiting_seconds == 200

    assert result.total_transition_count == 8


def test_handoffs_are_aggregated():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        ),
        make_agent(
            agent_id="agent-002",
            name="Agent Two",
            team="Team-A",
        ),
    ]

    analytics = [
        make_analytics(
            agent_id="agent-001",
            team="Team-A",
            assigned=3,
            open_tickets=2,
            closed_tickets=1,
            high_priority=0,
            waiting=1,
            lifecycle=3000,
            waiting_seconds=500,
            transitions=3,
            incoming=4,
            outgoing=2,
        ),
        make_analytics(
            agent_id="agent-002",
            team="Team-A",
            assigned=2,
            open_tickets=1,
            closed_tickets=1,
            high_priority=1,
            waiting=0,
            lifecycle=2000,
            waiting_seconds=0,
            transitions=2,
            incoming=1,
            outgoing=3,
        ),
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=analytics,
    )

    assert result.incoming_handoff_count == 5
    assert result.outgoing_handoff_count == 5


def test_analytics_from_other_team_are_excluded():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        ),
        make_agent(
            agent_id="agent-002",
            name="Agent Two",
            team="Team-B",
        ),
    ]

    analytics = [
        make_analytics(
            agent_id="agent-001",
            team="Team-A",
            assigned=5,
            open_tickets=3,
            closed_tickets=2,
            high_priority=1,
            waiting=1,
            lifecycle=5000,
            waiting_seconds=500,
            transitions=4,
            incoming=2,
            outgoing=1,
        ),
        make_analytics(
            agent_id="agent-002",
            team="Team-B",
            assigned=100,
            open_tickets=50,
            closed_tickets=50,
            high_priority=20,
            waiting=10,
            lifecycle=100000,
            waiting_seconds=10000,
            transitions=100,
            incoming=50,
            outgoing=50,
        ),
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=analytics,
    )

    assert result.assigned_ticket_count == 5
    assert result.total_lifecycle_seconds == 5000


def test_zero_tickets_produce_zero_average_times():
    agents = [
        make_agent(
            agent_id="agent-001",
            name="Agent One",
            team="Team-A",
        )
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=[],
    )

    assert result.assigned_ticket_count == 0
    assert result.average_lifecycle_seconds == 0.0
    assert result.average_waiting_seconds == 0.0


def test_agent_with_analytics_but_without_agent_entity_is_still_aggregated():
    agents = []

    analytics = [
        make_analytics(
            agent_id="agent-001",
            team="Team-A",
            assigned=4,
            open_tickets=3,
            closed_tickets=1,
            high_priority=1,
            waiting=1,
            lifecycle=4000,
            waiting_seconds=800,
            transitions=4,
            incoming=1,
            outgoing=2,
        )
    ]

    result = build_team_analytics(
        team="Team-A",
        agents=agents,
        agent_analytics=analytics,
    )

    assert result.assigned_ticket_count == 4
    assert result.total_lifecycle_seconds == 4000
    assert result.agent_count == 0