from backend.app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
)
from backend.app.intelligence.analytics.organization import (
    OrganizationAnalytics,
)
from backend.app.intelligence.analytics.team_analytics import (
    TeamAnalytics,
)
from backend.app.intelligence.schemas.overview import (
    DistributionItem,
)
from backend.app.intelligence.services.organization_intelligence import (
    OrganizationIntelligence,
    build_organization_intelligence,
)
from backend.app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
)
from backend.app.intelligence.workflow.variants import (
    WorkflowVariant,
)


def make_organization() -> OrganizationAnalytics:
    return OrganizationAnalytics(
        agent_count=2,
        active_agent_count=2,
        team_count=1,
        total_ticket_count=10,
        open_ticket_count=6,
        closed_ticket_count=4,
        high_priority_ticket_count=2,
        waiting_ticket_count=3,
        total_lifecycle_seconds=10000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=2000,
        average_waiting_seconds=200,
        total_transition_count=25,
        total_handoff_count=5,
        priority_distribution=[
            DistributionItem(
                label="Medium",
                count=8,
            ),
            DistributionItem(
                label="High",
                count=2,
            ),
        ],
        status_distribution=[
            DistributionItem(
                label="open",
                count=6,
            ),
            DistributionItem(
                label="closed",
                count=4,
            ),
        ],
        team_workload=[
            DistributionItem(
                label="Team-A",
                count=10,
            ),
        ],
        emerging_issues=[
            DistributionItem(
                label="Login failures",
                count=4,
            ),
        ],
        bottlenecks=[],
        workflow_variants=[],
    )


def make_agent_analytics() -> AgentAnalytics:
    return AgentAnalytics(
        agent_id="agent-001",
        name="Agent One",
        team="Team-A",
        active=True,
        assigned_ticket_count=5,
        open_ticket_count=3,
        closed_ticket_count=2,
        high_priority_ticket_count=1,
        waiting_ticket_count=1,
        total_lifecycle_seconds=5000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=1000,
        average_waiting_seconds=200,
        total_transition_count=12,
        incoming_handoff_count=2,
        outgoing_handoff_count=1,
        unique_project_count=1,
        unique_issue_type_count=2,
    )


def make_team_analytics() -> TeamAnalytics:
    return TeamAnalytics(
        team="Team-A",
        agent_count=2,
        active_agent_count=2,
        assigned_ticket_count=10,
        open_ticket_count=6,
        closed_ticket_count=4,
        high_priority_ticket_count=2,
        waiting_ticket_count=3,
        total_lifecycle_seconds=10000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=2000,
        average_waiting_seconds=200,
        total_transition_count=25,
        incoming_handoff_count=5,
        outgoing_handoff_count=5,
    )


def make_bottleneck() -> BottleneckCandidate:
    return BottleneckCandidate(
        status="waiting",
        interval_count=10,
        ticket_count=5,
        total_seconds=2000,
        average_seconds=400,
        volume_share=0.5,
        duration_share=0.5,
        recurrence_rate=0.5,
        bottleneck_score=0.125,
    )


def make_variant() -> WorkflowVariant:
    return WorkflowVariant(
        variant_id="variant-001",
        path=[
            "in_progress",
            "waiting",
            "closed",
        ],
        path_string=(
            "in_progress -> waiting -> closed"
        ),
        ticket_count=5,
        percentage=50.0,
    )


def test_build_organization_intelligence_returns_expected_model():
    result = build_organization_intelligence(
        organization=make_organization(),
    )

    assert isinstance(
        result,
        OrganizationIntelligence,
    )


def test_organization_is_preserved():
    organization = make_organization()

    result = build_organization_intelligence(
        organization=organization,
    )

    assert result.organization == organization


def test_agent_analytics_are_preserved():
    agent = make_agent_analytics()

    result = build_organization_intelligence(
        organization=make_organization(),
        agent_analytics=[agent],
    )

    assert result.agent_analytics == [agent]


def test_team_analytics_are_preserved():
    team = make_team_analytics()

    result = build_organization_intelligence(
        organization=make_organization(),
        team_analytics=[team],
    )

    assert result.team_analytics == [team]


def test_bottlenecks_are_preserved():
    bottleneck = make_bottleneck()

    result = build_organization_intelligence(
        organization=make_organization(),
        bottlenecks=[bottleneck],
    )

    assert result.bottlenecks == [
        bottleneck
    ]


def test_workflow_variants_are_preserved():
    variant = make_variant()

    result = build_organization_intelligence(
        organization=make_organization(),
        workflow_variants=[variant],
    )

    assert result.workflow_variants == [
        variant
    ]


def test_emerging_issues_are_preserved():
    issue = DistributionItem(
        label="Login failures",
        count=4,
    )

    result = build_organization_intelligence(
        organization=make_organization(),
        emerging_issues=[issue],
    )

    assert result.emerging_issues == [
        issue
    ]


def test_missing_optional_data_returns_empty_lists():
    result = build_organization_intelligence(
        organization=make_organization(),
    )

    assert result.agent_analytics == []
    assert result.team_analytics == []
    assert result.bottlenecks == []
    assert result.workflow_variants == []
    assert result.emerging_issues == []


def test_full_organization_intelligence_composition():
    agent = make_agent_analytics()
    team = make_team_analytics()
    bottleneck = make_bottleneck()
    variant = make_variant()

    issue = DistributionItem(
        label="Login failures",
        count=4,
    )

    result = build_organization_intelligence(
        organization=make_organization(),
        agent_analytics=[agent],
        team_analytics=[team],
        bottlenecks=[bottleneck],
        workflow_variants=[variant],
        emerging_issues=[issue],
    )

    assert result.organization.total_ticket_count == 10
    assert result.agent_analytics == [agent]
    assert result.team_analytics == [team]
    assert result.bottlenecks == [bottleneck]
    assert result.workflow_variants == [variant]
    assert result.emerging_issues == [issue]