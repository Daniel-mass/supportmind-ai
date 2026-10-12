from backend.app.intelligence.analytics.organization import (
    OrganizationAnalytics,
)
from backend.app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
)
from backend.app.intelligence.schemas.overview import (
    DistributionItem,
)
from backend.app.intelligence.services.evidence_service import (
    build_evidence,
    build_organization_evidence,
    build_ticket_evidence,
)


def make_ticket_analytics() -> TicketAnalytics:
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


def make_organization_analytics() -> OrganizationAnalytics:
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
            )
        ],
        status_distribution=[
            DistributionItem(
                label="open",
                count=6,
            )
        ],
        team_workload=[
            DistributionItem(
                label="Team-A",
                count=10,
            )
        ],
        emerging_issues=[],
        bottlenecks=[],
        workflow_variants=[],
    )


def test_ticket_evidence_is_created():
    analytics = make_ticket_analytics()

    evidence = build_ticket_evidence(
        analytics
    )

    assert len(evidence) == 5

    assert all(
        item.ticket_id
        == "PROJ:100"
        for item in evidence
    )


def test_ticket_evidence_contains_expected_metrics():
    analytics = make_ticket_analytics()

    evidence = build_ticket_evidence(
        analytics
    )

    metric_names = {
        item.metric_name
        for item in evidence
    }

    assert metric_names == {
        "lifecycle_seconds",
        "waiting_seconds",
        "transition_count",
        "handoff_count",
        "priority",
    }


def test_ticket_evidence_is_traceable():
    analytics = make_ticket_analytics()

    evidence = build_ticket_evidence(
        analytics,
        source_dataset="helpdesk",
    )

    for item in evidence:
        assert (
            item.source_dataset
            == "helpdesk"
        )

        assert (
            len(item.source_fields)
            > 0
        )

        assert (
            item.metadata[
                "derived_from"
            ]
            == "TicketAnalytics"
        )


def test_organization_evidence_is_created():
    analytics = (
        make_organization_analytics()
    )

    evidence = (
        build_organization_evidence(
            analytics
        )
    )

    assert len(evidence) == 4

    assert all(
        item.scope == "organization"
        for item in evidence
    )


def test_organization_evidence_contains_expected_metrics():
    analytics = (
        make_organization_analytics()
    )

    evidence = (
        build_organization_evidence(
            analytics
        )
    )

    metric_names = {
        item.metric_name
        for item in evidence
    }

    assert metric_names == {
        "total_ticket_count",
        "open_ticket_count",
        "high_priority_ticket_count",
        "waiting_ticket_count",
    }


def test_combined_evidence_contains_ticket_and_organization_evidence():
    result = build_evidence(
        ticket_analytics=[
            make_ticket_analytics()
        ],
        organization_analytics=(
            make_organization_analytics()
        ),
    )

    assert len(result) == 9

    scopes = {
        item.scope
        for item in result
    }

    assert scopes == {
        "ticket",
        "organization",
    }


def test_empty_evidence_input_returns_empty_list():
    result = build_evidence()

    assert result == []