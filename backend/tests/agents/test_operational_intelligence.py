#
from __future__ import annotations

import pytest

from backend.app.agents.escalation_agent.operational_intelligence import (
    EscalationOperationalIntelligence,
)
from backend.app.agents.escalation_agent.contracts.state import (
    EscalationSignal,
)
from backend.app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
)
from backend.app.intelligence.analytics.organization import (
    OrganizationAnalytics,
)
from backend.app.intelligence.analytics.team_analytics import (
    TeamAnalytics,
)
from backend.app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
)
from backend.app.intelligence.schemas.overview import DistributionItem
from backend.app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
)
from backend.app.intelligence.workflow.variants import (
    WorkflowVariant,
)


def signal_names(signals: list[EscalationSignal]) -> set[str]:
    return {signal.name for signal in signals}


def test_ticket_analytics_are_converted_to_signals() -> None:
    analytics = TicketAnalytics(
        ticket_id="TICKET-001",
        project="Helpdesk",
        issue_num=101,
        issue_type="Login",
        priority="high",
        status="waiting",
        lifecycle_seconds=1000.0,
        active_seconds=300.0,
        waiting_seconds=700.0,
        active_ratio=0.3,
        waiting_ratio=0.7,
        transition_count=4,
        unique_status_count=3,
        unique_assignee_count=2,
        handoff_count=1,
        is_high_priority=True,
        is_waiting=True,
    )

    signals = (
        EscalationOperationalIntelligence.from_ticket_analytics(
            analytics
        )
    )

    assert len(signals) == 15
    assert signal_names(signals) == {
        "ticket.project",
        "ticket.issue_type",
        "ticket.priority",
        "ticket.status",
        "ticket.lifecycle_seconds",
        "ticket.active_seconds",
        "ticket.waiting_seconds",
        "ticket.active_ratio",
        "ticket.waiting_ratio",
        "ticket.transition_count",
        "ticket.unique_status_count",
        "ticket.unique_assignee_count",
        "ticket.handoff_count",
        "ticket.is_high_priority",
        "ticket.is_waiting",
    }

    waiting_signal = next(
        signal
        for signal in signals
        if signal.name == "ticket.waiting_seconds"
    )

    assert waiting_signal.value == "700.0"
    assert waiting_signal.source == "ticket_analytics"
    assert waiting_signal.metadata["ticket_id"] == "TICKET-001"


def test_agent_analytics_are_converted_to_signals() -> None:
    analytics = AgentAnalytics(
        agent_id="agent-001",
        name="Alice",
        team="helpdesk",
        active=True,
        assigned_ticket_count=10,
        open_ticket_count=6,
        closed_ticket_count=4,
        high_priority_ticket_count=2,
        waiting_ticket_count=3,
        total_lifecycle_seconds=10000.0,
        average_lifecycle_seconds=1000.0,
        total_waiting_seconds=3000.0,
        average_waiting_seconds=300.0,
        total_transition_count=20,
        incoming_handoff_count=4,
        outgoing_handoff_count=2,
        unique_project_count=2,
        unique_issue_type_count=5,
    )

    signals = (
        EscalationOperationalIntelligence.from_agent_analytics(
            analytics
        )
    )

    assert len(signals) == 14
    assert "agent.assigned_ticket_count" in signal_names(signals)
    assert "agent.waiting_ticket_count" in signal_names(signals)

    assigned = next(
        signal
        for signal in signals
        if signal.name == "agent.assigned_ticket_count"
    )

    assert assigned.value == "10"
    assert assigned.source == "agent_analytics"
    assert assigned.metadata["agent_id"] == "agent-001"
    assert assigned.metadata["team"] == "helpdesk"


def test_team_analytics_are_converted_to_signals() -> None:
    analytics = TeamAnalytics(
        team="helpdesk",
        agent_count=5,
        active_agent_count=4,
        assigned_ticket_count=50,
        open_ticket_count=30,
        closed_ticket_count=20,
        high_priority_ticket_count=8,
        waiting_ticket_count=10,
        total_lifecycle_seconds=50000.0,
        average_lifecycle_seconds=1000.0,
        total_waiting_seconds=15000.0,
        average_waiting_seconds=300.0,
        total_transition_count=100,
        incoming_handoff_count=20,
        outgoing_handoff_count=15,
    )

    signals = (
        EscalationOperationalIntelligence.from_team_analytics(
            analytics
        )
    )

    assert len(signals) == 12
    assert "team.assigned_ticket_count" in signal_names(signals)
    assert "team.waiting_ticket_count" in signal_names(signals)

    waiting = next(
        signal
        for signal in signals
        if signal.name == "team.waiting_ticket_count"
    )

    assert waiting.value == "10"
    assert waiting.metadata["team"] == "helpdesk"


def test_organization_analytics_are_converted_to_signals() -> None:
    analytics = OrganizationAnalytics(
        agent_count=10,
        active_agent_count=8,
        team_count=3,
        total_ticket_count=100,
        open_ticket_count=60,
        closed_ticket_count=40,
        high_priority_ticket_count=15,
        waiting_ticket_count=20,
        total_lifecycle_seconds=100000.0,
        average_lifecycle_seconds=1000.0,
        total_waiting_seconds=30000.0,
        average_waiting_seconds=300.0,
        total_transition_count=250,
        total_handoff_count=80,
        priority_distribution=[
            DistributionItem(label="high", count=15)
        ],
        status_distribution=[
            DistributionItem(label="waiting", count=20)
        ],
        team_workload=[
            DistributionItem(label="helpdesk", count=60)
        ],
        emerging_issues=[],
        bottlenecks=[],
        workflow_variants=[],
    )

    signals = (
        EscalationOperationalIntelligence.from_organization_analytics(
            analytics
        )
    )

    assert len(signals) == 14
    assert "organization.total_ticket_count" in signal_names(signals)
    assert "organization.total_handoff_count" in signal_names(signals)

    total_tickets = next(
        signal
        for signal in signals
        if signal.name == "organization.total_ticket_count"
    )

    assert total_tickets.value == "100"
    assert total_tickets.source == "organization_analytics"


def test_bottlenecks_are_passed_through_without_recalculation() -> None:
    bottleneck = BottleneckCandidate(
        status="waiting",
        interval_count=20,
        ticket_count=15,
        total_seconds=5000.0,
        average_seconds=250.0,
        volume_share=0.2,
        duration_share=0.5,
        recurrence_rate=0.75,
        bottleneck_score=0.075,
    )

    signals = (
        EscalationOperationalIntelligence.from_bottlenecks(
            [bottleneck]
        )
    )

    assert len(signals) == 8

    score = next(
        signal
        for signal in signals
        if signal.name == "workflow.bottleneck.score"
    )

    assert score.value == "0.075"
    assert score.source == "bottleneck_analysis"
    assert score.metadata["status"] == "waiting"

    recurrence = next(
        signal
        for signal in signals
        if signal.name == "workflow.bottleneck.recurrence_rate"
    )

    assert recurrence.value == "0.75"


def test_workflow_variants_are_descriptive_signals() -> None:
    variant = WorkflowVariant(
        variant_id="VAR-0001",
        path=["open", "waiting", "resolved"],
        path_string="open → waiting → resolved",
        ticket_count=25,
        percentage=25.0,
    )

    signals = (
        EscalationOperationalIntelligence.from_workflow_variants(
            [variant]
        )
    )

    assert len(signals) == 3
    assert signal_names(signals) == {
        "workflow.variant.path",
        "workflow.variant.ticket_count",
        "workflow.variant.percentage",
    }

    path_signal = next(
        signal
        for signal in signals
        if signal.name == "workflow.variant.path"
    )

    assert path_signal.value == "open → waiting → resolved"
    assert path_signal.source == "workflow_variant_analysis"
    assert path_signal.metadata["variant_id"] == "VAR-0001"


def test_build_operational_signals_combines_all_sources() -> None:
    ticket = TicketAnalytics(
        ticket_id="TICKET-001",
        project="Helpdesk",
        issue_num=101,
        issue_type="Login",
        priority="high",
        status="waiting",
        lifecycle_seconds=1000.0,
        active_seconds=300.0,
        waiting_seconds=700.0,
        active_ratio=0.3,
        waiting_ratio=0.7,
        transition_count=4,
        unique_status_count=3,
        unique_assignee_count=2,
        handoff_count=1,
        is_high_priority=True,
        is_waiting=True,
    )

    agent = AgentAnalytics(
        agent_id="agent-001",
        name="Alice",
        team="helpdesk",
        active=True,
        assigned_ticket_count=10,
        open_ticket_count=6,
        closed_ticket_count=4,
        high_priority_ticket_count=2,
        waiting_ticket_count=3,
        total_lifecycle_seconds=10000.0,
        average_lifecycle_seconds=1000.0,
        total_waiting_seconds=3000.0,
        average_waiting_seconds=300.0,
        total_transition_count=20,
        incoming_handoff_count=4,
        outgoing_handoff_count=2,
        unique_project_count=2,
        unique_issue_type_count=5,
    )

    team = TeamAnalytics(
        team="helpdesk",
        agent_count=5,
        active_agent_count=4,
        assigned_ticket_count=50,
        open_ticket_count=30,
        closed_ticket_count=20,
        high_priority_ticket_count=8,
        waiting_ticket_count=10,
        total_lifecycle_seconds=50000.0,
        average_lifecycle_seconds=1000.0,
        total_waiting_seconds=15000.0,
        average_waiting_seconds=300.0,
        total_transition_count=100,
        incoming_handoff_count=20,
        outgoing_handoff_count=15,
    )

    organization = OrganizationAnalytics(
        agent_count=10,
        active_agent_count=8,
        team_count=3,
        total_ticket_count=100,
        open_ticket_count=60,
        closed_ticket_count=40,
        high_priority_ticket_count=15,
        waiting_ticket_count=20,
        total_lifecycle_seconds=100000.0,
        average_lifecycle_seconds=1000.0,
        total_waiting_seconds=30000.0,
        average_waiting_seconds=300.0,
        total_transition_count=250,
        total_handoff_count=80,
        priority_distribution=[],
        status_distribution=[],
        team_workload=[],
        emerging_issues=[],
        bottlenecks=[],
        workflow_variants=[],
    )

    bottleneck = BottleneckCandidate(
        status="waiting",
        interval_count=20,
        ticket_count=15,
        total_seconds=5000.0,
        average_seconds=250.0,
        volume_share=0.2,
        duration_share=0.5,
        recurrence_rate=0.75,
        bottleneck_score=0.075,
    )

    variant = WorkflowVariant(
        variant_id="VAR-0001",
        path=["open", "waiting", "resolved"],
        path_string="open → waiting → resolved",
        ticket_count=25,
        percentage=25.0,
    )

    signals = EscalationOperationalIntelligence.build_operational_signals(
        ticket_analytics=ticket,
        agent_analytics=agent,
        team_analytics=team,
        organization_analytics=organization,
        bottlenecks=[bottleneck],
        workflow_variants=[variant],
    )

    assert len(signals) == (
        15 + 14 + 12 + 14 + 8 + 3
    )

    sources = {signal.source for signal in signals}

    assert sources == {
        "ticket_analytics",
        "agent_analytics",
        "team_analytics",
        "organization_analytics",
        "bottleneck_analysis",
        "workflow_variant_analysis",
    }


def test_invalid_analytics_types_are_rejected() -> None:
    with pytest.raises(TypeError):
        EscalationOperationalIntelligence.from_ticket_analytics(
            object()  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        EscalationOperationalIntelligence.from_agent_analytics(
            object()  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        EscalationOperationalIntelligence.from_team_analytics(
            object()  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        EscalationOperationalIntelligence.from_organization_analytics(
            object()  # type: ignore[arg-type]
        )


def test_invalid_workflow_objects_are_rejected() -> None:
    with pytest.raises(TypeError):
        EscalationOperationalIntelligence.from_bottlenecks(
            [object()]  # type: ignore[list-item]
        )

    with pytest.raises(TypeError):
        EscalationOperationalIntelligence.from_workflow_variants(
            [object()]  # type: ignore[list-item]
        )


def test_empty_optional_sources_return_empty_signal_list() -> None:
    signals = (
        EscalationOperationalIntelligence.build_operational_signals()
    )

    assert signals == []