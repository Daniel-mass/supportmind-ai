from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.analytics.agent_analytics import AgentAnalytics
from app.intelligence.analytics.team_analytics import TeamAnalytics
from app.intelligence.analytics.ticket_analytics import TicketAnalytics
from app.intelligence.schemas.agent import Agent
from app.intelligence.schemas.overview import DistributionItem
from app.intelligence.workflow.bottlenecks import BottleneckCandidate
from app.intelligence.workflow.variants import WorkflowVariant


class OrganizationAnalytics(BaseModel):
    """
    Dashboard-ready operational analytics for the support organization.

    This model aggregates already-computed analytics and workflow signals.
    It does not execute ML, topic modeling, retrieval, or graph logic.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_count: int = Field(ge=0)
    active_agent_count: int = Field(ge=0)
    team_count: int = Field(ge=0)

    total_ticket_count: int = Field(ge=0)
    open_ticket_count: int = Field(ge=0)
    closed_ticket_count: int = Field(ge=0)

    high_priority_ticket_count: int = Field(ge=0)
    waiting_ticket_count: int = Field(ge=0)

    total_lifecycle_seconds: float = Field(ge=0)
    average_lifecycle_seconds: float = Field(ge=0)

    total_waiting_seconds: float = Field(ge=0)
    average_waiting_seconds: float = Field(ge=0)

    total_transition_count: int = Field(ge=0)
    total_handoff_count: int = Field(ge=0)

    priority_distribution: list[DistributionItem]
    status_distribution: list[DistributionItem]

    team_workload: list[DistributionItem]
    emerging_issues: list[DistributionItem]

    bottlenecks: list[BottleneckCandidate]
    workflow_variants: list[WorkflowVariant]


def _distribution(
    counter: Counter[str],
) -> list[DistributionItem]:
    """
    Convert a Counter into deterministic dashboard distribution items.
    """
    return [
        DistributionItem(
            label=label,
            count=count,
        )
        for label, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def build_organization_analytics(
    agents: Iterable[Agent],
    ticket_analytics: Iterable[TicketAnalytics],
    agent_analytics: Iterable[AgentAnalytics],
    team_analytics: Iterable[TeamAnalytics],
    bottlenecks: Iterable[BottleneckCandidate] | None = None,
    workflow_variants: Iterable[WorkflowVariant] | None = None,
    emerging_issues: Iterable[DistributionItem] | None = None,
) -> OrganizationAnalytics:
    """
    Aggregate organization-level operational intelligence.

    TicketAnalytics is treated as the source of truth for organization-wide
    ticket totals, lifecycle metrics, waiting metrics, and distributions.

    AgentAnalytics and TeamAnalytics provide organizational structure and
    workload context.

    Bottlenecks, workflow variants, and emerging issues are passed through
    as already-computed operational signals.
    """

    agent_list = list(agents)
    ticket_list = list(ticket_analytics)
    agent_analytics_list = list(agent_analytics)
    team_analytics_list = list(team_analytics)

    bottleneck_list = list(bottlenecks or [])
    workflow_variant_list = list(workflow_variants or [])
    emerging_issue_list = list(emerging_issues or [])

    agent_count = len(agent_list)

    active_agent_count = sum(
        1
        for agent in agent_list
        if agent.active
    )

    team_names = {
        agent.team.strip()
        for agent in agent_list
        if agent.team is not None
        and agent.team.strip()
    }

    total_ticket_count = len(ticket_list)

    closed_ticket_count = sum(
        1
        for ticket in ticket_list
        if ticket.status.strip().lower()
        in {"closed", "resolved", "done"}
    )

    open_ticket_count = max(
        total_ticket_count - closed_ticket_count,
        0,
    )

    high_priority_ticket_count = sum(
        1
        for ticket in ticket_list
        if ticket.is_high_priority
    )

    waiting_ticket_count = sum(
        1
        for ticket in ticket_list
        if ticket.is_waiting
    )

    total_lifecycle_seconds = sum(
        max(ticket.lifecycle_seconds, 0.0)
        for ticket in ticket_list
    )

    total_waiting_seconds = sum(
        max(ticket.waiting_seconds, 0.0)
        for ticket in ticket_list
    )

    total_transition_count = sum(
        max(ticket.transition_count, 0)
        for ticket in ticket_list
    )

    total_handoff_count = sum(
        max(ticket.handoff_count, 0)
        for ticket in ticket_list
    )

    if total_ticket_count > 0:
        average_lifecycle_seconds = (
            total_lifecycle_seconds / total_ticket_count
        )

        average_waiting_seconds = (
            total_waiting_seconds / total_ticket_count
        )
    else:
        average_lifecycle_seconds = 0.0
        average_waiting_seconds = 0.0

    priority_counter = Counter(
        ticket.priority
        for ticket in ticket_list
    )

    status_counter = Counter(
        ticket.status
        for ticket in ticket_list
    )

    priority_distribution = _distribution(priority_counter)
    status_distribution = _distribution(status_counter)

    team_workload = [
        DistributionItem(
            label=team.team,
            count=team.assigned_ticket_count,
        )
        for team in sorted(
            team_analytics_list,
            key=lambda item: (
                -item.assigned_ticket_count,
                item.team,
            ),
        )
    ]

    sorted_bottlenecks = sorted(
        bottleneck_list,
        key=lambda item: (
            -item.bottleneck_score,
            -item.total_seconds,
            item.status,
        ),
    )

    sorted_variants = sorted(
        workflow_variant_list,
        key=lambda item: (
            -item.ticket_count,
            -item.percentage,
            item.variant_id,
        ),
    )

    # Keep the materialized list so aggregation inputs are explicitly
    # consumed and validated even though lower-level analytics already
    # contain the needed values.
    _ = agent_analytics_list

    return OrganizationAnalytics(
        agent_count=agent_count,
        active_agent_count=active_agent_count,
        team_count=len(team_names),
        total_ticket_count=total_ticket_count,
        open_ticket_count=open_ticket_count,
        closed_ticket_count=closed_ticket_count,
        high_priority_ticket_count=high_priority_ticket_count,
        waiting_ticket_count=waiting_ticket_count,
        total_lifecycle_seconds=total_lifecycle_seconds,
        average_lifecycle_seconds=average_lifecycle_seconds,
        total_waiting_seconds=total_waiting_seconds,
        average_waiting_seconds=average_waiting_seconds,
        total_transition_count=total_transition_count,
        total_handoff_count=total_handoff_count,
        priority_distribution=priority_distribution,
        status_distribution=status_distribution,
        team_workload=team_workload,
        emerging_issues=emerging_issue_list,
        bottlenecks=sorted_bottlenecks,
        workflow_variants=sorted_variants,
    )
