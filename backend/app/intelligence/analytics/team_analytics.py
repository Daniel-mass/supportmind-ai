from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.analytics.agent_analytics import AgentAnalytics
from app.intelligence.schemas.agent import Agent


class TeamAnalytics(BaseModel):
    """
    Dashboard-ready operational analytics for a support team.

    Metrics are aggregated from agent-level operational analytics.
    No performance ranking or predictive scoring is performed.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    team: str

    agent_count: int = Field(ge=0)
    active_agent_count: int = Field(ge=0)

    assigned_ticket_count: int = Field(ge=0)
    open_ticket_count: int = Field(ge=0)
    closed_ticket_count: int = Field(ge=0)

    high_priority_ticket_count: int = Field(ge=0)
    waiting_ticket_count: int = Field(ge=0)

    total_lifecycle_seconds: float = Field(ge=0)
    average_lifecycle_seconds: float = Field(ge=0)

    total_waiting_seconds: float = Field(ge=0)
    average_waiting_seconds: float = Field(ge=0)

    total_transition_count: int = Field(ge=0)

    incoming_handoff_count: int = Field(ge=0)
    outgoing_handoff_count: int = Field(ge=0)


def build_team_analytics(
    team: str,
    agents: Iterable[Agent],
    agent_analytics: Iterable[AgentAnalytics],
) -> TeamAnalytics:
    """
    Aggregate agent-level operational analytics for one team.

    Only agents belonging to the supplied team are included.

    Average time metrics are calculated from team-level totals divided
    by the team's assigned ticket count, avoiding an unweighted average
    of per-agent averages.
    """

    team_agents = [
        agent
        for agent in agents
        if agent.team is not None
        and agent.team.strip() == team
    ]

    team_agent_ids = {
        agent.agent_id
        for agent in team_agents
    }

    team_analytics = [
        analytics
        for analytics in agent_analytics
        if analytics.team == team
        or analytics.agent_id in team_agent_ids
    ]

    agent_count = len(team_agents)

    active_agent_count = sum(
        1
        for agent in team_agents
        if agent.active
    )

    assigned_ticket_count = sum(
        analytics.assigned_ticket_count
        for analytics in team_analytics
    )

    open_ticket_count = sum(
        analytics.open_ticket_count
        for analytics in team_analytics
    )

    closed_ticket_count = sum(
        analytics.closed_ticket_count
        for analytics in team_analytics
    )

    high_priority_ticket_count = sum(
        analytics.high_priority_ticket_count
        for analytics in team_analytics
    )

    waiting_ticket_count = sum(
        analytics.waiting_ticket_count
        for analytics in team_analytics
    )

    total_lifecycle_seconds = sum(
        analytics.total_lifecycle_seconds
        for analytics in team_analytics
    )

    total_waiting_seconds = sum(
        analytics.total_waiting_seconds
        for analytics in team_analytics
    )

    total_transition_count = sum(
        analytics.total_transition_count
        for analytics in team_analytics
    )

    incoming_handoff_count = sum(
        analytics.incoming_handoff_count
        for analytics in team_analytics
    )

    outgoing_handoff_count = sum(
        analytics.outgoing_handoff_count
        for analytics in team_analytics
    )

    if assigned_ticket_count > 0:
        average_lifecycle_seconds = (
            total_lifecycle_seconds
            / assigned_ticket_count
        )

        average_waiting_seconds = (
            total_waiting_seconds
            / assigned_ticket_count
        )
    else:
        average_lifecycle_seconds = 0.0
        average_waiting_seconds = 0.0

    return TeamAnalytics(
        team=team,
        agent_count=agent_count,
        active_agent_count=active_agent_count,
        assigned_ticket_count=assigned_ticket_count,
        open_ticket_count=open_ticket_count,
        closed_ticket_count=closed_ticket_count,
        high_priority_ticket_count=high_priority_ticket_count,
        waiting_ticket_count=waiting_ticket_count,
        total_lifecycle_seconds=total_lifecycle_seconds,
        average_lifecycle_seconds=average_lifecycle_seconds,
        total_waiting_seconds=total_waiting_seconds,
        average_waiting_seconds=average_waiting_seconds,
        total_transition_count=total_transition_count,
        incoming_handoff_count=incoming_handoff_count,
        outgoing_handoff_count=outgoing_handoff_count,
    )
