from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.schemas.agent import Agent
from app.intelligence.schemas.ticket import Ticket
from app.intelligence.workflow.handoffs import HandoffRecord
from app.intelligence.workflow.metrics import WorkflowMetrics


HIGH_PRIORITY_VALUES = {
    "high",
    "highest",
    "blocker",
}

CLOSED_STATUSES = {
    "closed",
    "resolved",
    "done",
}


class AgentAnalytics(BaseModel):
    """
    Dashboard-ready operational analytics for a support agent.

    These metrics describe workload and workflow activity.
    They are not a predictive performance score.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_id: str
    name: str | None = None
    team: str | None = None
    active: bool

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

    unique_project_count: int = Field(ge=0)
    unique_issue_type_count: int = Field(ge=0)


def _enum_value(value: object) -> str:
    """
    Return the raw value for Enum-like objects.
    """
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def build_agent_analytics(
    agent: Agent,
    tickets: Iterable[Ticket],
    workflow_metrics_by_ticket: Mapping[str, WorkflowMetrics],
    handoffs: Iterable[HandoffRecord] | None = None,
) -> AgentAnalytics:
    """
    Build dashboard-ready operational analytics for one support agent.

    Only tickets currently assigned to the supplied agent are included
    in workload and workflow aggregates.

    Parameters
    ----------
    agent:
        Canonical support-agent entity.

    tickets:
        Canonical ticket collection.

    workflow_metrics_by_ticket:
        Mapping from ticket_id to precomputed WorkflowMetrics.

    handoffs:
        Workflow handoff records used to count incoming and outgoing
        handoffs for this agent.

    Returns
    -------
    AgentAnalytics
        Structured workload and workflow analytics.
    """

    assigned_tickets = [
        ticket
        for ticket in tickets
        if ticket.assignee is not None
        and str(ticket.assignee).strip() == agent.agent_id
    ]

    assigned_ticket_count = len(assigned_tickets)

    closed_ticket_count = 0
    high_priority_ticket_count = 0
    waiting_ticket_count = 0

    total_lifecycle_seconds = 0.0
    total_waiting_seconds = 0.0
    total_transition_count = 0

    projects: set[str] = set()
    issue_types: set[str] = set()

    for ticket in assigned_tickets:
        status = _enum_value(ticket.status).strip().lower()
        priority = _enum_value(ticket.priority).strip().lower()

        if status in CLOSED_STATUSES:
            closed_ticket_count += 1

        if priority in HIGH_PRIORITY_VALUES:
            high_priority_ticket_count += 1

        if status in {"waiting", "pending", "pending_customer_approval"}:
            waiting_ticket_count += 1

        projects.add(ticket.project)
        issue_types.add(ticket.issue_type)

        metrics = workflow_metrics_by_ticket.get(ticket.ticket_id)

        if metrics is not None:
            total_lifecycle_seconds += max(
                float(metrics.lifecycle_seconds),
                0.0,
            )

            total_waiting_seconds += max(
                float(metrics.waiting_seconds),
                0.0,
            )

            total_transition_count += max(
                int(metrics.transition_count),
                0,
            )

    open_ticket_count = max(
        assigned_ticket_count - closed_ticket_count,
        0,
    )

    if assigned_ticket_count > 0:
        average_lifecycle_seconds = (
            total_lifecycle_seconds / assigned_ticket_count
        )
        average_waiting_seconds = (
            total_waiting_seconds / assigned_ticket_count
        )
    else:
        average_lifecycle_seconds = 0.0
        average_waiting_seconds = 0.0

    incoming_handoff_count = 0
    outgoing_handoff_count = 0

    if handoffs is not None:
        for handoff in handoffs:
            if handoff.to_agent == agent.agent_id:
                incoming_handoff_count += 1

            if handoff.from_agent == agent.agent_id:
                outgoing_handoff_count += 1

    return AgentAnalytics(
        agent_id=agent.agent_id,
        name=agent.name,
        team=agent.team,
        active=agent.active,
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
        unique_project_count=len(projects),
        unique_issue_type_count=len(issue_types),
    )
