#ticket_intelligence.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
)
from app.intelligence.schemas.ticket import Ticket
from app.intelligence.schemas.workflow import (
    WorkflowEvent,
)
from app.intelligence.workflow.handoffs import (
    HandoffRecord,
)
from app.intelligence.workflow.metrics import (
    WorkflowMetrics,
)


class TicketIntelligence(BaseModel):
    """
    Unified intelligence response for a single support ticket.

    This is an orchestration contract for API/dashboard consumers.
    It does not perform new ML or analytics calculations.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    ticket: Ticket
    analytics: TicketAnalytics

    workflow_metrics: WorkflowMetrics | None = None

    workflow_events: list[
        WorkflowEvent
    ]

    handoffs: list[
        HandoffRecord
    ]


def build_ticket_intelligence(
    *,
    ticket: Ticket,
    analytics: TicketAnalytics,
    workflow_metrics: WorkflowMetrics | None = None,
    workflow_events: list[
        WorkflowEvent
    ] | None = None,
    handoffs: list[
        HandoffRecord
    ] | None = None,
) -> TicketIntelligence:
    """
    Compose all already-computed intelligence for one ticket.
    """

    return TicketIntelligence(
        ticket=ticket,
        analytics=analytics,
        workflow_metrics=workflow_metrics,
        workflow_events=list(
            workflow_events or []
        ),
        handoffs=list(
            handoffs or []
        ),
    )
