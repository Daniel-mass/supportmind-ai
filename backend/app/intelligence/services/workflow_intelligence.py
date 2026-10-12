#workflow_intelligence.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.intelligence.schemas.workflow import (
    WorkflowEvent,
)
from app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
)
from app.intelligence.workflow.handoffs import (
    HandoffRecord,
)
from app.intelligence.workflow.metrics import (
    WorkflowMetrics,
)
from app.intelligence.workflow.variants import (
    WorkflowVariant,
)


class WorkflowIntelligence(BaseModel):
    """
    Unified workflow intelligence response.

    This model composes already-computed workflow outputs.
    It does not perform new workflow calculations.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    ticket_id: str

    events: list[WorkflowEvent]
    metrics: WorkflowMetrics | None = None
    handoffs: list[HandoffRecord]
    bottlenecks: list[BottleneckCandidate]
    variants: list[WorkflowVariant]


def build_workflow_intelligence(
    *,
    ticket_id: str,
    events: list[WorkflowEvent] | None = None,
    metrics: WorkflowMetrics | None = None,
    handoffs: list[HandoffRecord] | None = None,
    bottlenecks: list[BottleneckCandidate] | None = None,
    variants: list[WorkflowVariant] | None = None,
) -> WorkflowIntelligence:
    """
    Compose previously computed workflow intelligence for a ticket.
    """

    return WorkflowIntelligence(
        ticket_id=ticket_id,
        events=list(events or []),
        metrics=metrics,
        handoffs=list(handoffs or []),
        bottlenecks=list(bottlenecks or []),
        variants=list(variants or []),
    )
