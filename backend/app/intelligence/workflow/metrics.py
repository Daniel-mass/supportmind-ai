from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.workflow.event_builder import (
    WorkflowInterval,
)


class WorkflowMetrics(BaseModel):
    """
    Aggregated workflow metrics for one ticket.

    Metrics are derived from canonical WorkflowInterval records.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    ticket_id: str = Field(..., min_length=1)

    lifecycle_seconds: float = Field(..., ge=0)
    active_seconds: float = Field(..., ge=0)
    waiting_seconds: float = Field(..., ge=0)

    status_durations: dict[str, float] = Field(
        default_factory=dict
    )

    interval_count: int = Field(..., ge=0)
    transition_count: int = Field(..., ge=0)
    unique_status_count: int = Field(..., ge=0)
    unique_assignee_count: int = Field(..., ge=0)


WAITING_STATUSES = {
    "waiting",
    "pending",
    "pending_customer_approval",
}


def calculate_workflow_metrics(
    intervals: list[WorkflowInterval],
) -> list[WorkflowMetrics]:
    """
    Aggregate workflow intervals into ticket-level metrics.

    Waiting time is defined using a conservative set of explicit
    waiting/pending statuses. All other interval durations are
    considered active time for this first deterministic version.
    """

    if not intervals:
        return []

    grouped: dict[str, list[WorkflowInterval]] = defaultdict(list)

    for interval in intervals:
        grouped[interval.ticket_id].append(interval)

    results: list[WorkflowMetrics] = []

    for ticket_id, ticket_intervals in grouped.items():

        ordered = sorted(
            ticket_intervals,
            key=lambda item: (
                item.start_timestamp,
                item.interval_id,
            ),
        )

        status_durations: dict[str, float] = defaultdict(float)

        lifecycle_seconds = sum(
            interval.duration_seconds
            for interval in ordered
        )

        waiting_seconds = 0.0

        unique_assignees: set[str] = set()
        unique_statuses: set[str] = set()

        for interval in ordered:
            duration = interval.duration_seconds

            if interval.status:
                status = interval.status.strip()
                status_key = status.lower()

                unique_statuses.add(status)

                status_durations[status] += duration

                if status_key in WAITING_STATUSES:
                    waiting_seconds += duration

            if interval.assignee:
                unique_assignees.add(interval.assignee)

        active_seconds = max(
            lifecycle_seconds - waiting_seconds,
            0.0,
        )

        results.append(
            WorkflowMetrics(
                ticket_id=ticket_id,
                lifecycle_seconds=lifecycle_seconds,
                active_seconds=active_seconds,
                waiting_seconds=waiting_seconds,
                status_durations=dict(
                    status_durations
                ),
                interval_count=len(ordered),
                transition_count=len(ordered),
                unique_status_count=len(unique_statuses),
                unique_assignee_count=len(unique_assignees),
            )
        )

    return results
