from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.workflow.event_builder import (
    WorkflowInterval,
)


class WorkflowVariant(BaseModel):
    """
    Dashboard-ready workflow path variant.

    A variant represents one ordered sequence of observed
    workflow statuses for a ticket.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    variant_id: str = Field(..., min_length=1)

    path: list[str] = Field(..., min_length=1)

    path_string: str = Field(..., min_length=1)

    ticket_count: int = Field(..., ge=0)

    percentage: float = Field(
        ...,
        ge=0,
        le=100,
    )


def identify_workflow_variants(
    intervals: list[WorkflowInterval],
) -> list[WorkflowVariant]:
    """
    Identify recurring workflow status-path variants.

    Each ticket contributes one ordered status path.
    Consecutive duplicate statuses are collapsed so that
    repeated observations of the same state do not create
    artificial path complexity.
    """

    if not intervals:
        return []

    grouped: dict[str, list[WorkflowInterval]] = defaultdict(list)

    for interval in intervals:
        grouped[interval.ticket_id].append(interval)

    variant_counts: dict[tuple[str, ...], int] = defaultdict(int)

    for ticket_id, ticket_intervals in grouped.items():

        ordered = sorted(
            ticket_intervals,
            key=lambda item: (
                item.start_timestamp,
                item.interval_id,
            ),
        )

        path: list[str] = []

        for interval in ordered:
            if not interval.status:
                continue

            status = interval.status.strip()

            if not status:
                continue

            if not path or path[-1] != status:
                path.append(status)

        if path:
            variant_counts[tuple(path)] += 1

    total_tickets = sum(variant_counts.values())

    variants: list[WorkflowVariant] = []

    for index, (path, ticket_count) in enumerate(
        sorted(
            variant_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ),
        start=1,
    ):
        percentage = (
            ticket_count / total_tickets * 100
            if total_tickets
            else 0.0
        )

        path_string = " â†’ ".join(path)

        variants.append(
            WorkflowVariant(
                variant_id=f"VAR-{index:04d}",
                path=list(path),
                path_string=path_string,
                ticket_count=ticket_count,
                percentage=percentage,
            )
        )

    return variants
