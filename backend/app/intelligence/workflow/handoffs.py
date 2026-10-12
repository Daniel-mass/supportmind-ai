from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.workflow.event_builder import (
    WorkflowInterval,
)


class HandoffRecord(BaseModel):
    """
    Dashboard-ready record describing one agent handoff.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    handoff_id: str = Field(..., min_length=1)

    ticket_id: str = Field(..., min_length=1)

    from_agent: str = Field(..., min_length=1)
    to_agent: str = Field(..., min_length=1)

    handoff_timestamp: object

    from_team: str | None = None
    to_team: str | None = None

    status_before: str | None = None
    status_after: str | None = None

    time_before_handoff_seconds: float = Field(
        ...,
        ge=0,
    )

    time_after_handoff_seconds: float = Field(
        ...,
        ge=0,
    )


def identify_handoffs(
    intervals: list[WorkflowInterval],
) -> list[HandoffRecord]:
    """
    Identify agent handoffs from consecutive workflow intervals.

    A handoff exists when two consecutive intervals belonging
    to the same ticket have different, non-null assignees.
    """

    if not intervals:
        return []

    grouped: dict[str, list[WorkflowInterval]] = {}

    for interval in intervals:
        grouped.setdefault(
            interval.ticket_id,
            [],
        ).append(interval)

    handoffs: list[HandoffRecord] = []

    for ticket_id, ticket_intervals in grouped.items():

        ordered = sorted(
            ticket_intervals,
            key=lambda item: (
                item.start_timestamp,
                item.interval_id,
            ),
        )

        for previous, current in zip(
            ordered,
            ordered[1:],
        ):
            if not previous.assignee:
                continue

            if not current.assignee:
                continue

            if previous.assignee == current.assignee:
                continue

            handoff = HandoffRecord(
                handoff_id=(
                    f"{previous.end_event_id}"
                    f"__handoff__"
                    f"{current.start_event_id}"
                ),
                ticket_id=ticket_id,
                from_agent=previous.assignee,
                to_agent=current.assignee,
                handoff_timestamp=current.start_timestamp,
                from_team=previous.team,
                to_team=current.team,
                status_before=previous.status,
                status_after=current.status,
                time_before_handoff_seconds=(
                    previous.duration_seconds
                ),
                time_after_handoff_seconds=(
                    current.duration_seconds
                ),
            )

            handoffs.append(handoff)

    return handoffs
