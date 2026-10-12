from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.schemas.workflow import WorkflowEvent


class WorkflowInterval(BaseModel):
    """
    Derived interval between two consecutive workflow events.

    The interval represents the amount of time attributable to
    the state/assignment represented by the starting event.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    interval_id: str = Field(..., min_length=1)

    ticket_id: str = Field(..., min_length=1)

    start_event_id: str = Field(..., min_length=1)
    end_event_id: str = Field(..., min_length=1)

    start_timestamp: datetime
    end_timestamp: datetime

    activity: str = Field(..., min_length=1)

    status: str | None = Field(default=None, min_length=1)
    assignee: str | None = Field(default=None, min_length=1)
    team: str | None = Field(default=None, min_length=1)

    duration_seconds: float = Field(..., ge=0)

    metadata: dict = Field(default_factory=dict)


def _event_sort_key(event: WorkflowEvent) -> tuple[str, datetime, str]:
    """
    Deterministic ordering for workflow events.

    Events are grouped by ticket, then ordered by timestamp,
    then event_id is used as a deterministic tie-breaker.
    """

    return (
        event.ticket_id,
        event.timestamp,
        event.event_id,
    )


def build_workflow_intervals(
    events: list[WorkflowEvent],
) -> list[WorkflowInterval]:
    """
    Build chronological workflow intervals from point events.

    For every ticket, each event is paired with the next event
    belonging to the same ticket.

    The final event of a ticket does not produce an interval
    because there is no known end timestamp for that state.

    Parameters
    ----------
    events:
        Canonical WorkflowEvent records.

    Returns
    -------
    list[WorkflowInterval]
        Deterministically ordered workflow intervals.
    """

    if not events:
        return []

    sorted_events = sorted(
        events,
        key=_event_sort_key,
    )

    intervals: list[WorkflowInterval] = []

    for current_event, next_event in zip(
        sorted_events,
        sorted_events[1:],
    ):
        # Never create an interval across two different tickets.
        if current_event.ticket_id != next_event.ticket_id:
            continue

        duration_seconds = (
            next_event.timestamp - current_event.timestamp
        ).total_seconds()

        if duration_seconds < 0:
            raise ValueError(
                "Workflow events must be chronological."
            )

        interval = WorkflowInterval(
            interval_id=(
                f"{current_event.event_id}"
                f"__{next_event.event_id}"
            ),
            ticket_id=current_event.ticket_id,
            start_event_id=current_event.event_id,
            end_event_id=next_event.event_id,
            start_timestamp=current_event.timestamp,
            end_timestamp=next_event.timestamp,
            activity=current_event.activity,
            status=current_event.status,
            assignee=current_event.assignee,
            team=current_event.team,
            duration_seconds=duration_seconds,
            metadata={
                "end_activity": next_event.activity,
                "end_status": next_event.status,
            },
        )

        intervals.append(interval)

    return intervals
