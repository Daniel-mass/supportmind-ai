from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field

from app.intelligence.workflow.event_builder import (
    WorkflowInterval,
)


class BottleneckCandidate(BaseModel):
    """
    Dashboard-ready workflow bottleneck candidate.

    The score is a deterministic operational signal, not a
    root-cause judgment.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    status: str = Field(..., min_length=1)

    interval_count: int = Field(..., ge=0)
    ticket_count: int = Field(..., ge=0)

    total_seconds: float = Field(..., ge=0)
    average_seconds: float = Field(..., ge=0)

    volume_share: float = Field(..., ge=0, le=1)
    duration_share: float = Field(..., ge=0, le=1)
    recurrence_rate: float = Field(..., ge=0, le=1)

    bottleneck_score: float = Field(..., ge=0, le=1)


def identify_bottlenecks(
    intervals: list[WorkflowInterval],
) -> list[BottleneckCandidate]:
    """
    Aggregate workflow intervals by status and identify
    recurring bottleneck candidates.

    The operational score combines:

        volume share
        Ã— duration share
        Ã— recurrence rate

    This intentionally remains a heuristic operational signal.
    It does not claim causal root cause.
    """

    if not intervals:
        return []

    status_intervals: dict[
        str,
        list[WorkflowInterval],
    ] = defaultdict(list)

    total_duration = 0.0
    total_intervals = len(intervals)

    all_tickets = {
        interval.ticket_id
        for interval in intervals
    }

    total_tickets = len(all_tickets)

    for interval in intervals:
        if not interval.status:
            continue

        status = interval.status.strip()

        if not status:
            continue

        status_intervals[status].append(interval)
        total_duration += interval.duration_seconds

    if not status_intervals or total_duration <= 0:
        return []

    candidates: list[BottleneckCandidate] = []

    for status, status_rows in status_intervals.items():

        interval_count = len(status_rows)

        ticket_ids = {
            interval.ticket_id
            for interval in status_rows
        }

        ticket_count = len(ticket_ids)

        total_seconds = sum(
            interval.duration_seconds
            for interval in status_rows
        )

        average_seconds = (
            total_seconds / interval_count
            if interval_count
            else 0.0
        )

        volume_share = (
            interval_count / total_intervals
            if total_intervals
            else 0.0
        )

        duration_share = (
            total_seconds / total_duration
            if total_duration
            else 0.0
        )

        recurrence_rate = (
            ticket_count / total_tickets
            if total_tickets
            else 0.0
        )

        bottleneck_score = (
            volume_share
            * duration_share
            * recurrence_rate
        )

        candidates.append(
            BottleneckCandidate(
                status=status,
                interval_count=interval_count,
                ticket_count=ticket_count,
                total_seconds=total_seconds,
                average_seconds=average_seconds,
                volume_share=volume_share,
                duration_share=duration_share,
                recurrence_rate=recurrence_rate,
                bottleneck_score=bottleneck_score,
            )
        )

    candidates.sort(
        key=lambda item: (
            item.bottleneck_score,
            item.total_seconds,
            item.ticket_count,
        ),
        reverse=True,
    )

    return candidates
