from datetime import datetime, timedelta, timezone

import pytest

from backend.app.intelligence.schemas.ticket import PriorityLevel
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
    identify_bottlenecks,
)
from backend.app.intelligence.workflow.event_builder import (
    build_workflow_intervals,
)


BASE_TIME = datetime(
    2026,
    1,
    1,
    9,
    0,
    tzinfo=timezone.utc,
)


def make_event(
    event_id: str,
    ticket_id: str,
    minutes: int,
    status: str,
) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        ticket_id=ticket_id,
        timestamp=BASE_TIME + timedelta(
            minutes=minutes
        ),
        activity="status_change",
        status=status,
        assignee="agent-1",
        priority=PriorityLevel.MEDIUM,
        source="helpdesk",
    )


def test_empty_intervals_return_empty_bottlenecks():
    assert identify_bottlenecks([]) == []


def test_bottleneck_candidates_are_created():
    events = [
        make_event("E1", "T1", 0, "open"),
        make_event("E2", "T1", 30, "waiting"),
        make_event("E3", "T1", 90, "resolved"),
        make_event("E4", "T2", 0, "open"),
        make_event("E5", "T2", 120, "waiting"),
        make_event("E6", "T2", 180, "resolved"),
    ]

    intervals = build_workflow_intervals(events)

    results = identify_bottlenecks(intervals)

    assert len(results) == 2

    statuses = {item.status for item in results}

    assert statuses == {
        "open",
        "waiting",
    }


def test_ticket_and_interval_counts_are_correct():
    events = [
        make_event("E1", "T1", 0, "open"),
        make_event("E2", "T1", 30, "waiting"),
        make_event("E3", "T1", 90, "waiting"),
        make_event("E4", "T1", 120, "resolved"),
        make_event("E5", "T2", 0, "open"),
        make_event("E6", "T2", 60, "waiting"),
        make_event("E7", "T2", 120, "resolved"),
    ]

    intervals = build_workflow_intervals(events)

    results = identify_bottlenecks(intervals)

    waiting = next(
        item
        for item in results
        if item.status == "waiting"
    )

    assert waiting.interval_count == 3
    assert waiting.ticket_count == 2


def test_duration_is_aggregated_by_status():
    events = [
        make_event("E1", "T1", 0, "open"),
        make_event("E2", "T1", 30, "waiting"),
        make_event("E3", "T1", 90, "resolved"),
    ]

    intervals = build_workflow_intervals(events)

    results = identify_bottlenecks(intervals)

    waiting = next(
        item
        for item in results
        if item.status == "waiting"
    )

    assert waiting.total_seconds == 3600.0
    assert waiting.average_seconds == 3600.0


def test_shares_are_between_zero_and_one():
    events = [
        make_event("E1", "T1", 0, "open"),
        make_event("E2", "T1", 30, "waiting"),
        make_event("E3", "T1", 90, "resolved"),
        make_event("E4", "T2", 0, "open"),
        make_event("E5", "T2", 60, "waiting"),
        make_event("E6", "T2", 120, "resolved"),
    ]

    intervals = build_workflow_intervals(events)

    results = identify_bottlenecks(intervals)

    for result in results:
        assert 0 <= result.volume_share <= 1
        assert 0 <= result.duration_share <= 1
        assert 0 <= result.recurrence_rate <= 1
        assert 0 <= result.bottleneck_score <= 1


def test_results_are_sorted_by_bottleneck_score():
    events = [
        make_event("E1", "T1", 0, "open"),
        make_event("E2", "T1", 10, "waiting"),
        make_event("E3", "T1", 110, "resolved"),
        make_event("E4", "T2", 0, "open"),
        make_event("E5", "T2", 10, "waiting"),
        make_event("E6", "T2", 110, "resolved"),
    ]

    intervals = build_workflow_intervals(events)

    results = identify_bottlenecks(intervals)

    scores = [
        item.bottleneck_score
        for item in results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_missing_status_is_ignored():
    events = [
        WorkflowEvent(
            event_id="E1",
            ticket_id="T1",
            timestamp=BASE_TIME,
            activity="created",
            status=None,
            priority=PriorityLevel.MEDIUM,
        ),
        WorkflowEvent(
            event_id="E2",
            ticket_id="T1",
            timestamp=BASE_TIME + timedelta(
                minutes=30
            ),
            activity="status_change",
            status="waiting",
            priority=PriorityLevel.MEDIUM,
        ),
    ]

    intervals = build_workflow_intervals(events)

    results = identify_bottlenecks(intervals)

    assert all(
        result.status != ""
        for result in results
    )


def test_bottleneck_model_rejects_negative_values():
    with pytest.raises(Exception):
        BottleneckCandidate(
            status="waiting",
            interval_count=-1,
            ticket_count=1,
            total_seconds=100,
            average_seconds=100,
            volume_share=0.5,
            duration_share=0.5,
            recurrence_rate=0.5,
            bottleneck_score=0.1,
        )