from datetime import datetime, timedelta, timezone

from backend.app.intelligence.schemas.ticket import PriorityLevel
from backend.app.intelligence.schemas.workflow import WorkflowEvent
from backend.app.intelligence.workflow.event_builder import (
    build_workflow_intervals,
)
from backend.app.intelligence.workflow.variants import (
    identify_workflow_variants,
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


def test_empty_intervals_return_empty_variants():
    assert identify_workflow_variants([]) == []


def test_single_workflow_variant_is_detected():
    events = [
        make_event("E1", "T1", 0, "Open"),
        make_event("E2", "T1", 30, "In Progress"),
        make_event("E3", "T1", 60, "Resolved"),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    assert len(variants) == 1

    variant = variants[0]

    assert variant.path == [
        "Open",
        "In Progress",
    ]

    assert variant.path_string == (
        "Open → In Progress"
    )

    assert variant.ticket_count == 1
    assert variant.percentage == 100.0


def test_multiple_variants_are_detected():
    events = [
        make_event("A1", "T1", 0, "Open"),
        make_event("A2", "T1", 30, "In Progress"),
        make_event("A3", "T1", 60, "Resolved"),

        make_event("B1", "T2", 0, "Open"),
        make_event("B2", "T2", 30, "Waiting"),
        make_event("B3", "T2", 90, "Resolved"),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    assert len(variants) == 2

    paths = {
        variant.path_string
        for variant in variants
    }

    assert "Open → In Progress" in paths
    assert "Open → Waiting" in paths


def test_repeated_statuses_are_collapsed():
    events = [
        make_event("E1", "T1", 0, "Open"),
        make_event("E2", "T1", 30, "In Progress"),
        make_event("E3", "T1", 60, "In Progress"),
        make_event("E4", "T1", 90, "Resolved"),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    assert len(variants) == 1

    assert variants[0].path == [
        "Open",
        "In Progress",
    ]


def test_variant_ticket_counts_are_correct():
    events = [
        make_event("A1", "T1", 0, "Open"),
        make_event("A2", "T1", 30, "In Progress"),
        make_event("A3", "T1", 60, "Resolved"),

        make_event("B1", "T2", 0, "Open"),
        make_event("B2", "T2", 30, "In Progress"),
        make_event("B3", "T2", 60, "Resolved"),

        make_event("C1", "T3", 0, "Open"),
        make_event("C2", "T3", 30, "Waiting"),
        make_event("C3", "T3", 60, "Resolved"),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    common_variant = next(
        item
        for item in variants
        if item.path_string
        == "Open → In Progress"
    )

    assert common_variant.ticket_count == 2


def test_variants_are_sorted_by_ticket_count():
    events = [
        make_event("A1", "T1", 0, "Open"),
        make_event("A2", "T1", 30, "In Progress"),
        make_event("A3", "T1", 60, "Resolved"),

        make_event("B1", "T2", 0, "Open"),
        make_event("B2", "T2", 30, "In Progress"),
        make_event("B3", "T2", 60, "Resolved"),

        make_event("C1", "T3", 0, "Open"),
        make_event("C2", "T3", 30, "Waiting"),
        make_event("C3", "T3", 60, "Resolved"),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    counts = [
        variant.ticket_count
        for variant in variants
    ]

    assert counts == sorted(
        counts,
        reverse=True,
    )


def test_percentages_sum_to_100():
    events = [
        make_event("A1", "T1", 0, "Open"),
        make_event("A2", "T1", 30, "Resolved"),

        make_event("B1", "T2", 0, "Open"),
        make_event("B2", "T2", 30, "Waiting"),

        make_event("C1", "T3", 0, "Open"),
        make_event("C2", "T3", 30, "Resolved"),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    total_percentage = sum(
        variant.percentage
        for variant in variants
    )

    assert round(total_percentage, 6) == 100.0


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
        make_event(
            "E2",
            "T1",
            30,
            "Waiting",
        ),
        make_event(
            "E3",
            "T1",
            60,
            "Resolved",
        ),
    ]

    intervals = build_workflow_intervals(events)

    variants = identify_workflow_variants(intervals)

    assert len(variants) == 1
    assert variants[0].path == ["Waiting"]


def test_variant_id_is_deterministic():
    events = [
        make_event("E1", "T1", 0, "Open"),
        make_event("E2", "T1", 30, "Resolved"),
        make_event("E3", "T2", 0, "Open"),
        make_event("E4", "T2", 30, "Waiting"),
    ]

    intervals = build_workflow_intervals(events)

    variants_first = identify_workflow_variants(
        intervals
    )

    variants_second = identify_workflow_variants(
        intervals
    )

    assert [
        variant.variant_id
        for variant in variants_first
    ] == [
        variant.variant_id
        for variant in variants_second
    ]