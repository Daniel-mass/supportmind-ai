from datetime import datetime, timezone

from backend.app.intelligence.schemas.workflow import (
    WorkflowEvent,
)
from backend.app.intelligence.services.workflow_intelligence import (
    WorkflowIntelligence,
    build_workflow_intelligence,
)
from backend.app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
)
from backend.app.intelligence.workflow.handoffs import (
    HandoffRecord,
)
from backend.app.intelligence.workflow.metrics import (
    WorkflowMetrics,
)
from backend.app.intelligence.workflow.variants import (
    WorkflowVariant,
)


TIMESTAMP = datetime(
    2026,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)


def make_event(
    *,
    event_id: str = "event-001",
) -> WorkflowEvent:
    return WorkflowEvent(
        event_id=event_id,
        ticket_id="PROJ:100",
        timestamp=TIMESTAMP,
        activity="status_change",
        status="in_progress",
        assignee="agent-001",
        team="Team-A",
        priority="High",
        category="Bug",
        source="helpdesk_change_history",
        metadata={},
    )


def make_metrics() -> WorkflowMetrics:
    return WorkflowMetrics(
        ticket_id="PROJ:100",
        lifecycle_seconds=1000,
        active_seconds=700,
        waiting_seconds=300,
        status_durations={
            "in_progress": 700,
            "waiting": 300,
        },
        interval_count=5,
        transition_count=4,
        unique_status_count=3,
        unique_assignee_count=2,
    )


def make_handoff() -> HandoffRecord:
    return HandoffRecord(
        handoff_id="handoff-001",
        ticket_id="PROJ:100",
        from_agent="agent-001",
        to_agent="agent-002",
        handoff_timestamp=TIMESTAMP,
        from_team="Team-A",
        to_team="Team-B",
        status_before="in_progress",
        status_after="waiting",
        time_before_handoff_seconds=100,
        time_after_handoff_seconds=200,
    )


def make_bottleneck() -> BottleneckCandidate:
    return BottleneckCandidate(
        status="waiting",
        interval_count=10,
        ticket_count=5,
        total_seconds=2000,
        average_seconds=400,
        volume_share=0.5,
        duration_share=0.5,
        recurrence_rate=0.5,
        bottleneck_score=0.125,
    )


def make_variant() -> WorkflowVariant:
    return WorkflowVariant(
        variant_id="variant-001",
        path=[
            "in_progress",
            "waiting",
            "closed",
        ],
        path_string=(
            "in_progress -> waiting -> closed"
        ),
        ticket_count=5,
        percentage=25.0,
    )


def test_build_workflow_intelligence_returns_expected_model():
    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
    )

    assert isinstance(
        result,
        WorkflowIntelligence,
    )


def test_ticket_id_is_preserved():
    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
    )

    assert result.ticket_id == "PROJ:100"


def test_events_are_preserved():
    event = make_event()

    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
        events=[event],
    )

    assert result.events == [event]


def test_metrics_are_preserved():
    metrics = make_metrics()

    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
        metrics=metrics,
    )

    assert result.metrics == metrics


def test_handoffs_are_preserved():
    handoff = make_handoff()

    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
        handoffs=[handoff],
    )

    assert result.handoffs == [handoff]


def test_bottlenecks_are_preserved():
    bottleneck = make_bottleneck()

    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
        bottlenecks=[bottleneck],
    )

    assert result.bottlenecks == [bottleneck]


def test_variants_are_preserved():
    variant = make_variant()

    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
        variants=[variant],
    )

    assert result.variants == [variant]


def test_missing_optional_data_returns_empty_lists():
    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
    )

    assert result.metrics is None
    assert result.events == []
    assert result.handoffs == []
    assert result.bottlenecks == []
    assert result.variants == []


def test_full_workflow_intelligence_composition():
    event = make_event()
    metrics = make_metrics()
    handoff = make_handoff()
    bottleneck = make_bottleneck()
    variant = make_variant()

    result = build_workflow_intelligence(
        ticket_id="PROJ:100",
        events=[event],
        metrics=metrics,
        handoffs=[handoff],
        bottlenecks=[bottleneck],
        variants=[variant],
    )

    assert result.ticket_id == "PROJ:100"
    assert result.metrics == metrics
    assert result.events == [event]
    assert result.handoffs == [handoff]
    assert result.bottlenecks == [bottleneck]
    assert result.variants == [variant]