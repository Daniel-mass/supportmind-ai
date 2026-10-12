from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.intelligence.api.workflow import (
    get_workflow_intelligence_provider,
    router,
)
from backend.app.intelligence.schemas.workflow import (
    WorkflowEvent,
)
from backend.app.intelligence.services.workflow_intelligence import (
    WorkflowIntelligence,
)
from backend.app.intelligence.workflow.handoffs import (
    HandoffRecord,
)
from backend.app.intelligence.workflow.metrics import (
    WorkflowMetrics,
)


TIMESTAMP = datetime(
    2026,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)


def make_workflow_intelligence() -> WorkflowIntelligence:
    event = WorkflowEvent(
        event_id="event-001",
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

    metrics = WorkflowMetrics(
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

    handoff = HandoffRecord(
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

    return WorkflowIntelligence(
        ticket_id="PROJ:100",
        events=[event],
        metrics=metrics,
        handoffs=[handoff],
        bottlenecks=[],
        variants=[],
    )


def make_app(
    provider,
) -> FastAPI:
    app = FastAPI()

    app.include_router(
        router,
        prefix="/api/intelligence",
    )

    app.dependency_overrides[
        get_workflow_intelligence_provider
    ] = lambda: provider

    return app


def test_get_workflow_intelligence_returns_200():
    expected = (
        make_workflow_intelligence()
    )

    def provider(
        ticket_id: str,
    ):
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/workflow/"
        "PROJ:100/intelligence"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["ticket_id"]
        == "PROJ:100"
    )


def test_ticket_id_is_passed_to_provider():
    expected = (
        make_workflow_intelligence()
    )

    captured = {}

    def provider(
        ticket_id: str,
    ):
        captured["ticket_id"] = ticket_id
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/workflow/"
        "PROJ:100/intelligence"
    )

    assert response.status_code == 200
    assert (
        captured["ticket_id"]
        == "PROJ:100"
    )


def test_missing_workflow_intelligence_returns_404():
    def provider(
        ticket_id: str,
    ):
        return None

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/workflow/"
        "UNKNOWN/intelligence"
    )

    assert response.status_code == 404

    assert (
        "not found"
        in response.json()["detail"].lower()
    )


def test_provider_mismatch_returns_500():
    expected = (
        make_workflow_intelligence()
    )

    def provider(
        ticket_id: str,
    ):
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/workflow/"
        "OTHER:999/intelligence"
    )

    assert response.status_code == 500

    assert (
        "mismatched ticket_id"
        in response.json()["detail"]
    )


def test_unconfigured_provider_returns_503():
    app = FastAPI()

    app.include_router(
        router,
        prefix="/api/intelligence",
    )

    client = TestClient(app)

    response = client.get(
        "/api/intelligence/workflow/"
        "PROJ:100/intelligence"
    )

    assert response.status_code == 503

    assert (
        "provider is not configured"
        in response.json()["detail"].lower()
    )


def test_workflow_response_contains_expected_sections():
    expected = (
        make_workflow_intelligence()
    )

    client = TestClient(
        make_app(
            lambda ticket_id: expected
        )
    )

    response = client.get(
        "/api/intelligence/workflow/"
        "PROJ:100/intelligence"
    )

    payload = response.json()

    assert "ticket_id" in payload
    assert "events" in payload
    assert "metrics" in payload
    assert "handoffs" in payload
    assert "bottlenecks" in payload
    assert "variants" in payload

    assert len(payload["events"]) == 1
    assert len(payload["handoffs"]) == 1