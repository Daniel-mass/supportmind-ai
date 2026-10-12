from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.intelligence.api.tickets import (
    get_ticket_intelligence_provider,
    router,
)
from backend.app.intelligence.analytics.ticket_analytics import (
    TicketAnalytics,
)
from backend.app.intelligence.schemas.ticket import Ticket
from backend.app.intelligence.services.ticket_intelligence import (
    TicketIntelligence,
)


TIMESTAMP = datetime(
    2026,
    1,
    1,
    10,
    0,
    tzinfo=timezone.utc,
)


def make_ticket_intelligence() -> TicketIntelligence:
    ticket = Ticket(
        ticket_id="PROJ:100",
        project="PROJ",
        issue_num=100,
        issue_type="Bug",
        priority="High",
        status="in_progress",
        created_at=TIMESTAMP,
        started_at=TIMESTAMP,
        ended_at=None,
        resolution_date=None,
        resolution=None,
        reporter="user@example.com",
        assignee="agent-001",
        contributor_count=1,
        comments_count=2,
        source="helpdesk",
        metadata={},
    )

    analytics = TicketAnalytics(
        ticket_id="PROJ:100",
        project="PROJ",
        issue_num=100,
        issue_type="Bug",
        priority="High",
        status="in_progress",
        lifecycle_seconds=1000,
        active_seconds=700,
        waiting_seconds=300,
        active_ratio=0.7,
        waiting_ratio=0.3,
        transition_count=4,
        unique_status_count=3,
        unique_assignee_count=2,
        handoff_count=1,
        is_high_priority=True,
        is_waiting=False,
    )

    return TicketIntelligence(
        ticket=ticket,
        analytics=analytics,
        workflow_metrics=None,
        workflow_events=[],
        handoffs=[],
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
        get_ticket_intelligence_provider
    ] = lambda: provider

    return app


def test_get_ticket_intelligence_returns_200():
    expected = (
        make_ticket_intelligence()
    )

    def provider(
        ticket_id: str,
    ):
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/tickets/PROJ:100/intelligence"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["ticket"]["ticket_id"]
        == "PROJ:100"
    )

    assert (
        payload["analytics"]["ticket_id"]
        == "PROJ:100"
    )


def test_ticket_id_is_trimmed():
    expected = (
        make_ticket_intelligence()
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
        "/api/intelligence/tickets/"
        "PROJ:100%20/intelligence"
    )

    assert response.status_code == 200
    assert captured["ticket_id"] == "PROJ:100"


def test_missing_ticket_returns_404():
    def provider(
        ticket_id: str,
    ):
        return None

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/tickets/UNKNOWN:intelligence"
    )

    assert response.status_code == 404

    payload = response.json()

    assert (
        "not found"
        in payload["detail"].lower()
    )


def test_provider_mismatch_returns_500():
    expected = (
        make_ticket_intelligence()
    )

    def provider(
        ticket_id: str,
    ):
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/tickets/OTHER:999/intelligence"
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
        "/api/intelligence/tickets/PROJ:100/intelligence"
    )

    assert response.status_code == 503

    assert (
        "provider is not configured"
        in response.json()["detail"].lower()
    )


def test_ticket_response_contains_expected_sections():
    expected = (
        make_ticket_intelligence()
    )

    client = TestClient(
        make_app(
            lambda ticket_id: expected
        )
    )

    response = client.get(
        "/api/intelligence/tickets/PROJ:100/intelligence"
    )

    payload = response.json()

    assert "ticket" in payload
    assert "analytics" in payload
    assert "workflow_metrics" in payload
    assert "workflow_events" in payload
    assert "handoffs" in payload