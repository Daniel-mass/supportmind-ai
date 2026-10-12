from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
)
from backend.app.intelligence.api.agents import (
    get_agent_analytics_provider,
    router,
)


def make_agent_analytics() -> AgentAnalytics:
    return AgentAnalytics(
        agent_id="agent-001",
        name="Agent One",
        team="Team-A",
        active=True,
        assigned_ticket_count=10,
        open_ticket_count=6,
        closed_ticket_count=4,
        high_priority_ticket_count=2,
        waiting_ticket_count=3,
        total_lifecycle_seconds=10000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=2000,
        average_waiting_seconds=200,
        total_transition_count=25,
        incoming_handoff_count=5,
        outgoing_handoff_count=4,
        unique_project_count=2,
        unique_issue_type_count=3,
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
        get_agent_analytics_provider
    ] = lambda: provider

    return app


def test_get_agent_intelligence_returns_200():
    expected = make_agent_analytics()

    def provider(
        agent_id: str,
    ):
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/agents/"
        "agent-001/intelligence"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["agent_id"]
        == "agent-001"
    )


def test_agent_id_is_passed_to_provider():
    expected = make_agent_analytics()

    captured = {}

    def provider(
        agent_id: str,
    ):
        captured["agent_id"] = agent_id
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/agents/"
        "agent-001/intelligence"
    )

    assert response.status_code == 200
    assert (
        captured["agent_id"]
        == "agent-001"
    )


def test_missing_agent_returns_404():
    def provider(
        agent_id: str,
    ):
        return None

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/agents/"
        "unknown-agent/intelligence"
    )

    assert response.status_code == 404

    assert (
        "not found"
        in response.json()["detail"].lower()
    )


def test_provider_mismatch_returns_500():
    expected = make_agent_analytics()

    def provider(
        agent_id: str,
    ):
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/agents/"
        "agent-999/intelligence"
    )

    assert response.status_code == 500

    assert (
        "mismatched agent_id"
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
        "/api/intelligence/agents/"
        "agent-001/intelligence"
    )

    assert response.status_code == 503

    assert (
        "provider is not configured"
        in response.json()["detail"].lower()
    )


def test_agent_response_contains_expected_fields():
    expected = make_agent_analytics()

    client = TestClient(
        make_app(
            lambda agent_id: expected
        )
    )

    response = client.get(
        "/api/intelligence/agents/"
        "agent-001/intelligence"
    )

    payload = response.json()

    assert payload["agent_id"] == "agent-001"
    assert payload["name"] == "Agent One"
    assert payload["team"] == "Team-A"
    assert payload["active"] is True

    assert (
        payload["assigned_ticket_count"]
        == 10
    )

    assert (
        payload["high_priority_ticket_count"]
        == 2
    )

    assert (
        payload["waiting_ticket_count"]
        == 3
    )

    assert (
        payload["incoming_handoff_count"]
        == 5
    )

    assert (
        payload["outgoing_handoff_count"]
        == 4
    )