from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.intelligence.analytics.organization import (
    OrganizationAnalytics,
)
from backend.app.intelligence.schemas.overview import (
    DistributionItem,
)
from backend.app.intelligence.api.organization import (
    get_organization_intelligence_provider,
    router,
)
from backend.app.intelligence.services.organization_intelligence import (
    OrganizationIntelligence,
)


def make_organization_intelligence(
) -> OrganizationIntelligence:
    organization = OrganizationAnalytics(
        agent_count=2,
        active_agent_count=2,
        team_count=1,
        total_ticket_count=10,
        open_ticket_count=6,
        closed_ticket_count=4,
        high_priority_ticket_count=2,
        waiting_ticket_count=3,
        total_lifecycle_seconds=10000,
        average_lifecycle_seconds=1000,
        total_waiting_seconds=2000,
        average_waiting_seconds=200,
        total_transition_count=25,
        total_handoff_count=5,
        priority_distribution=[
            DistributionItem(
                label="Medium",
                count=8,
            ),
            DistributionItem(
                label="High",
                count=2,
            ),
        ],
        status_distribution=[
            DistributionItem(
                label="open",
                count=6,
            ),
            DistributionItem(
                label="closed",
                count=4,
            ),
        ],
        team_workload=[
            DistributionItem(
                label="Team-A",
                count=10,
            ),
        ],
        emerging_issues=[
            DistributionItem(
                label="Login failures",
                count=4,
            ),
        ],
        bottlenecks=[],
        workflow_variants=[],
    )

    return OrganizationIntelligence(
        organization=organization,
        agent_analytics=[],
        team_analytics=[],
        bottlenecks=[],
        workflow_variants=[],
        emerging_issues=[
            DistributionItem(
                label="Login failures",
                count=4,
            )
        ],
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
        get_organization_intelligence_provider
    ] = lambda: provider

    return app


def test_get_organization_intelligence_returns_200():
    expected = (
        make_organization_intelligence()
    )

    def provider():
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/organization/intelligence"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["organization"][
            "total_ticket_count"
        ]
        == 10
    )


def test_organization_provider_is_called():
    expected = (
        make_organization_intelligence()
    )

    called = {
        "value": False
    }

    def provider():
        called["value"] = True
        return expected

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/organization/intelligence"
    )

    assert response.status_code == 200
    assert called["value"] is True


def test_missing_organization_returns_404():
    def provider():
        return None

    client = TestClient(
        make_app(provider)
    )

    response = client.get(
        "/api/intelligence/organization/intelligence"
    )

    assert response.status_code == 404

    assert (
        "not found"
        in response.json()["detail"].lower()
    )


def test_unconfigured_provider_returns_503():
    app = FastAPI()

    app.include_router(
        router,
        prefix="/api/intelligence",
    )

    client = TestClient(app)

    response = client.get(
        "/api/intelligence/organization/intelligence"
    )

    assert response.status_code == 503

    assert (
        "provider is not configured"
        in response.json()["detail"].lower()
    )


def test_organization_response_contains_expected_sections():
    expected = (
        make_organization_intelligence()
    )

    client = TestClient(
        make_app(
            lambda: expected
        )
    )

    response = client.get(
        "/api/intelligence/organization/intelligence"
    )

    payload = response.json()

    assert "organization" in payload
    assert "agent_analytics" in payload
    assert "team_analytics" in payload
    assert "bottlenecks" in payload
    assert "workflow_variants" in payload
    assert "emerging_issues" in payload


def test_organization_metrics_are_serialized():
    expected = (
        make_organization_intelligence()
    )

    client = TestClient(
        make_app(
            lambda: expected
        )
    )

    response = client.get(
        "/api/intelligence/organization/intelligence"
    )

    payload = response.json()

    organization = payload[
        "organization"
    ]

    assert (
        organization[
            "agent_count"
        ]
        == 2
    )

    assert (
        organization[
            "active_agent_count"
        ]
        == 2
    )

    assert (
        organization[
            "team_count"
        ]
        == 1
    )

    assert (
        organization[
            "open_ticket_count"
        ]
        == 6
    )

    assert (
        organization[
            "closed_ticket_count"
        ]
        == 4
    )