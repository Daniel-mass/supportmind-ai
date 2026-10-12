#organization_intelligence.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
)
from app.intelligence.analytics.organization import (
    OrganizationAnalytics,
)
from app.intelligence.analytics.team_analytics import (
    TeamAnalytics,
)
from app.intelligence.schemas.overview import (
    DistributionItem,
)
from app.intelligence.workflow.bottlenecks import (
    BottleneckCandidate,
)
from app.intelligence.workflow.variants import (
    WorkflowVariant,
)


class OrganizationIntelligence(BaseModel):
    """
    Unified organization-level intelligence response.

    This is a composition contract for dashboard/API consumers.
    It does not perform new analytics, ML, NLP, or graph calculations.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    organization: OrganizationAnalytics

    agent_analytics: list[AgentAnalytics]
    team_analytics: list[TeamAnalytics]

    bottlenecks: list[BottleneckCandidate]
    workflow_variants: list[WorkflowVariant]
    emerging_issues: list[DistributionItem]


def build_organization_intelligence(
    *,
    organization: OrganizationAnalytics,
    agent_analytics: list[AgentAnalytics] | None = None,
    team_analytics: list[TeamAnalytics] | None = None,
    bottlenecks: list[BottleneckCandidate] | None = None,
    workflow_variants: list[WorkflowVariant] | None = None,
    emerging_issues: list[DistributionItem] | None = None,
) -> OrganizationIntelligence:
    """
    Compose already-computed organization-level intelligence.
    """

    return OrganizationIntelligence(
        organization=organization,
        agent_analytics=list(
            agent_analytics or []
        ),
        team_analytics=list(
            team_analytics or []
        ),
        bottlenecks=list(
            bottlenecks or []
        ),
        workflow_variants=list(
            workflow_variants or []
        ),
        emerging_issues=list(
            emerging_issues or []
        ),
    )
