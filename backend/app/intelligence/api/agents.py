from __future__ import annotations

from collections.abc import Callable

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.intelligence.analytics.agent_analytics import (
    AgentAnalytics,
)
from app.intelligence.providers.helpdesk_provider import (
    HelpDeskProviderError,
    get_helpdesk_provider,
)


AgentAnalyticsProvider = Callable[
    [str],
    AgentAnalytics | None,
]


router = APIRouter(
    prefix="/agents",
    tags=["intelligence-agents"],
)


def get_agent_analytics_provider() -> (
    AgentAnalyticsProvider
):
    """
    Provide the application-level agent intelligence
    retrieval function.
    """

    provider = get_helpdesk_provider()

    return provider.get_agent_intelligence


@router.get(
    "/{agent_id}/intelligence",
    response_model=AgentAnalytics,
    status_code=status.HTTP_200_OK,
)
def get_agent_intelligence(
    agent_id: str,
    provider: AgentAnalyticsProvider = Depends(
        get_agent_analytics_provider
    ),
) -> AgentAnalytics:
    """
    Return dashboard-ready intelligence for one support agent.
    """

    normalized_agent_id = agent_id.strip()

    if not normalized_agent_id:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="agent_id must not be empty.",
        )

    try:
        result = provider(
            normalized_agent_id
        )

    except HelpDeskProviderError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=str(exc),
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Agent analytics not found "
                f"for agent_id="
                f"'{normalized_agent_id}'."
            ),
        )

    if (
        result.agent_id
        != normalized_agent_id
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Agent analytics provider "
                "returned a mismatched agent_id."
            ),
        )

    return result