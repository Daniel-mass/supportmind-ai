
from __future__ import annotations

from collections.abc import Callable

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.intelligence.providers.helpdesk_provider import (
    HelpDeskProviderError,
    get_helpdesk_provider,
)
from app.intelligence.schemas.trends import OrganizationTrends
from app.intelligence.services.organization_intelligence import (
    OrganizationIntelligence,
)


OrganizationIntelligenceProvider = Callable[
    [],
    OrganizationIntelligence | None,
]

OrganizationTrendsProvider = Callable[
    [],
    OrganizationTrends | None,
]


router = APIRouter(
    prefix="/organization",
    tags=["intelligence-organization"],
)


def get_organization_intelligence_provider(
) -> OrganizationIntelligenceProvider:
    provider = get_helpdesk_provider()

    return provider.get_organization_intelligence


def get_organization_trends_provider(
) -> OrganizationTrendsProvider:
    provider = get_helpdesk_provider()

    return provider.get_organization_trends


@router.get(
    "/intelligence",
    response_model=OrganizationIntelligence,
    status_code=status.HTTP_200_OK,
)
def get_organization_intelligence(
    provider: OrganizationIntelligenceProvider = Depends(
        get_organization_intelligence_provider
    ),
) -> OrganizationIntelligence:
    try:
        result = provider()

    except HelpDeskProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization intelligence not found.",
        )

    return result


@router.get(
    "/trends",
    response_model=OrganizationTrends,
    status_code=status.HTTP_200_OK,
)
def get_organization_trends(
    provider: OrganizationTrendsProvider = Depends(
        get_organization_trends_provider
    ),
) -> OrganizationTrends:
    try:
        result = provider()

    except HelpDeskProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization trends not found.",
        )

    return result