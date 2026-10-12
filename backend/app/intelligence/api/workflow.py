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
from app.intelligence.services.workflow_intelligence import (
    WorkflowIntelligence,
)


WorkflowIntelligenceProvider = Callable[
    [str],
    WorkflowIntelligence | None,
]


router = APIRouter(
    prefix="/workflow",
    tags=["intelligence-workflow"],
)


def get_workflow_intelligence_provider(
) -> WorkflowIntelligenceProvider:
    provider = get_helpdesk_provider()

    return provider.get_workflow_intelligence


@router.get(
    "/{ticket_id}/intelligence",
    response_model=WorkflowIntelligence,
    status_code=status.HTTP_200_OK,
)
def get_workflow_intelligence(
    ticket_id: str,
    provider: WorkflowIntelligenceProvider = Depends(
        get_workflow_intelligence_provider
    ),
) -> WorkflowIntelligence:
    normalized_ticket_id = ticket_id.strip()

    if not normalized_ticket_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ticket_id must not be empty.",
        )

    try:
        result = provider(
            normalized_ticket_id
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
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Workflow intelligence not found "
                f"for ticket_id="
                f"'{normalized_ticket_id}'."
            ),
        )

    if result.ticket_id != normalized_ticket_id:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Workflow intelligence provider "
                "returned a mismatched ticket_id."
            ),
        )

    return result