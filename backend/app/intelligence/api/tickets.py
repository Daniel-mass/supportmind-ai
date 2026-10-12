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
from app.intelligence.services.ticket_intelligence import (
    TicketIntelligence,
)


TicketIntelligenceProvider = Callable[
    [str],
    TicketIntelligence | None,
]


router = APIRouter(
    prefix="/tickets",
    tags=["intelligence-tickets"],
)


def get_ticket_intelligence_provider(
) -> TicketIntelligenceProvider:
    """
    Return the application-level ticket intelligence provider.
    """

    provider = get_helpdesk_provider()

    return provider.get_ticket_intelligence


@router.get(
    "/{ticket_id}/intelligence",
    response_model=TicketIntelligence,
    status_code=status.HTTP_200_OK,
)
def get_ticket_intelligence(
    ticket_id: str,
    provider: TicketIntelligenceProvider = Depends(
        get_ticket_intelligence_provider
    ),
) -> TicketIntelligence:
    """
    Return intelligence for a single Help Desk ticket.
    """

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
                "Ticket intelligence not found "
                f"for ticket_id="
                f"'{normalized_ticket_id}'."
            ),
        )

    if (
        result.ticket.ticket_id
        != normalized_ticket_id
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Ticket intelligence provider "
                "returned a mismatched ticket_id."
            ),
        )

    return result