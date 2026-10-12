from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.support.schemas.assist import (
    SupportAssistRequest,
    SupportAssistResponse,
)
from app.support.services.support_assist import (
    SupportAssistService,
)


router = APIRouter(
    prefix="/support",
    tags=["support-assist"],
)


@lru_cache(maxsize=1)
def get_support_assist_service() -> SupportAssistService:
    """
    Return the process-level SupportMind workflow service.

    The expensive retrieval, graph, model, and agent dependencies
    are initialized once and reused across requests.
    """

    service = SupportAssistService()

    try:
        service.initialize()
    except Exception:
        service.close()
        raise

    return service


@router.post(
    "/assist",
    response_model=SupportAssistResponse,
    status_code=status.HTTP_200_OK,
)
async def support_assist(
    payload: SupportAssistRequest,
    service: SupportAssistService = Depends(
        get_support_assist_service
    ),
) -> SupportAssistResponse:
    """
    Execute the complete SupportMind support workflow.

    Public input is a natural-language freestyle prompt.
    """

    try:
        result = await service.assist(
            request=payload.request,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "SupportMind support-assist workflow "
                "could not be executed."
            ),
        ) from exc

    if result.intelligence.workflow.status.value == "failed":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "SupportMind support-assist workflow "
                "failed to complete."
            ),
        )

    return result