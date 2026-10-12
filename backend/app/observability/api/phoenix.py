from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query


router = APIRouter(
    prefix="/phoenix",
    tags=["Phoenix"],
)


PHOENIX_BASE_URL = os.getenv(
    "PHOENIX_BASE_URL",
    "http://127.0.0.1:6006",
)

PHOENIX_API_KEY = os.getenv(
    "PHOENIX_API_KEY"
)


def _headers() -> dict[str, str]:
    if PHOENIX_API_KEY:
        return {
            "Authorization": f"Bearer {PHOENIX_API_KEY}"
        }

    return {}


async def _get(
    path: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:

    url = f"{PHOENIX_BASE_URL.rstrip('/')}{path}"

    try:
        async with httpx.AsyncClient(
            timeout=15.0
        ) as client:

            response = await client.get(
                url,
                params=params,
                headers=_headers(),
            )

            response.raise_for_status()

            return response.json()

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail={
                "phoenix_status": exc.response.status_code,
                "message": exc.response.text,
            },
        )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unavailable",
                "phoenix_url": url,
                "error": str(exc),
            },
        )


@router.get("/health")
async def phoenix_health() -> dict[str, Any]:

    try:
        data = await _get("/v1/projects")

        return {
            "status": "healthy",
            "base_url": PHOENIX_BASE_URL,
            "project_count": len(
                data.get("data", [])
            ),
        }

    except HTTPException:
        raise


@router.get("/projects")
async def list_projects(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
) -> dict[str, Any]:

    return await _get(
        "/v1/projects",
        params={
            "limit": limit,
        },
    )


@router.get("/projects/{project_identifier}")
async def get_project(
    project_identifier: str,
) -> dict[str, Any]:

    return await _get(
        f"/v1/projects/{project_identifier}"
    )


@router.get(
    "/projects/{project_identifier}/traces"
)
async def list_traces(
    project_identifier: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    cursor: str | None = None,
) -> dict[str, Any]:

    params: dict[str, Any] = {
        "limit": limit,
    }

    if cursor:
        params["cursor"] = cursor

    return await _get(
        f"/v1/projects/{project_identifier}/traces",
        params=params,
    )


@router.get(
    "/projects/{project_identifier}/sessions"
)
async def list_sessions(
    project_identifier: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    cursor: str | None = None,
    order: str = Query(
        default="desc",
        pattern="^(asc|desc)$",
    ),
) -> dict[str, Any]:

    params: dict[str, Any] = {
        "limit": limit,
        "order": order,
    }

    if cursor:
        params["cursor"] = cursor

    return await _get(
        f"/v1/projects/{project_identifier}/sessions",
        params=params,
    )