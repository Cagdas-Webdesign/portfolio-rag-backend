"""Liveness endpoint.

Deliberately *not* versioned: ``/health`` is an operational contract for
process supervisors and uptime checks, not part of the public product API, and
those callers must not have to follow API version changes.
"""

from __future__ import annotations

from fastapi import APIRouter

from portfolio_rag.api.dependencies import AppSettings
from portfolio_rag.api.schemas.health import HealthResponse

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Reports that the process is up and which build is serving. "
        "Performs no dependency checks, so it is safe to poll frequently."
    ),
)
async def health(settings: AppSettings) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        version=settings.version,
    )
