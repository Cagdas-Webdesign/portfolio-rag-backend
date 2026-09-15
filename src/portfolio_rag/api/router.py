"""Composition of the HTTP surface.

Two roots, on purpose:

* ``/`` — operational endpoints that must stay stable forever (``/health``).
* ``/api/v1`` — the versioned product API. Breaking changes get a new prefix
  rather than mutating an existing one, so the portfolio client can migrate on
  its own schedule.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter

from portfolio_rag.api.routes import chat, health

API_V1_PREFIX: Final = "/api/v1"

operational_router = APIRouter()
operational_router.include_router(health.router)

api_v1_router = APIRouter(prefix=API_V1_PREFIX)
api_v1_router.include_router(chat.router)
