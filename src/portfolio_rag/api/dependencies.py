"""Request-scoped dependencies.

Settings are resolved from the **application instance**, not from the
process-wide singleton. An app built with explicit settings must use those
settings everywhere — in its routes, in its lifespan and in its logging
configuration. Reading :func:`~portfolio_rag.core.config.get_settings` inside a
handler would quietly reintroduce a second source of truth, which is exactly
the bug this module exists to prevent.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from portfolio_rag.core.config import Settings
from portfolio_rag.rag.service import GroundedAnswerService


def get_app_settings(request: Request) -> Settings:
    """Return the settings bound to the application handling this request."""
    settings = request.app.state.settings
    if not isinstance(settings, Settings):  # pragma: no cover - wiring bug
        raise RuntimeError("Application was built without bound settings.")
    return settings


def get_answer_service(request: Request) -> GroundedAnswerService:
    """Return the query stack this application was started with.

    Built once, during startup, from the composition root. A route that wired
    its own adapters would be a second composition root with none of the first
    one's checks.
    """
    service = getattr(request.app.state, "answer_service", None)
    if not isinstance(service, GroundedAnswerService):  # pragma: no cover - wiring bug
        raise RuntimeError("Application was built without a query stack.")
    return service


#: Annotated shorthands for route signatures.
AppSettings = Annotated[Settings, Depends(get_app_settings)]
AnswerService = Annotated[GroundedAnswerService, Depends(get_answer_service)]
