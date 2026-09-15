"""Correlation id and access logging.

Every request gets an id — reused from the ``X-Request-ID`` header when the
caller supplied a well-formed one, generated otherwise. The id is bound to the
context (so all log records carry it), stored on ``request.state`` (so error
handlers can reference it) and returned in the response header.

The access log records shape, not content: method, route, status and duration.
Query strings, headers and bodies are never logged — they can carry user
content.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from portfolio_rag.core.logging import get_logger
from portfolio_rag.core.request_context import (
    REQUEST_ID_HEADER,
    reset_request_id,
    sanitize_request_id,
    set_request_id,
)

_logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the request, and echo the id back."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = sanitize_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        token = set_request_id(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # The exception itself is logged with its traceback by the
            # registered handler. Here we only close the access log entry so a
            # failed request is never silently missing from it.
            _logger.warning(
                "request failed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            raise
        else:
            response.headers[REQUEST_ID_HEADER] = request_id
            _logger.info(
                "request completed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": response.status_code,
                    "duration_ms": _elapsed_ms(started),
                },
            )
            return response
        finally:
            # Runs after the else/except block, so every log record above still
            # carries the correlation id.
            reset_request_id(token)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
