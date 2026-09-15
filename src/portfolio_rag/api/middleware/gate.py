"""The checks a request must pass before it is allowed to cost anything.

Two guards, both deliberately stupid, both running before routing, body
parsing or any dependency is resolved:

* **:class:`EdgeAuthMiddleware`** — did this request come through the gateway?
* **:class:`BodySizeLimitMiddleware`** — is it small enough to be a question?

They sit in front of the expensive route only. ``/health`` stays public,
because a platform's health check does not carry a shared secret and a
liveness probe that can be locked out is worse than no probe.

**Why middleware rather than a dependency.** A dependency is resolved after
FastAPI has matched a route and parsed a body; middleware runs before either.
For the size guard that is the whole point — a 50 MB body must be refused
while it is still on the wire, not after it has been read into memory and
validated. For the auth guard it means an unauthenticated request is answered
without the application ever looking at what it contains. It also keeps the
header out of every request model: it is transport, and a client has no
business seeing it in a schema.

**What the edge secret is and is not.** It authenticates the *gateway*, not a
user. There are no accounts here and nothing to log in to; what it buys is that
the public origin URL cannot be used to skip the bot check, the Turnstile
verification and the rate limiter in front of it
([ADR 0008](../../../../docs/adr/0008-abuse-boundary-at-the-edge.md)). Anyone
holding the secret is the gateway as far as this service is concerned, which is
why it is a deployment secret and never reaches a browser.

Refusals are the standard error envelope with the standard correlation id, and
say nothing else: not which header was expected, not how long it should be, not
whether it was absent or merely wrong. A guard that explains itself is a guard
that helps.
"""

from __future__ import annotations

import hmac
from collections.abc import Iterable
from typing import Final

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from portfolio_rag.api.schemas.errors import ErrorBody, ErrorResponse
from portfolio_rag.core.errors import ErrorCode
from portfolio_rag.core.logging import get_logger
from portfolio_rag.core.request_context import get_request_id

_logger = get_logger(__name__)

#: The header the gateway sets on every request it forwards. Not a standard
#: one, and deliberately not `Authorization`: nothing here is a user
#: credential, and reusing that name invites a client to try sending one.
EDGE_AUTH_HEADER: Final = "X-Edge-Auth"

#: Ceiling on the body of one question. A question is a sentence; 16 KiB is
#: already far more than the 2000 characters the schema accepts, and the slack
#: is there so that a legitimate request never trips this instead of getting a
#: field-level validation error.
DEFAULT_MAX_BODY_BYTES: Final = 16 * 1024

_FORBIDDEN_MESSAGE: Final = "This endpoint is not directly accessible."
_TOO_LARGE_MESSAGE: Final = "The request body is too large."

_HTTP_403_FORBIDDEN: Final = 403
_HTTP_413_CONTENT_TOO_LARGE: Final = 413


def _refusal(code: ErrorCode, message: str, status_code: int) -> JSONResponse:
    """Build the same envelope the error handlers produce, from middleware."""
    body = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=get_request_id()))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))


def _header(scope: Scope, name: str) -> str | None:
    wanted = name.lower().encode("latin-1")
    headers: Iterable[tuple[bytes, bytes]] = scope.get("headers", ())
    for key, value in headers:
        if key == wanted:
            return value.decode("latin-1")
    return None


def _is_protected(scope: Scope, prefixes: tuple[str, ...]) -> bool:
    path = scope.get("path", "")
    return any(path.startswith(prefix) for prefix in prefixes)


class EdgeAuthMiddleware:
    """Refuse requests to the protected paths that did not come via the gateway.

    Inert unless it was given a secret, which is what keeps local development
    and the test suite on the ordinary path: the composition decides whether the
    guard is armed, and an unarmed guard is not installed at all.
    """

    def __init__(self, app: ASGIApp, *, secret: str, protected_prefixes: tuple[str, ...]) -> None:
        if not secret:
            raise ValueError("an edge shared secret is required to arm this guard")
        self._app = app
        self._secret = secret
        self._prefixes = protected_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_protected(scope, self._prefixes):
            await self._app(scope, receive, send)
            return

        # A browser preflight carries no custom headers by definition, and the
        # CORS layer answers it without ever reaching the route. Refusing it
        # here would break the very clients the gateway is for.
        if scope.get("method") == "OPTIONS":
            await self._app(scope, receive, send)
            return

        if not self._is_from_the_gateway(_header(scope, EDGE_AUTH_HEADER)):
            # Counted, never quoted: the presented value is an attacker's
            # input, and a rejected secret is still a secret.
            _logger.warning(
                "request refused at the edge boundary",
                extra={"http_path": scope.get("path", ""), "reason": "edge_auth"},
            )
            await _refusal(ErrorCode.FORBIDDEN, _FORBIDDEN_MESSAGE, _HTTP_403_FORBIDDEN)(
                scope, receive, send
            )
            return

        await self._app(scope, receive, send)

    def _is_from_the_gateway(self, presented: str | None) -> bool:
        """Compare in constant time, so the answer leaks no prefix of the secret."""
        if presented is None:
            return False
        return hmac.compare_digest(presented, self._secret)


class BodySizeLimitMiddleware:
    """Refuse a body larger than *max_bytes* on the protected paths.

    Two checks, because one is not enough. ``Content-Length`` is what a
    well-behaved client sends and lets the request be refused before a single
    byte of body is read — but it is a claim, and a chunked request carries no
    such header at all. So the body is also read here, up to the limit, and a
    request that goes over is answered before the application is called.

    Reading it here is affordable precisely because the limit is small: at 16
    KiB the buffer is nothing, and it buys a real ``413`` in the chunked case
    instead of a route that fails halfway through a body it should never have
    been offered.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_bytes: int = DEFAULT_MAX_BODY_BYTES,
        protected_prefixes: tuple[str, ...],
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._app = app
        self._max_bytes = max_bytes
        self._prefixes = protected_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_protected(scope, self._prefixes):
            await self._app(scope, receive, send)
            return

        if self._declares_too_much(_header(scope, "content-length")):
            await self._refuse(scope, receive, send, reason="content_length")
            return

        buffered, disconnected = [], False
        received = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                # A disconnect, most likely. Keep it in the replay so the
                # application sees exactly what it would have seen.
                buffered.append(message)
                disconnected = True
                break
            received += len(message.get("body", b""))
            if received > self._max_bytes:
                await self._refuse(scope, receive, send, reason="body_too_large")
                return
            buffered.append(message)
            if not message.get("more_body", False):
                break

        await self._app(scope, _replaying(buffered, receive, exhausted=disconnected), send)

    def _declares_too_much(self, content_length: str | None) -> bool:
        if content_length is None:
            return False
        try:
            return int(content_length) > self._max_bytes
        except ValueError:
            # An unparsable length is a malformed request, not an oversized
            # one. The server decides what to do with it; this guard does not
            # invent a verdict it cannot support.
            return False

    async def _refuse(self, scope: Scope, receive: Receive, send: Send, *, reason: str) -> None:
        _logger.warning(
            "request refused for size",
            extra={"http_path": scope.get("path", ""), "reason": reason},
        )
        await _refusal(
            ErrorCode.PAYLOAD_TOO_LARGE, _TOO_LARGE_MESSAGE, _HTTP_413_CONTENT_TOO_LARGE
        )(scope, receive, send)


def _replaying(buffered: list[Message], receive: Receive, *, exhausted: bool) -> Receive:
    """Hand the application the body this guard already read, then get out of the way."""
    pending = list(buffered)

    async def replay() -> Message:
        if pending:
            return pending.pop(0)
        if exhausted:
            return {"type": "http.disconnect"}
        return await receive()

    return replay
