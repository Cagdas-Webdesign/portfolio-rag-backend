"""Per-request correlation id, propagated without threading it through signatures.

The id is set by :mod:`portfolio_rag.api.middleware.request_id` at the edge and
read by the logging filter and the error handlers. Keeping it in a
:class:`~contextvars.ContextVar` means application and domain code stay free of
transport concerns.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar, Token
from typing import Final

#: Header used to accept and return the correlation id.
REQUEST_ID_HEADER: Final = "X-Request-ID"

#: Client-supplied ids are echoed into logs, so they are constrained to a
#: conservative character set and length to prevent log injection or unbounded
#: log growth. Anything else is replaced by a freshly generated id.
_SAFE_REQUEST_ID: Final = re.compile(r"\A[A-Za-z0-9_.:-]{8,64}\Z")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    """Generate a fresh, unique request id."""
    return uuid.uuid4().hex


def sanitize_request_id(candidate: str | None) -> str:
    """Return *candidate* if it is a safe correlation id, else a new one."""
    if candidate is not None and _SAFE_REQUEST_ID.match(candidate):
        return candidate
    return new_request_id()


def set_request_id(request_id: str) -> Token[str | None]:
    """Bind *request_id* to the current context. Reset with :func:`reset_request_id`."""
    return _request_id.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Restore the previous context value produced by :func:`set_request_id`."""
    _request_id.reset(token)


def get_request_id() -> str | None:
    """Return the current request id, or ``None`` outside of a request."""
    return _request_id.get()
