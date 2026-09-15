"""Application error taxonomy.

Errors carry a stable, machine-readable :class:`ErrorCode` — clients branch on
the code, never on the human-readable message. The mapping from code to HTTP
status lives in the API layer (``portfolio_rag.api.error_handlers``) so this
module stays transport-agnostic.

Only codes the application can actually produce today are listed. New codes are
added together with the behaviour that raises them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class ErrorCode(StrEnum):
    """Stable error identifiers that form part of the public API contract."""

    VALIDATION_ERROR = "VALIDATION_ERROR"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    GENERATION_UNAVAILABLE = "GENERATION_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Base class for errors that are safe to surface to API clients.

    The ``message`` of an :class:`AppError` is returned verbatim in the
    response body, so it must never contain internal details, user content or
    secrets. Anything that is *not* an ``AppError`` is reported to the client
    as a generic ``INTERNAL_ERROR``.
    """

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR
    default_message: ClassVar[str] = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class UpstreamUnavailableError(AppError):
    """A service this request depends on could not be used.

    The distinction from :class:`AppError` itself is what a client can do about
    it: an upstream failure is worth retrying, an internal error is not. The
    *cause* — which provider, which status code — is logged and never sent.
    """

    default_message = "A service this request depends on is temporarily unavailable."
