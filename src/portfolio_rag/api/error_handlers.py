"""Translation of exceptions into the public error envelope.

Every failure — expected or not — leaves the application as an
:class:`~portfolio_rag.api.schemas.errors.ErrorResponse`. Two rules govern what
goes into it:

* the ``code`` is stable and machine-readable;
* the ``message`` is written for a client, never derived from an exception's
  string representation, so internals (paths, SQL, provider payloads,
  tracebacks) cannot leak. Tracebacks go to the log, correlated by request id.
"""

from __future__ import annotations

from typing import Any, Final, cast

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from portfolio_rag.api.schemas.errors import ErrorBody, ErrorDetail, ErrorResponse
from portfolio_rag.core.errors import AppError, ErrorCode
from portfolio_rag.core.logging import get_logger
from portfolio_rag.core.request_context import REQUEST_ID_HEADER, get_request_id

_logger = get_logger(__name__)

#: Starlette has renamed its 422 and 413 constants across versions; the status
#: *number* is the contract, so each is named once here and imported from this
#: module.
HTTP_422_UNPROCESSABLE_CONTENT: Final = 422
HTTP_413_CONTENT_TOO_LARGE: Final = 413

_STATUS_BY_CODE: Final[dict[ErrorCode, int]] = {
    ErrorCode.VALIDATION_ERROR: HTTP_422_UNPROCESSABLE_CONTENT,
    ErrorCode.FORBIDDEN: status.HTTP_403_FORBIDDEN,
    ErrorCode.NOT_FOUND: status.HTTP_404_NOT_FOUND,
    ErrorCode.METHOD_NOT_ALLOWED: status.HTTP_405_METHOD_NOT_ALLOWED,
    ErrorCode.PAYLOAD_TOO_LARGE: HTTP_413_CONTENT_TOO_LARGE,
    ErrorCode.RETRIEVAL_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.GENERATION_UNAVAILABLE: status.HTTP_503_SERVICE_UNAVAILABLE,
    ErrorCode.INTERNAL_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
}

#: Client-safe wording for the HTTP errors Starlette raises on our behalf.
_HTTP_ERRORS: Final[dict[int, tuple[ErrorCode, str]]] = {
    status.HTTP_404_NOT_FOUND: (ErrorCode.NOT_FOUND, "The requested resource does not exist."),
    status.HTTP_405_METHOD_NOT_ALLOWED: (
        ErrorCode.METHOD_NOT_ALLOWED,
        "This method is not allowed for the requested resource.",
    ),
}

_GENERIC_ERROR: Final = "An unexpected error occurred."


def _request_id_of(request: Request) -> str | None:
    """Resolve the correlation id.

    The context variable is the normal source, but the handler for unexpected
    exceptions runs in Starlette's outermost middleware — by then the request
    context has already been unwound. The id is therefore also kept on the
    request scope, which survives that unwinding.
    """
    from_scope = getattr(request.state, "request_id", None)
    return from_scope if isinstance(from_scope, str) else get_request_id()


def _error_response(
    request: Request,
    *,
    code: ErrorCode,
    message: str,
    details: list[ErrorDetail] | None = None,
    status_code: int | None = None,
) -> JSONResponse:
    request_id = _request_id_of(request)
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request_id, details=details)
    )
    headers = {REQUEST_ID_HEADER: request_id} if request_id else None
    return JSONResponse(
        status_code=status_code or _STATUS_BY_CODE[code],
        content=body.model_dump(mode="json", exclude_none=True),
        headers=headers,
    )


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Expected, self-described failures."""
    return _error_response(request, code=exc.code, message=exc.message)


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Malformed input: report *which* field is wrong, never echo its value."""
    details = [
        ErrorDetail(
            field=".".join(str(part) for part in error["loc"]),
            message=str(error["msg"]),
        )
        for error in exc.errors()
    ]
    return _error_response(
        request,
        code=ErrorCode.VALIDATION_ERROR,
        message="The request payload is invalid.",
        details=details,
    )


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Routing-level failures raised by Starlette (404, 405, …)."""
    code, message = _HTTP_ERRORS.get(exc.status_code, (ErrorCode.INTERNAL_ERROR, _GENERIC_ERROR))
    return _error_response(request, code=code, message=message, status_code=exc.status_code)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log the full traceback, tell the client nothing about it."""
    _logger.exception("unhandled exception", exc_info=exc)
    return _error_response(request, code=ErrorCode.INTERNAL_ERROR, message=_GENERIC_ERROR)


def register_error_handlers(app: FastAPI) -> None:
    """Wire the handlers above into *app*.

    Starlette types its handler registry as ``(Request, Exception) -> Response``.
    The handlers above are annotated with the exception type they actually
    receive, which is more useful to read and to type-check; the casts here are
    the single place where that narrowing is asserted.
    """
    app.add_exception_handler(AppError, cast("Any", handle_app_error))
    app.add_exception_handler(RequestValidationError, cast("Any", handle_validation_error))
    app.add_exception_handler(StarletteHTTPException, cast("Any", handle_http_exception))
    app.add_exception_handler(Exception, cast("Any", handle_unexpected_error))
