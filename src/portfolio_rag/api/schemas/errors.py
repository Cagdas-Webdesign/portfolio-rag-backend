"""The single error envelope used by every failing API response.

Clients can rely on this shape for *all* non-2xx responses, including
validation failures and unexpected server errors::

    {"error": {"code": "VALIDATION_ERROR", "message": "...", "request_id": "..."}}
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from portfolio_rag.core.errors import ErrorCode


class ErrorDetail(BaseModel):
    """One field-level problem. Only ever produced for validation errors."""

    model_config = ConfigDict(extra="forbid")

    field: str = Field(description="Dotted path to the offending field, e.g. `body.message`.")
    message: str = Field(description="What is wrong with that field.")


class ErrorBody(BaseModel):
    """Machine-readable description of a failure."""

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode = Field(description="Stable identifier — branch on this, not on `message`.")
    message: str = Field(description="Human-readable summary. Never contains internal details.")
    request_id: str | None = Field(
        default=None,
        description="Correlation id of the request; also returned in the `X-Request-ID` header.",
    )
    details: list[ErrorDetail] | None = Field(
        default=None,
        description="Field-level problems. Present only for `VALIDATION_ERROR`.",
    )


class ErrorResponse(BaseModel):
    """Envelope returned by every error response."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "error": {
                        "code": "GENERATION_UNAVAILABLE",
                        "message": "Answer generation is temporarily unavailable.",
                        "request_id": "0f9a1c2b3d4e5f60718293a4b5c6d7e8",
                    }
                }
            ]
        },
    )

    error: ErrorBody
