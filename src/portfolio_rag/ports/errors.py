"""Failures an adapter behind a port may raise.

These live with the ports rather than with any particular adapter, because they
are part of the contract: application code catches *these*, and never a
provider SDK's exception type or an HTTP library's. That is what keeps the core
free of vendor knowledge.

Deliberately coarse. "The provider did not give us usable embeddings" is the
distinction that changes what a caller does; whether the cause was a 429 or a
malformed body is detail for the message, not a separate class to catch.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from portfolio_rag.ports.llm import TokenUsage


class PortErrorCode(StrEnum):
    """Stable, machine-readable reasons an adapter failed."""

    EMBEDDING_PROVIDER_ERROR = "EMBEDDING_PROVIDER_ERROR"
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    VECTOR_STORE_ERROR = "VECTOR_STORE_ERROR"
    OPERATION_NOT_SUPPORTED = "OPERATION_NOT_SUPPORTED"


class PortError(Exception):
    """Base for adapter failures.

    ``message`` is written for a developer reading a terminal. It may name the
    provider and the operation; it must never contain a credential, an
    ``Authorization`` header or a raw response body, any of which can carry
    secrets or user content.

    ``retry_after_seconds`` carries how long the provider asked the caller to
    wait, when it said so. Reporting it and honouring it are deliberately
    separate: an adapter's own retry budget is a per-request transport
    concern with a caller waiting on the other end, while a wait measured in
    tens of seconds only makes sense to something running a batch. The adapter
    passes the number on; whoever is in a position to wait that long decides
    whether to.
    """

    code: ClassVar[PortErrorCode] = PortErrorCode.VECTOR_STORE_ERROR

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
    ) -> None:
        self.message = message
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        super().__init__(message)

    def describe(self) -> str:
        """One-line rendering for developer output."""
        return f"{self.code.value}: {self.message}"


class ProviderFailureKind(StrEnum):
    """What a generation adapter ran into. Data on the error, not a class to catch.

    The reasoning at the top of this module still holds: a caller does the same
    thing whatever the cause, so there is one error class. But "the provider
    failed" is not enough to act on afterwards — a timeout, a rate limit and a
    refused request need three different fixes — and reading the cause back out
    of a message string is how a diagnosis starts depending on wording.
    """

    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    RATE_LIMITED = "rate_limited"
    HTTP_STATUS = "http_status"
    MALFORMED_RESPONSE = "malformed_response"
    """The provider answered, and the body was not a completion."""

    UNSPECIFIED = "unspecified"


class EmbeddingProviderError(PortError):
    """An embedding provider could not deliver usable vectors.

    ``kind`` and ``status_code``: what happened on the wire, when the adapter
    knows. Diagnostics, like the message — never a body or a credential.
    """

    code = PortErrorCode.EMBEDDING_PROVIDER_ERROR

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        kind: ProviderFailureKind = ProviderFailureKind.UNSPECIFIED,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable, retry_after_seconds=retry_after_seconds)
        self.kind = kind
        self.status_code = status_code


class LLMProviderError(PortError):
    """A generation provider could not deliver a usable completion.

    "Usable" stops at the transport: a provider that answered with text has
    succeeded here, even if the text turns out not to satisfy the answer
    contract. Judging the content is the query side's job.

    ``kind``, ``status_code`` and ``attempts`` are diagnostics. They say what
    happened on the wire and how often it was tried; like the message, they
    never carry a credential, a header or a response body.

    ``provider_error_code`` and ``rate_limit_kind`` identify which of a
    provider's documented failures it was — for a 429, a daily allocation and a
    capacity limit need different reactions. Numbers and fixed words only.

    ``finish_reason`` and ``usage`` are what a provider reported about a
    response that was nevertheless not a completion — set by an adapter only
    when the response carried them, ``None`` otherwise. Metadata, never text:
    a malformed response says why it ended and what it cost, or nothing.
    """

    code = PortErrorCode.LLM_PROVIDER_ERROR

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        kind: ProviderFailureKind = ProviderFailureKind.UNSPECIFIED,
        status_code: int | None = None,
        finish_reason: str | None = None,
        usage: TokenUsage | None = None,
        provider_error_code: int | None = None,
        rate_limit_kind: str | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable, retry_after_seconds=retry_after_seconds)
        self.kind = kind
        self.status_code = status_code
        self.finish_reason = finish_reason
        self.usage = usage
        #: The provider's own numeric error code from a failed response, when it
        #: sent one — a number, never the message beside it.
        self.provider_error_code = provider_error_code
        #: Which limit a rate-limited request hit, when the provider's code says
        #: so by its own documentation: ``daily_free_allocation_exhausted``,
        #: ``capacity_exceeded``. ``None`` when it does not — never guessed.
        self.rate_limit_kind = rate_limit_kind
        #: Requests made before this error was given up on — every HTTP
        #: request, whichever layer made it. Whoever retried writes it; an
        #: error nobody retried was tried once.
        self.attempts = 1
        #: The adapter's own transport attempts in the last round.
        self.transport_attempts = 1
        #: Rounds of a retrying decorator above the adapter (the evaluation's
        #: pacer); ``1`` when there was none.
        self.pacing_attempts = 1


class VectorStoreError(PortError):
    """A vector store could not complete an operation.

    ``kind`` and ``status_code`` are what happened on the wire, set by the
    adapter where it knows. ``retryable`` stays the adapter's judgement for
    *any* operation — a write whose outcome is unknown is never retryable —
    while ``kind`` lets a caller that only reads decide for itself.
    """

    code = PortErrorCode.VECTOR_STORE_ERROR

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        kind: ProviderFailureKind = ProviderFailureKind.UNSPECIFIED,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable, retry_after_seconds=retry_after_seconds)
        self.kind = kind
        self.status_code = status_code


class UnsupportedVectorStoreOperationError(VectorStoreError):
    """The store cannot do this at all — a capability gap, not a failure.

    Raised, for example, when something asks a store that can only fetch by id
    to enumerate its contents.
    """

    code = PortErrorCode.OPERATION_NOT_SUPPORTED
