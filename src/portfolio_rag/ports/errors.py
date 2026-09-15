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


class EmbeddingProviderError(PortError):
    """An embedding provider could not deliver usable vectors."""

    code = PortErrorCode.EMBEDDING_PROVIDER_ERROR


class LLMProviderError(PortError):
    """A generation provider could not deliver a usable completion.

    "Usable" stops at the transport: a provider that answered with text has
    succeeded here, even if the text turns out not to satisfy the answer
    contract. Judging the content is the query side's job.
    """

    code = PortErrorCode.LLM_PROVIDER_ERROR


class VectorStoreError(PortError):
    """A vector store could not complete an operation."""

    code = PortErrorCode.VECTOR_STORE_ERROR


class UnsupportedVectorStoreOperationError(VectorStoreError):
    """The store cannot do this at all — a capability gap, not a failure.

    Raised, for example, when something asks a store that can only fetch by id
    to enumerate its contents.
    """

    code = PortErrorCode.OPERATION_NOT_SUPPORTED
