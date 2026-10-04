"""Failures on the query side.

Unlike the ingestion and indexing taxonomies, these **do** become HTTP
responses — a user is waiting for one — so they extend :class:`AppError` and
carry a client-safe message. The cause (which provider, which status, which
space) is logged where it is raised and never travels with the exception.

Deliberately short. The distinctions that exist are the ones a caller acts on:

* the question itself is unusable          → the client fixes it     (422)
* the knowledge side could not be reached  → retry later             (503)
* the generation side could not be reached → retry later             (503)
* the request broke an internal invariant  → nobody outside can help (500)

Note what is *not* here: "no relevant knowledge" and "the answer was not
grounded". Neither is an error. A knowledge base that does not cover a question
is a normal, successful, honest answer — see
:mod:`portfolio_rag.rag.service`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from portfolio_rag.core.errors import AppError, ErrorCode, UpstreamUnavailableError
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind

if TYPE_CHECKING:
    from portfolio_rag.rag.retrieval import RetrievalOutcome
    from portfolio_rag.rag.telemetry import ProviderCallRecord


class QueryValidationError(AppError):
    """The question cannot be processed as asked.

    Message is client-safe by construction: it describes the *rule* that was
    broken (empty, too long) and never echoes the question back.
    """

    code: ClassVar[ErrorCode] = ErrorCode.VALIDATION_ERROR
    default_message = "The question is empty or too long."


class QueryEmbeddingError(UpstreamUnavailableError):
    """The question could not be turned into a vector."""

    code: ClassVar[ErrorCode] = ErrorCode.RETRIEVAL_UNAVAILABLE
    default_message = "The knowledge search is temporarily unavailable."


class RetrievalUnavailableError(UpstreamUnavailableError):
    """The vector index could not be searched."""

    code: ClassVar[ErrorCode] = ErrorCode.RETRIEVAL_UNAVAILABLE
    default_message = "The knowledge search is temporarily unavailable."


class ProviderCallType(StrEnum):
    """Which step of the pipeline called the generation provider."""

    GENERATION = "generation"
    GROUNDING_CHECK = "grounding_check"


class GenerationFailureCategory(StrEnum):
    """Why a call to the generation provider could not be used.

    One category per *reaction*: two failures share a category exactly when
    the pipeline must do the same thing about them. What each category makes
    the pipeline do is decided in one place, :mod:`portfolio_rag.rag.failure_policy`;
    the finer cause — the parser rule, the transport failure — is the
    failure's ``detail``.
    """

    PROVIDER_STATUS = "provider_status"
    """The provider refused the request with a status that is not worth retrying."""

    TIMEOUT = "timeout"
    """A provider request timed out, or the request's own deadline passed."""

    RETRYABLE_PROVIDER_ERROR = "retryable_provider_error"
    """A rate limit, a transient status or an unreachable provider."""

    MALFORMED_RESPONSE = "malformed_response"
    """The provider answered, and the response was not a completion."""

    OUTPUT_TRUNCATED = "output_truncated"
    """The reply broke its contract and the provider says it stopped at the
    output limit: the reply was cut off before the object closed."""

    UNPARSEABLE_OUTPUT = "unparseable_output"
    """The reply broke its contract although the model finished."""

    UNCLASSIFIED = "unclassified"
    """Nothing above describes it: an unexpected failure at the provider
    boundary. Treated as technical and never retried."""


@dataclass(frozen=True, slots=True)
class GenerationFailure:
    """What is known about one failed generation. Technical facts only.

    For a developer, a log line and an evaluation export — never for a client,
    whose response is the same sentence whatever happened here. Nothing in it is
    content: no prompt, no passage, no reply text, no header, no credential.
    ``detail`` is either a fixed word naming the rule a reply broke or the
    message an adapter composed from a status code.
    """

    category: GenerationFailureCategory
    detail: str
    step: ProviderCallType = ProviderCallType.GENERATION
    """Which call failed: the generation or the grounding check."""

    status_code: int | None = None
    retryable: bool | None = None
    attempts: int = 1
    retry_after_seconds: float | None = None
    """How long the provider asked to be left alone, when it said so."""

    finish_reason: str | None = None
    reply_characters: int | None = None
    reply_visible_characters: int | None = None
    """Characters of the reply that are not whitespace. Beside
    :attr:`reply_characters` and :attr:`output_tokens` it tells a long answer
    cut off at the limit apart from a reply that spent the limit on blanks —
    two problems with different fixes — without a character of it being
    kept."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    """Token usage as the provider reported it — for an unusable reply, and for
    a response that was not a completion but still carried usage. ``None``
    when nothing was reported, never an estimate."""

    def fields(self) -> dict[str, str | int | float | bool | None]:
        """The same facts as plain values, for a log record or an export."""
        return {
            "failure_category": self.category.value,
            "failure_detail": self.detail,
            "failure_step": self.step.value,
            "status_code": self.status_code,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "retry_after_seconds": self.retry_after_seconds,
            "finish_reason": self.finish_reason,
            "reply_characters": self.reply_characters,
            "reply_visible_characters": self.reply_visible_characters,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def visible_characters(text: str) -> int:
    """How many characters of *text* are not whitespace. A count, never content."""
    return sum(1 for character in text if not character.isspace())


def provider_failure(
    error: LLMProviderError, *, step: ProviderCallType = ProviderCallType.GENERATION
) -> GenerationFailure:
    """Describe a port-level failure without naming a vendor.

    The category follows from facts the adapter set, in this order: a timeout;
    a response that was not a completion; anything the adapter judged worth
    retrying; a refused status; and, failing all of those, unclassified.
    """
    if error.kind is ProviderFailureKind.TIMEOUT:
        category = GenerationFailureCategory.TIMEOUT
    elif error.kind is ProviderFailureKind.MALFORMED_RESPONSE:
        category = GenerationFailureCategory.MALFORMED_RESPONSE
    elif error.retryable or error.kind is ProviderFailureKind.RATE_LIMITED:
        category = GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR
    elif error.kind is ProviderFailureKind.HTTP_STATUS:
        category = GenerationFailureCategory.PROVIDER_STATUS
    else:
        category = GenerationFailureCategory.UNCLASSIFIED
    usage = error.usage
    return GenerationFailure(
        category=category,
        detail=error.kind.value,
        step=step,
        status_code=error.status_code,
        retryable=error.retryable,
        attempts=error.attempts,
        retry_after_seconds=error.retry_after_seconds,
        # What a provider reported about a response that was not a completion.
        # Vendor-neutral by construction: the adapter has already read them.
        finish_reason=error.finish_reason,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
    )


class GenerationUnavailableError(UpstreamUnavailableError):
    """The generation provider failed, or answered with something unusable.

    One code for both because the client's options are identical: a provider
    that returns malformed JSON and one that returns a 503 are equally
    unavailable from here, and splitting them would give a caller a decision it
    has no way to act on.

    ``failure``, ``retrieval`` and ``provider_calls`` are what a developer
    needs afterwards and a client never sees: why the generation failed, what
    had been retrieved for the question before it did, and every provider call
    the question had made up to and including the failing one. None of them is
    part of the message, and the HTTP response is built from the code and the
    message alone.
    """

    code: ClassVar[ErrorCode] = ErrorCode.GENERATION_UNAVAILABLE
    default_message = "Answer generation is temporarily unavailable."

    def __init__(
        self, message: str | None = None, *, failure: GenerationFailure | None = None
    ) -> None:
        super().__init__(message)
        self.failure = failure
        self.retrieval: RetrievalOutcome | None = None
        self.provider_calls: tuple[ProviderCallRecord, ...] = ()


class EmbeddingSpaceMismatchError(AppError):
    """The query space and the index space are not the same space.

    A configuration fault, not a transient one: retrying changes nothing, and
    searching anyway would return confident nonsense. Equal dimensionality is
    explicitly not equal identity.
    """

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR
    default_message = "The knowledge index is not configured correctly."


class ContextBudgetError(AppError):
    """Not a single retrieved passage fits in the context budget.

    Also a configuration fault: chunks are bounded at ingestion time, so a
    corpus whose smallest unit does not fit means the budget and the chunking
    policy disagree. Truncating silently would be the other option, and a
    half-sentence of evidence is worse than none.
    """

    code: ClassVar[ErrorCode] = ErrorCode.INTERNAL_ERROR
    default_message = "The answer context could not be assembled."
