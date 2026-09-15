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

from typing import ClassVar

from portfolio_rag.core.errors import AppError, ErrorCode, UpstreamUnavailableError


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


class GenerationUnavailableError(UpstreamUnavailableError):
    """The generation provider failed, or answered with something unusable.

    One code for both because the client's options are identical: a provider
    that returns malformed JSON and one that returns a 503 are equally
    unavailable from here, and splitting them would give a caller a decision it
    has no way to act on.
    """

    code: ClassVar[ErrorCode] = ErrorCode.GENERATION_UNAVAILABLE
    default_message = "Answer generation is temporarily unavailable."


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
