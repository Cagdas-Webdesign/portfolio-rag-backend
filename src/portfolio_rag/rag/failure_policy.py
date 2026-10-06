"""What the pipeline does about each way a provider call can fail.

One row per :class:`~portfolio_rag.rag.errors.GenerationFailureCategory`, and
one place that says, for each: who detects it, whether a transport retry is
allowed beneath the port, whether the step that failed — the generation or the
grounding check — may be asked once more, and what the request ends as.
:mod:`portfolio_rag.rag.service` reads its recovery decision from here; the
adapters' transport behaviour is what the ``transport_retry`` column
describes, and the fault-injection tests prove each adapter does exactly that.
The failure matrix test asserts that every category has its row — a category
without one is a build failure, not a gap found in production.

**Every row ends as a technical error** — after its one recovery, where it has
one. That is an invariant, not an oversight: a failure here means the provider
did not deliver something the pipeline could judge, so it is never a statement
about the knowledge base. The outcomes that *are* such statements — the model
declining, a citation that does not resolve, a grounding check that reads
``not_supported`` — are not failures at all; they end as
:attr:`AnswerOutcome.NOT_GROUNDED` through the answer path, and are quality
findings. One root cause, one terminal outcome, one gate.

**Classification precedence** (:func:`classify_unusable_reply` and
:func:`~portfolio_rag.rag.errors.provider_failure`), strongest evidence first:

1. No reply at all — the port raised — is classified from what the adapter
   set: timeout, malformed response, retryable, refused status, unclassified.
2. A reply the parser accepts is not a failure, whatever its finish reason.
3. A rejected reply the provider says it cut off at the output limit is
   :attr:`OUTPUT_TRUNCATED`; the parser's rule stays as the detail.
4. Any other rejected reply is :attr:`UNPARSEABLE_OUTPUT`.

**A recovery is one more request of the same step, never a third.** Both
steps allow it for the same two categories: a response that was not a
completion, and a reply cut off at the output limit. When the provider said it
stopped at the limit (``finish_reason`` ``length``), the second request may
spend more output — :func:`needs_larger_output`; the first request of a step
never does. Measured against `@cf/openai/gpt-oss-120b`, a reasoning model
whose reasoning counts against the cap: both kinds of failure arrive with
``length`` and the whole cap spent, the first with no text at all.

A reply made mostly of whitespace is not a category. It is visible in the
telemetry (``reply_visible_characters``) and gets the reaction its category
already has; it becomes a row of its own only when evidence calls for a
different reaction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from portfolio_rag.rag.errors import GenerationFailure, GenerationFailureCategory

#: What a provider says when it stopped because the output limit was reached.
#: The de-facto name across chat completion APIs.
TRUNCATED_FINISH_REASON: Final = "length"


class Owner(StrEnum):
    """Which layer detects the failure. Descriptive: nothing branches on it."""

    ADAPTER = "infrastructure_adapter"
    """Transport, status codes and the provider's response shape."""

    PARSER = "rag_contract"
    """The answer contract (`rag.generation`) or the verdict contract
    (`rag.verification`) the reply was held to."""

    PROVIDER_BOUNDARY = "rag_service_boundary"
    """The answering service, where an adapter raised something the port does
    not define, or the request's deadline passed."""


class TransportRetry(StrEnum):
    """Whether the adapter retries beneath the port before the failure is final."""

    BOUNDED = "bounded"
    """The adapter's own retries, at most its configured attempts."""

    NONE = "none"


class Terminal(StrEnum):
    TECHNICAL_ERROR = "technical_error"
    """The request fails closed: nothing is published, the client gets the
    shared error envelope with a 503, and an evaluation counts a pipeline
    error."""


class Gate(StrEnum):
    AVAILABILITY = "availability"


@dataclass(frozen=True, slots=True)
class FailurePolicy:
    """What one category of failure makes the pipeline do."""

    owner: Owner
    transport_retry: TransportRetry
    recovery: bool
    """Whether the step that failed may be asked once more — the answer
    generated again, or the grounding check repeated. Never a third time."""

    terminal: Terminal = Terminal.TECHNICAL_ERROR
    gate: Gate = Gate.AVAILABILITY


_C = GenerationFailureCategory

FAILURE_POLICY: Final = MappingProxyType(
    {
        _C.TIMEOUT: FailurePolicy(Owner.ADAPTER, TransportRetry.BOUNDED, False),
        _C.RETRYABLE_PROVIDER_ERROR: FailurePolicy(Owner.ADAPTER, TransportRetry.BOUNDED, False),
        _C.PROVIDER_STATUS: FailurePolicy(Owner.ADAPTER, TransportRetry.NONE, False),
        # The provider was reached and its response was not a completion.
        # Nothing was generated, so a fresh sample is the remedy — for a
        # generation and for a grounding check alike.
        _C.MALFORMED_RESPONSE: FailurePolicy(Owner.ADAPTER, TransportRetry.NONE, True),
        # Cut off before the object closed: a property of the sample, measured
        # against the same model answering the same question in full.
        _C.OUTPUT_TRUNCATED: FailurePolicy(Owner.PARSER, TransportRetry.NONE, True),
        # The model finished and ignored the contract: asking again is asking
        # it to ignore it again.
        _C.UNPARSEABLE_OUTPUT: FailurePolicy(Owner.PARSER, TransportRetry.NONE, False),
        _C.UNCLASSIFIED: FailurePolicy(Owner.PROVIDER_BOUNDARY, TransportRetry.NONE, False),
    }
)


def policy_for(failure: GenerationFailure) -> FailurePolicy:
    return FAILURE_POLICY[failure.category]


def may_recover(failure: GenerationFailure) -> bool:
    """Whether *failure* allows its step to be asked once more.

    The same rule for both steps. A grounding check that delivered a verdict —
    ``supported`` or ``not_supported`` — never gets here: it is no failure. The
    caller enforces "once": a second failure of the same step is terminal.
    """
    return policy_for(failure).recovery


def needs_larger_output(failure: GenerationFailure) -> bool:
    """Whether the recovery of *failure* should get the larger output cap.

    Only when the provider said it stopped at the output limit: then the same
    cap is the likeliest reason for the same failure. Any other recoverable
    failure is asked again as it was.
    """
    return (
        failure.finish_reason == TRUNCATED_FINISH_REASON
        or failure.category is GenerationFailureCategory.OUTPUT_TRUNCATED
    )


def classify_unusable_reply(
    failure: GenerationFailure, *, finish_reason: str | None, cut_off: bool = False
) -> GenerationFailure:
    """Place a reply the contract rejected: truncated at the limit, or not.

    *failure* is what the contract reported — its rule is kept as the detail.
    Precedence 3 and 4 of the module docstring. A reply is truncated when the
    provider says it stopped at the limit, or — when the provider says
    nothing about why it stopped — when the reply itself is a JSON object
    that never closes (*cut_off*, :func:`~portfolio_rag.rag.generation.cut_off_json`).
    A finish reason other than the limit is believed: a reply that says it
    finished and broke the contract is unparseable, however it ends.
    """
    truncated = finish_reason == TRUNCATED_FINISH_REASON or (finish_reason is None and cut_off)
    category = (
        GenerationFailureCategory.OUTPUT_TRUNCATED
        if truncated
        else GenerationFailureCategory.UNPARSEABLE_OUTPUT
    )
    return replace(failure, category=category)
