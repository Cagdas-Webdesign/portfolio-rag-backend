"""What one request to the generation provider looked like, without what it said.

Every call the query side makes through :class:`~portfolio_rag.ports.llm.LLMProvider`
— each generation attempt and each grounding check — leaves exactly one
:class:`ProviderCallRecord`, whether the call succeeded, produced an unusable
reply or failed outright. Successful calls matter as much as failed ones: a
reply of 800 tokens and 800 characters means something only next to what an
ordinary reply measures.

**Metadata, never content.** A record holds sizes, counts, durations, the
model's own stop reason and the fixed words this backend uses to name a broken
rule. It never holds a prompt, a passage, a question, an answer, a reply or
anything from the conversation — there is no field that could carry one.

**Observation, not policy.** Nothing here decides anything. Records are built
from values the pipeline already had, after it had acted on them; routing,
retries and refusals are exactly what they were without this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from portfolio_rag.ports.llm import GenerationResponse, ResponseFormat
from portfolio_rag.rag.errors import GenerationFailure, ProviderCallType, visible_characters

__all__ = [
    "CallResult",
    "ProviderCallRecord",
    "ProviderCallType",
    "failed_call",
    "replied_call",
]


class CallResult(StrEnum):
    """How a call ended, as coarsely as the pipeline distinguishes it.

    The finer reason — which parser rule a reply broke, which transport failure
    occurred — is the failure's ``detail``, in the words the parser and the
    adapters already use. This is not a second classification of it.
    """

    PARSED = "parsed"
    """A reply arrived and satisfied its contract. For a grounding check that
    includes ``not_supported``: a verdict the backend could read."""

    UNUSABLE_REPLY = "unusable_reply"
    """A reply arrived and broke its contract."""

    PROVIDER_ERROR = "provider_error"
    """No reply arrived: the provider failed or its response was not a
    completion."""


@dataclass(frozen=True, slots=True)
class ProviderCallRecord:
    """One call to the generation provider. Technical facts only."""

    call_type: ProviderCallType
    attempt: int
    """1 for the first call of its type in this answer; 2 for its one
    recovery — a regeneration, or a repeated grounding check."""

    model: str
    response_format: ResponseFormat
    elapsed_seconds: float
    """Service-observed elapsed time of one ``LLMProvider.generate`` call:
    everything below the port, as one number. That includes an adapter's
    transport retries and, in a paced evaluation, the pacing wait before the
    request — which can dominate it. It is **not** network, provider or
    inference latency, and must not be read as one."""

    result: CallResult
    max_output_tokens: int | None = None
    """The output cap this call requested: the reserve for a first attempt,
    the larger recovery cap for a recovery after a reply stopped at the limit.
    ``None`` when the request set none."""

    finish_reason: str | None = None
    input_tokens: int | None = None
    """As the provider reported it. ``None`` when it reported nothing — never
    an estimate."""

    output_tokens: int | None = None
    reply_characters: int | None = None
    """``None`` when no reply arrived, which is different from an empty one."""

    reply_visible_characters: int | None = None
    failure: GenerationFailure | None = None
    """Why the call could not be used, when it could not. ``None`` for a
    parsed reply."""

    @property
    def is_regeneration(self) -> bool:
        return self.call_type is ProviderCallType.GENERATION and self.attempt > 1

    @property
    def is_recovery(self) -> bool:
        """The second call of its step: a regeneration or a repeated check."""
        return self.attempt > 1

    @property
    def usage_reported(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    def fields(self) -> dict[str, str | int | float | bool | None]:
        """The record as plain values, for a log line or an export."""
        return {
            "call_type": self.call_type.value,
            "attempt": self.attempt,
            "regeneration": self.is_regeneration,
            "recovery": self.is_recovery,
            "model": self.model,
            "response_format": self.response_format.value,
            "elapsed_seconds": self.elapsed_seconds,
            "result": self.result.value,
            "max_output_tokens": self.max_output_tokens,
            "failure_category": self.failure.category.value if self.failure else None,
            "failure_detail": self.failure.detail if self.failure else None,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reply_characters": self.reply_characters,
            "reply_visible_characters": self.reply_visible_characters,
        }


def replied_call(
    *,
    call_type: ProviderCallType,
    attempt: int,
    model: str,
    response_format: ResponseFormat,
    elapsed_seconds: float,
    response: GenerationResponse,
    failure: GenerationFailure | None,
    max_output_tokens: int | None = None,
) -> ProviderCallRecord:
    """The record of a call that returned a reply, usable or not."""
    usage = response.usage
    return ProviderCallRecord(
        call_type=call_type,
        attempt=attempt,
        model=model,
        response_format=response_format,
        elapsed_seconds=elapsed_seconds,
        result=CallResult.PARSED if failure is None else CallResult.UNUSABLE_REPLY,
        max_output_tokens=max_output_tokens,
        finish_reason=response.finish_reason,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        reply_characters=len(response.text),
        reply_visible_characters=visible_characters(response.text),
        failure=failure,
    )


def failed_call(
    *,
    call_type: ProviderCallType,
    attempt: int,
    model: str,
    response_format: ResponseFormat,
    elapsed_seconds: float,
    failure: GenerationFailure,
    max_output_tokens: int | None = None,
) -> ProviderCallRecord:
    """The record of a call that returned no reply at all.

    A response that was not a completion may still have said why it ended and
    what it cost; whatever the adapter could read is on *failure* and is kept.
    """
    return ProviderCallRecord(
        call_type=call_type,
        attempt=attempt,
        model=model,
        response_format=response_format,
        elapsed_seconds=elapsed_seconds,
        result=CallResult.PROVIDER_ERROR,
        max_output_tokens=max_output_tokens,
        finish_reason=failure.finish_reason,
        input_tokens=failure.input_tokens,
        output_tokens=failure.output_tokens,
        failure=failure,
    )
