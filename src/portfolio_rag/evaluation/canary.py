"""One tiny request before an acceptance run, to learn whether to start it at all.

An acceptance run that meets a 429 on its first question has spent its
preflight, its budget decision and — under the rerun rule — its one attempt,
to learn something a single request could have said. The canary is that
request: one call through the generation port, minimal input, a small output
cap, no retrieval, no grounding check, no portfolio question. When it does not
come back cleanly, the run does not start.

It is a health check of the provider path today, not a measurement of the
pipeline. It never retries beyond what the adapter beneath it does (an
evaluation builds that adapter with a single transport attempt), and it is not
repeated automatically: a canary that failed is the answer for now.

Nothing of the reply is kept but its classification and its size.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from portfolio_rag.evaluation.budget import CostProfile
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import (
    GenerationRequest,
    LLMProvider,
    MessageRole,
    PromptMessage,
    ResponseFormat,
)
from portfolio_rag.rag.generation import json_candidate

#: The output cap of the canary request: a lightweight health-check cap, not
#: proof of generation capacity — that is what the questions measure.
#:
#: Small, but not so small that a model which reasons before it answers is
#: cut off on a two-token reply. **Not validated live.** The closest measured
#: task is the grounding check, which also answers in a few tokens: across 49
#: checks it spent 116 to 786 output tokens (median 347) reasoning about 520 to 1270
#: input tokens of evidence. The canary has nothing to reason about, so it is
#: expected far below that, but if a live canary ever fails as
#: ``output_truncated`` this is the number to revisit — with that measurement,
#: not before. A canary that fails costs one small request and asks no question.
CANARY_MAX_OUTPUT_TOKENS: Final = 512

#: A generous bound on the canary's input, for its cost estimate: the two
#: short messages below are a few dozen tokens, plus the provider's chat
#: template.
CANARY_INPUT_TOKEN_BOUND: Final = 256

_SYSTEM: Final = "You are a connectivity check. Reply with one JSON object and nothing else."
_USER: Final = 'Reply with valid JSON: {"ok":true}'

_AUTH_STATUSES: Final = frozenset({401, 403})


class CanaryStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - a health-check result, not a credential
    RATE_LIMITED = "rate_limited"
    AUTH_FAILURE = "auth_failure"
    PROVIDER_FAILURE = "provider_failure"
    """A 5xx: the provider is failing on its side."""

    TIMEOUT_NETWORK = "timeout_network"
    MALFORMED_RESPONSE = "malformed_response"
    """The provider answered, and the answer was not the requested object."""

    PROVIDER_REFUSED = "provider_refused"
    """Another status the provider would not serve, or a request invalid locally."""


@dataclass(frozen=True, slots=True)
class CanaryResult:
    """What the canary found. Technical facts only — never reply text."""

    status: CanaryStatus
    status_code: int | None = None
    failure_category: str | None = None
    """The port's failure kind, or the rule the reply broke."""

    retry_after_seconds: float | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status is CanaryStatus.PASS

    @property
    def usage_reported(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None

    def fields(self) -> dict[str, Any]:
        return {
            "canary_status": self.status.value,
            "canary_status_code": self.status_code,
            "canary_failure_category": self.failure_category,
            "retry_after_seconds": self.retry_after_seconds,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "max_output_tokens": CANARY_MAX_OUTPUT_TOKENS,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def canary_request() -> GenerationRequest:
    """The one request a canary makes. Fixed text: nothing from the corpus,
    the dataset or a visitor."""
    return GenerationRequest(
        messages=(
            PromptMessage(role=MessageRole.SYSTEM, content=_SYSTEM),
            PromptMessage(role=MessageRole.USER, content=_USER),
        ),
        max_output_tokens=CANARY_MAX_OUTPUT_TOKENS,
        temperature=0.0,
        response_format=ResponseFormat.JSON_OBJECT,
    )


async def run_canary(llm: LLMProvider) -> CanaryResult:
    """Make the one canary call and classify it."""
    started = time.perf_counter()
    try:
        response = await llm.generate(canary_request())
    except LLMProviderError as exc:
        return _classified(exc, time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    usage = response.usage
    input_tokens = usage.input_tokens if usage else None
    output_tokens = usage.output_tokens if usage else None
    if not _is_ok_object(response.text):
        return CanaryResult(
            status=CanaryStatus.MALFORMED_RESPONSE,
            failure_category=(
                "output_truncated" if response.finish_reason == "length" else "reply_not_ok_json"
            ),
            finish_reason=response.finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_seconds=elapsed,
        )
    return CanaryResult(
        status=CanaryStatus.PASS,
        finish_reason=response.finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=elapsed,
    )


def canary_neurons_bound(profile: CostProfile) -> float:
    """The most the canary can cost: its input bound and its full output cap."""
    return profile.estimate(CANARY_INPUT_TOKEN_BOUND, CANARY_MAX_OUTPUT_TOKENS)


def canary_neurons(result: CanaryResult, profile: CostProfile) -> float:
    """What the canary cost by its reported usage; the bound when it reported
    none — never free, since a refused request may still have been counted."""
    if result.input_tokens is not None and result.output_tokens is not None:
        return profile.estimate(result.input_tokens, result.output_tokens)
    return canary_neurons_bound(profile)


def _classified(error: LLMProviderError, elapsed: float) -> CanaryResult:
    status_code = error.status_code
    if status_code == 429 or error.kind is ProviderFailureKind.RATE_LIMITED:
        status = CanaryStatus.RATE_LIMITED
    elif status_code in _AUTH_STATUSES:
        status = CanaryStatus.AUTH_FAILURE
    elif status_code is not None and status_code >= 500:
        status = CanaryStatus.PROVIDER_FAILURE
    elif error.kind in {ProviderFailureKind.TIMEOUT, ProviderFailureKind.UNREACHABLE}:
        status = CanaryStatus.TIMEOUT_NETWORK
    elif error.kind is ProviderFailureKind.MALFORMED_RESPONSE:
        status = CanaryStatus.MALFORMED_RESPONSE
    else:
        status = CanaryStatus.PROVIDER_REFUSED
    usage = error.usage
    truncated = status is CanaryStatus.MALFORMED_RESPONSE and error.finish_reason == "length"
    return CanaryResult(
        status=status,
        status_code=status_code,
        # A reply with no text that stopped at the cap spent it reasoning.
        failure_category="output_truncated" if truncated else error.kind.value,
        retry_after_seconds=error.retry_after_seconds,
        finish_reason=error.finish_reason,
        input_tokens=usage.input_tokens if usage else None,
        output_tokens=usage.output_tokens if usage else None,
        elapsed_seconds=elapsed,
    )


def _is_ok_object(text: str) -> bool:
    """``{"ok": true}``, read with the answer contract's own tolerance: plain
    or inside one Markdown fence, nothing else. A health check judges the
    provider, not a formatting habit production would accept."""
    try:
        value = json.loads(json_candidate(text))
    except ValueError:
        return False
    return isinstance(value, dict) and value.get("ok") is True
