"""Keeping an evaluation run inside a provider's rate limits.

A full dataset run is a burst: forty-nine questions, one after another, as fast
as the pipeline can produce them. That is nothing like production traffic, and
it is what a free-tier account notices first — the run dies partway through
with ``GENERATION_UNAVAILABLE`` and nothing measured, while the same pipeline
serving one visitor a minute never comes close to a limit.

    question → retrieval → [nothing found]   → no generation, no waiting
                         → generation        → wait for the slot, then call
                                             → HTTP 429 → wait as asked, retry

This is a decorator over :class:`~portfolio_rag.ports.llm.LLMProvider`, and
that placement is the whole design:

* It paces **generations**, not questions. A question the corpus cannot answer
  short-circuits before any provider is reached, so it never waits for a slot
  it was not going to use.
* A retry re-issues **one provider call**, not one question. Retrieval is not
  repeated and no second record is produced, so a question that needed two
  attempts still counts exactly once for every metric.
* It is the **only** layer retrying. The adapter underneath is built with a
  single transport attempt for an evaluation run, because two bounded retry
  budgets stacked on each other do not add — they multiply, and three attempts
  over an adapter that tries three times is nine requests fired at the
  account that was already refusing one.
* It names no vendor. It reacts to
  :class:`~portfolio_rag.ports.errors.LLMProviderError` and the wait the
  provider asked for, both of which are port-level facts.

**Why this is not in the adapter.** The adapter retries transient transport
failures for a caller who is waiting for an answer: a handful of fast attempts,
one second apart. Sitting out a rate-limit window takes tens of seconds, which
is unacceptable for an HTTP request and entirely fine for a batch nobody is
watching. So the two are not the same retry with different numbers, and the
evaluation does not get both: it switches the adapter's budget off and keeps
this one. The adapter reports the wait it was asked for; this decides to take
it. Nothing here is wired into the server, and without it the evaluation
behaves exactly as it did before.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse, LLMProvider

#: How many times one generation may be attempted in total. Bounded on purpose:
#: a provider that is still refusing on the third try is rate limiting the
#: whole run, and the honest outcome is a failed evaluation rather than a
#: harness that sits in a loop pretending to make progress.
#:
#: This is also the *whole* budget for a question, not this layer's share of
#: one. The evaluation builds its generation adapter with a single transport
#: attempt (see :func:`portfolio_rag.composition.build_llm_provider`), so an
#: attempt here is exactly one request on the wire: three attempts are three
#: HTTP requests, never three times whatever the adapter would have tried.
DEFAULT_MAX_ATTEMPTS: Final = 3

#: First wait after a rate limit that did not say how long to wait for,
#: doubled on each further attempt.
DEFAULT_BACKOFF_SECONDS: Final = 5.0

#: Ceiling on any single wait, whether it came from a header or from the
#: backoff. An evaluation that has to pause for minutes is not worth finishing.
DEFAULT_MAX_WAIT_SECONDS: Final = 120.0

#: Injected so tests exercise the waiting without doing any.
Sleeper = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class GenerationPacing:
    """How hard an evaluation run is allowed to push a generation provider.

    ``min_interval_seconds`` is the gap between the *starts* of two generation
    calls. It is deliberately not derived from a provider's published limits:
    those are several numbers at once — requests per second and tokens per
    minute, of which the token budget is usually the binding one — and guessing
    a prompt's token count to schedule around it would be a scheduler. One
    measured number, chosen by whoever knows the account, is smaller and
    honest.
    """

    min_interval_seconds: float = 0.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS

    def __post_init__(self) -> None:
        if self.min_interval_seconds < 0:
            raise ValueError("min_interval_seconds cannot be negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        if self.max_wait_seconds < 0:
            raise ValueError("max_wait_seconds cannot be negative")

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for the run header."""
        return (
            ("minimum interval", f"{self.min_interval_seconds:.1f}s"),
            ("attempts per generation", str(self.max_attempts)),
            ("backoff", f"{self.backoff_seconds:.1f}s, doubling"),
            ("longest wait", f"{self.max_wait_seconds:.1f}s"),
        )


class PacedLLMProvider:
    """An :class:`LLMProvider` that spaces its calls out and waits out limits.

    Stateful across calls by necessity — the gap it enforces is between one
    call and the next — and single-threaded by assumption, which is what an
    evaluation run is.
    """

    def __init__(
        self,
        inner: LLMProvider,
        pacing: GenerationPacing,
        *,
        sleeper: Sleeper | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._inner = inner
        self._pacing = pacing
        self._sleep: Sleeper = sleeper or asyncio.sleep
        self._clock: Clock = clock or time.monotonic
        self._last_started: float | None = None

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def pacing(self) -> GenerationPacing:
        return self._pacing

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate once, waiting for a slot and retrying a refused call."""
        last_error: LLMProviderError | None = None
        requests_made = 0
        for attempt in range(1, self._pacing.max_attempts + 1):
            await self._wait(self._delay_before(attempt, last_error))
            self._last_started = self._clock()
            try:
                return await self._inner.generate(request)
            except LLMProviderError as exc:
                # A provider that will not answer differently next time is
                # reported now. So is one that has used up the budget: the run
                # fails, which is the result, rather than retrying forever.
                requests_made += exc.attempts
                if not exc.retryable or attempt == self._pacing.max_attempts:
                    # The error that ends the generation reports every request
                    # made for it, not only the adapter's share of the last one.
                    exc.attempts = requests_made
                    raise
                last_error = exc

        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    # --- waiting -------------------------------------------------------------

    def _delay_before(self, attempt: int, last_error: LLMProviderError | None) -> float:
        """How long to wait before *attempt*.

        A recovery wait never shortens the pacing gap: a provider answering
        "try again in one second" is describing its request limit, and the
        token budget that the interval exists for is still counting.
        """
        interval = self._remaining_interval()
        if last_error is None:
            return interval
        return max(interval, self._recovery_delay(attempt, last_error))

    def _remaining_interval(self) -> float:
        if self._pacing.min_interval_seconds <= 0 or self._last_started is None:
            return 0.0
        elapsed = self._clock() - self._last_started
        return max(0.0, self._pacing.min_interval_seconds - elapsed)

    def _recovery_delay(self, attempt: int, error: LLMProviderError) -> float:
        """Take the provider's own number when it gave one, otherwise back off.

        *attempt* is the one about to be made, so the first retry is attempt 2
        and gets one backoff interval.
        """
        requested = error.retry_after_seconds
        if requested is not None and requested > 0:
            return min(requested, self._pacing.max_wait_seconds)
        backoff = self._pacing.backoff_seconds * float(2 ** (attempt - 2))
        return min(backoff, self._pacing.max_wait_seconds)

    async def _wait(self, seconds: float) -> None:
        """Sleep, or do nothing at all when there is nothing to wait for.

        The distinction matters for more than tidiness: with pacing switched
        off this decorator makes no scheduling call whatsoever, which is what
        keeps "off" equal to the behaviour that existed before it.
        """
        if seconds > 0:
            await self._sleep(seconds)
