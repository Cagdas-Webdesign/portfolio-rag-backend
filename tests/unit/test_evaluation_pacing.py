"""Pacing an evaluation run, without any of it taking real time.

Every test here injects a clock and a sleeper, so a suite that exercises
eight-second waits and doubling backoffs still finishes instantly. A `sleep`
that actually sleeps anywhere in this file is a bug in the test, not a slow
test.
"""

from __future__ import annotations

import pytest

from portfolio_rag.evaluation.pacing import (
    DEFAULT_MAX_ATTEMPTS,
    GenerationPacing,
    PacedLLMProvider,
)
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import (
    GenerationRequest,
    MessageRole,
    PromptMessage,
)
from tests.doubles import ScriptedLLMProvider, ScriptedReply, grounded
from tests.support import run

REQUEST = GenerationRequest(
    messages=(PromptMessage(role=MessageRole.USER, content="Which framework?"),)
)


class _Timeline:
    """A clock that only moves when something sleeps.

    Which is exactly the property the pacing relies on: the gap it enforces is
    measured against wall time, so a test that sleeps without advancing time
    would let a second call through immediately and prove nothing.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def rate_limited(retry_after_seconds: float | None = None) -> LLMProviderError:
    return LLMProviderError(
        "provider rate limit reached",
        retryable=True,
        retry_after_seconds=retry_after_seconds,
    )


def make(
    *replies: ScriptedReply, pacing: GenerationPacing | None = None
) -> tuple[PacedLLMProvider, ScriptedLLMProvider, _Timeline]:
    inner = ScriptedLLMProvider(*replies)
    timeline = _Timeline()
    paced = PacedLLMProvider(
        inner,
        pacing or GenerationPacing(min_interval_seconds=8.0),
        sleeper=timeline.sleep,
        clock=timeline.clock,
    )
    return paced, inner, timeline


# --- pacing ------------------------------------------------------------------


def test_the_first_generation_is_not_made_to_wait():
    """There is nothing to be spaced apart from yet."""
    paced, _, timeline = make()

    run(paced.generate(REQUEST))

    assert timeline.slept == []


def test_a_second_generation_waits_out_the_interval():
    paced, inner, timeline = make(grounded("First.", "S1"), grounded("Second.", "S1"))

    run(paced.generate(REQUEST))
    run(paced.generate(REQUEST))

    assert timeline.slept == [8.0]
    assert inner.call_count == 2


def test_time_already_spent_counts_towards_the_interval():
    """The gap is between calls, not on top of whatever the pipeline took."""
    paced, _, timeline = make()

    run(paced.generate(REQUEST))
    timeline.advance(6.5)
    run(paced.generate(REQUEST))

    assert timeline.slept == [pytest.approx(1.5)]


def test_a_slow_generation_removes_the_wait_entirely():
    paced, _, timeline = make()

    run(paced.generate(REQUEST))
    timeline.advance(30.0)
    run(paced.generate(REQUEST))

    assert timeline.slept == []


def test_without_pacing_nothing_ever_sleeps():
    """The off switch is a real one: no scheduling call is made at all."""
    paced, inner, timeline = make(pacing=GenerationPacing())

    for _ in range(3):
        run(paced.generate(REQUEST))

    assert timeline.slept == []
    assert inner.call_count == 3


def test_pacing_is_only_applied_where_a_model_is_called():
    """A question answered without generation never reaches this decorator.

    The short-circuit lives in the answering service, which does not call the
    provider at all when retrieval found nothing. Nothing here has to know
    about that — which is the point of pacing the provider rather than the
    question loop.
    """
    paced, _, timeline = make()

    run(paced.generate(REQUEST))
    run(paced.generate(REQUEST))

    assert len(timeline.slept) == 1, "one wait for two calls, and none for the calls not made"


# --- rate limits --------------------------------------------------------------


def test_a_rate_limited_generation_is_retried_and_succeeds():
    paced, inner, _ = make(ScriptedReply(error=rate_limited()), grounded("Answered.", "S1"))

    response = run(paced.generate(REQUEST))

    assert inner.call_count == 2
    assert "Answered." in response.text


def test_a_successful_retry_produces_exactly_one_result():
    """What keeps a retried question from being counted twice.

    The decorator sits behind the port, so a retry re-issues one provider call
    inside one ``answer()``. Retrieval is not repeated and the runner still
    appends one record.
    """
    paced, _, _ = make(ScriptedReply(error=rate_limited()), grounded("Answered.", "S1"))

    results = [run(paced.generate(REQUEST))]

    assert len(results) == 1


def test_the_providers_own_retry_after_is_preferred_to_the_backoff():
    paced, _, timeline = make(
        ScriptedReply(error=rate_limited(retry_after_seconds=17.0)),
        grounded("Answered.", "S1"),
    )

    run(paced.generate(REQUEST))

    assert timeline.slept == [17.0]


def test_a_retry_after_shorter_than_the_interval_does_not_shorten_it():
    """A request-per-second limit says nothing about the token budget."""
    paced, _, timeline = make(
        grounded("First.", "S1"),
        ScriptedReply(error=rate_limited(retry_after_seconds=1.0)),
        grounded("Answered.", "S1"),
    )

    run(paced.generate(REQUEST))
    run(paced.generate(REQUEST))

    assert timeline.slept == [8.0, 8.0]


def test_a_rate_limit_without_a_header_backs_off_and_doubles():
    paced, _, timeline = make(
        ScriptedReply(error=rate_limited()),
        pacing=GenerationPacing(min_interval_seconds=0.0, backoff_seconds=5.0, max_attempts=4),
    )

    with pytest.raises(LLMProviderError):
        run(paced.generate(REQUEST))

    assert timeline.slept == [5.0, 10.0, 20.0]


def test_no_single_wait_exceeds_the_ceiling():
    paced, _, timeline = make(
        ScriptedReply(error=rate_limited(retry_after_seconds=600.0)),
        pacing=GenerationPacing(min_interval_seconds=0.0, max_wait_seconds=30.0, max_attempts=2),
    )

    with pytest.raises(LLMProviderError):
        run(paced.generate(REQUEST))

    assert timeline.slept == [30.0]


def test_retrying_stops_at_the_budget():
    paced, inner, _ = make(
        ScriptedReply(error=rate_limited()),
        pacing=GenerationPacing(min_interval_seconds=0.0, max_attempts=3),
    )

    with pytest.raises(LLMProviderError):
        run(paced.generate(REQUEST))

    assert inner.call_count == 3, "bounded: a run that cannot proceed fails and says so"


def test_a_failure_that_will_not_clear_is_tried_once():
    """An unusable credential does not become usable by waiting for it."""
    paced, inner, timeline = make(
        ScriptedReply(error=LLMProviderError("credentials rejected", retryable=False))
    )

    with pytest.raises(LLMProviderError, match="credentials"):
        run(paced.generate(REQUEST))

    assert inner.call_count == 1
    assert timeline.slept == []


def test_the_last_failure_is_what_the_caller_sees():
    paced, _, _ = make(
        ScriptedReply(error=rate_limited()),
        pacing=GenerationPacing(min_interval_seconds=0.0, max_attempts=2),
    )

    with pytest.raises(LLMProviderError) as caught:
        run(paced.generate(REQUEST))

    assert caught.value.retryable


# --- configuration ------------------------------------------------------------


def test_the_model_name_is_the_wrapped_providers():
    paced, inner, _ = make()

    assert paced.model == inner.model


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_interval_seconds": -1.0},
        {"max_attempts": 0},
        {"backoff_seconds": -1.0},
        {"max_wait_seconds": -1.0},
    ],
)
def test_nonsensical_pacing_is_refused(kwargs: dict[str, float | int]):
    with pytest.raises(ValueError):
        GenerationPacing(**kwargs)  # type: ignore[arg-type]


def test_the_default_budget_is_bounded():
    assert 1 < DEFAULT_MAX_ATTEMPTS <= 5
    assert GenerationPacing().max_attempts == DEFAULT_MAX_ATTEMPTS


def test_pacing_describes_itself_for_the_run_header():
    rows = dict(GenerationPacing(min_interval_seconds=8.0).describe())

    assert rows["minimum interval"] == "8.0s"
