"""How many real HTTP requests one evaluation question is allowed to cost.

Two bounded retry budgets in series do not add, they multiply: a pacer that
tries three times over an adapter that tries three times is nine requests, and
every one of them lands on the account that was already refusing the first.
This file counts the requests that actually reach the transport, so that the
number is measured rather than reasoned about.

The rule it pins down: **in a paced evaluation run the pacer is the only layer
that retries.** The adapter below it is built with a single transport attempt,
so one attempt is one request, and the pacer's budget is the whole budget.

Both real chat adapters are exercised, because the problem is structural rather
than a property of one vendor. No credential and no network: every response
comes from an in-process transport, and the account ids and tokens are
fixtures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.cli import EVALUATION_GENERATION_ATTEMPTS
from portfolio_rag.evaluation.pacing import GenerationPacing, PacedLLMProvider
from portfolio_rag.infrastructure.llm import MistralChatProvider, WorkersAIChatProvider
from portfolio_rag.infrastructure.llm import mistral as mistral_module
from portfolio_rag.infrastructure.llm import workers_ai as workers_ai_module
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import LLMProvider
from tests.contracts.llm_provider import REQUEST
from tests.support import run

FAKE_KEY = "test-key-not-a-real-credential"
FAKE_ACCOUNT = "test-account-id"

#: The budget under test, taken from the pacer's own default rather than
#: repeated here — if the default moves, these counts move with it.
EVAL_ATTEMPTS = GenerationPacing().max_attempts

ANSWER = '{"answer": "Yes.", "sources": ["S1"]}'


class _Wire:
    """An in-process transport that counts what actually reaches it."""

    def __init__(self, responses: list[Callable[[], httpx.Response]]) -> None:
        self._responses = responses
        self.requests = 0

    def __call__(self, _: httpx.Request) -> httpx.Response:
        self.requests += 1
        index = min(self.requests - 1, len(self._responses) - 1)
        return self._responses[index]()


def ok(vendor: str) -> Callable[[], httpx.Response]:
    """A successful completion in that vendor's response shape."""
    if vendor == "workers_ai":
        payload: dict[str, Any] = {
            "success": True,
            "result": {"choices": [{"index": 0, "message": {"content": ANSWER}}]},
        }
    else:
        payload = {"choices": [{"index": 0, "message": {"content": ANSWER}}]}
    return lambda: httpx.Response(200, json=payload)


def failing(status_code: int, retry_after: str | None = None) -> Callable[[], httpx.Response]:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return lambda: httpx.Response(status_code, json={"errors": []}, headers=headers)


class _Timeline:
    """Time that only moves when something waits for it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _adapter(vendor: str, wire: _Wire, *, attempts: int | None) -> LLMProvider:
    """The real adapter, wired to the counting transport."""
    client = httpx.AsyncClient(
        base_url="https://provider.test", transport=httpx.MockTransport(wire)
    )

    async def no_sleep(_: float) -> None:
        return None

    if vendor == "workers_ai":
        return WorkersAIChatProvider(
            account_id=FAKE_ACCOUNT,
            api_token=FAKE_KEY,
            client=client,
            max_attempts=attempts,
            sleeper=no_sleep,
        )
    return MistralChatProvider(
        api_key=FAKE_KEY, client=client, max_attempts=attempts, sleeper=no_sleep
    )


def eval_stack(
    vendor: str, *responses: Callable[[], httpx.Response]
) -> tuple[PacedLLMProvider, _Wire, _Timeline]:
    """Exactly how a paced evaluation run is wired, counting transport.

    The adapter's budget is the constant the CLI passes, not a literal: if that
    ever stops being 1, these counts fail rather than quietly tripling.
    """
    wire = _Wire(list(responses))
    timeline = _Timeline()
    paced = PacedLLMProvider(
        _adapter(vendor, wire, attempts=EVALUATION_GENERATION_ATTEMPTS),
        GenerationPacing(min_interval_seconds=8.0),
        sleeper=timeline.sleep,
        clock=timeline.clock,
    )
    return paced, wire, timeline


@pytest.fixture(params=["workers_ai", "mistral"])
def vendor(request: pytest.FixtureRequest) -> Iterator[str]:
    yield request.param


# --- the budget, counted on the wire ------------------------------------------


def test_a_question_that_succeeds_costs_one_request(vendor: str):
    paced, wire, timeline = eval_stack(vendor, ok(vendor))

    response = run(paced.generate(REQUEST))

    assert wire.requests == 1
    assert response.text == ANSWER
    assert timeline.slept == [], "nothing to wait for on a first, successful call"


def test_one_rate_limit_then_success_costs_two_requests(vendor: str):
    paced, wire, _ = eval_stack(vendor, failing(429), ok(vendor))

    response = run(paced.generate(REQUEST))

    assert wire.requests == 2
    assert response.text == ANSWER


def test_two_rate_limits_then_success_costs_three_requests(vendor: str):
    paced, wire, _ = eval_stack(vendor, failing(429), failing(429), ok(vendor))

    response = run(paced.generate(REQUEST))

    assert wire.requests == 3
    assert response.text == ANSWER


def test_a_permanent_rate_limit_costs_exactly_three_requests(vendor: str):
    """The number this change exists for. It was twelve."""
    paced, wire, _ = eval_stack(vendor, failing(429))

    with pytest.raises(LLMProviderError) as caught:
        run(paced.generate(REQUEST))

    assert wire.requests == EVAL_ATTEMPTS == 3
    assert caught.value.retryable
    assert "rate limit" in caught.value.message


@pytest.mark.parametrize("status_code", [401, 403])
def test_a_rejected_credential_costs_one_request(vendor: str, status_code: int):
    paced, wire, timeline = eval_stack(vendor, failing(status_code))

    with pytest.raises(LLMProviderError) as caught:
        run(paced.generate(REQUEST))

    assert wire.requests == 1, "neither layer retries what cannot succeed"
    assert not caught.value.retryable
    assert timeline.slept == []


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_a_permanent_client_error_costs_one_request(vendor: str, status_code: int):
    paced, wire, _ = eval_stack(vendor, failing(status_code))

    with pytest.raises(LLMProviderError):
        run(paced.generate(REQUEST))

    assert wire.requests == 1


def test_the_budget_is_a_ceiling_and_not_a_target(vendor: str):
    """Three attempts available, one used, two not spent."""
    paced, wire, _ = eval_stack(vendor, ok(vendor))

    run(paced.generate(REQUEST))

    assert wire.requests < EVAL_ATTEMPTS


# --- Retry-After --------------------------------------------------------------


def test_the_pacer_waits_the_retry_after_between_the_requests(vendor: str):
    """Nine seconds of waiting, and no attempt fired inside that window."""
    paced, wire, timeline = eval_stack(vendor, failing(429, retry_after="9"), ok(vendor))

    run(paced.generate(REQUEST))

    assert wire.requests == 2
    assert timeline.slept == [9.0]


def test_no_one_second_adapter_waits_happen_underneath(vendor: str):
    """The symptom the old cascade left behind.

    Eight one-second sleeps between twelve requests meant eight attempts fired
    too early to have been anything but refused. With a single transport
    attempt there is nothing underneath to sleep, so every wait recorded here
    belongs to the pacer and is the wait the provider actually asked for.
    """
    paced, wire, timeline = eval_stack(vendor, failing(429, retry_after="9"))

    with pytest.raises(LLMProviderError):
        run(paced.generate(REQUEST))

    assert wire.requests == 3
    assert timeline.slept == [9.0, 9.0]
    assert 1.0 not in timeline.slept


def test_a_retry_after_shorter_than_the_pacing_interval_does_not_shorten_it(vendor: str):
    paced, _, timeline = eval_stack(vendor, failing(429, retry_after="2"), ok(vendor))

    run(paced.generate(REQUEST))

    assert timeline.slept == [8.0], "the token budget the interval exists for is still counting"


def test_without_a_retry_after_the_pacer_backs_off_on_its_own(vendor: str):
    paced, wire, timeline = eval_stack(vendor, failing(429))

    with pytest.raises(LLMProviderError):
        run(paced.generate(REQUEST))

    assert wire.requests == 3
    assert timeline.slept == [8.0, 10.0], "the pacing interval, then one doubling of the backoff"


# --- the adapter on its own ----------------------------------------------------


def test_a_single_attempt_adapter_does_not_retry_by_itself(vendor: str):
    """Directly, with no pacer above it: one call, one request."""
    wire = _Wire([failing(429)])
    adapter = _adapter(vendor, wire, attempts=EVALUATION_GENERATION_ATTEMPTS)

    with pytest.raises(LLMProviderError):
        run(adapter.generate(REQUEST))

    assert wire.requests == 1


def test_the_evaluation_asks_for_exactly_one_transport_attempt():
    assert EVALUATION_GENERATION_ATTEMPTS == 1


# --- production is untouched ----------------------------------------------------


def test_an_unconfigured_adapter_keeps_its_own_retry_budget(vendor: str):
    """What a normal backend request still gets: three transport attempts."""
    wire = _Wire([failing(503)])
    adapter = _adapter(vendor, wire, attempts=None)

    with pytest.raises(LLMProviderError):
        run(adapter.generate(REQUEST))

    assert wire.requests == 3


@pytest.mark.parametrize("module", [workers_ai_module, mistral_module])
def test_the_adapter_defaults_are_unchanged(module: Any):
    """Production's numbers, pinned. Nothing in this change moved them."""
    assert module.DEFAULT_MAX_ATTEMPTS == 3
    assert module.DEFAULT_RETRY_DELAY_SECONDS == 1.0
