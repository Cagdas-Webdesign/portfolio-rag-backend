"""The Mistral chat adapter, without a network and without a key.

Requests are served by an in-process transport, so every branch — the happy
path, each failure mode, the retry policy — is exercised deterministically and
for free. The live smoke test is separate and opt-in.

No real credential appears anywhere in this file.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.llm.mistral import (
    DEFAULT_MODEL,
    MistralChatProvider,
)
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import (
    GenerationRequest,
    LLMProvider,
    MessageRole,
    PromptMessage,
    ResponseFormat,
)
from tests.contracts.llm_provider import REQUEST, LLMProviderContract
from tests.support import run

FAKE_KEY = "test-key-not-a-real-credential"


def completion(text: str = '{"answer": "Yes.", "sources": ["S1"]}', **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }
    payload.update(extra)
    return payload


class _Recorder:
    """Captures requests and replays scripted responses."""

    def __init__(self, responses: Sequence[Callable[[httpx.Request], httpx.Response]]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.responses[min(len(self.requests) - 1, len(self.responses) - 1)]
        return handler(request)


def json_response(
    payload: Any, status_code: int = 200
) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(status_code, json=payload)


def status_response(status_code: int) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(status_code, json={"message": "sk-live-should-never-be-echoed"})


def throttled(retry_after: str) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(
        429, json={"message": "slow down"}, headers={"Retry-After": retry_after}
    )


def raising(exception: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def _raise(_: httpx.Request) -> httpx.Response:
        raise exception

    return _raise


def make_provider(
    *responses: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[MistralChatProvider, _Recorder]:
    recorder = _Recorder(responses or [json_response(completion())])
    client = httpx.AsyncClient(
        base_url="https://api.mistral.test",
        transport=httpx.MockTransport(recorder),
        headers={"Authorization": f"Bearer {FAKE_KEY}"},
    )

    async def no_sleep(_: float) -> None:
        return None

    kwargs.setdefault("sleeper", no_sleep)
    return MistralChatProvider(api_key=FAKE_KEY, client=client, **kwargs), recorder


class TestMistralChatProviderContract(LLMProviderContract):
    def make_provider(self) -> LLMProvider:
        return make_provider()[0]

    def make_failing_provider(self) -> LLMProvider:
        return make_provider(status_response(500), max_attempts=1)[0]


# --- the request ------------------------------------------------------------


def test_it_posts_to_the_chat_completions_endpoint():
    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    assert recorder.requests[0].url.path == "/v1/chat/completions"
    assert recorder.requests[0].method == "POST"


def test_the_credential_travels_as_a_bearer_token():
    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    assert recorder.requests[0].headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_the_configured_model_is_the_one_requested():
    import json

    provider, recorder = make_provider(model="mistral-medium-latest")

    run(provider.generate(REQUEST))

    assert json.loads(recorder.requests[0].content)["model"] == "mistral-medium-latest"
    assert provider.model == "mistral-medium-latest"


def test_roles_and_content_are_mapped_one_to_one():
    import json

    provider, recorder = make_provider()

    run(provider.generate(REQUEST))
    sent = json.loads(recorder.requests[0].content)["messages"]

    assert [message["role"] for message in sent] == ["system", "user"]
    assert sent[0]["content"] == REQUEST.messages[0].content
    assert sent[1]["content"] == REQUEST.messages[1].content


def test_the_context_reaches_the_provider_inside_the_user_message():
    import json

    provider, recorder = make_provider()

    run(provider.generate(REQUEST))
    sent = json.loads(recorder.requests[0].content)["messages"]

    assert "[SOURCE S1]" in sent[1]["content"]
    assert "[SOURCE S1]" not in sent[0]["content"]


def test_structured_output_is_asked_for_when_the_request_says_so():
    import json

    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    assert json.loads(recorder.requests[0].content)["response_format"] == {"type": "json_object"}


def test_plain_text_requests_do_not_ask_for_json():
    import json

    provider, recorder = make_provider()

    run(provider.generate(REQUEST.model_copy(update={"response_format": ResponseFormat.TEXT})))

    assert "response_format" not in json.loads(recorder.requests[0].content)


def test_generation_limits_are_passed_through():
    import json

    provider, recorder = make_provider()

    run(provider.generate(REQUEST))
    payload = json.loads(recorder.requests[0].content)

    assert payload["max_tokens"] == 256
    assert payload["temperature"] == pytest.approx(0.2)


def test_optional_limits_are_omitted_rather_than_sent_as_null():
    import json

    provider, recorder = make_provider()
    minimal = GenerationRequest(messages=(PromptMessage(role=MessageRole.USER, content="hi"),))

    run(provider.generate(minimal))
    payload = json.loads(recorder.requests[0].content)

    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_a_timeout_is_configured_on_the_client():
    provider = MistralChatProvider(api_key=FAKE_KEY, timeout_seconds=7.5)

    assert provider._client.timeout.read == pytest.approx(7.5)
    run(provider.aclose())


# --- the response ------------------------------------------------------------


def test_the_message_content_is_returned_verbatim():
    provider, _ = make_provider(json_response(completion('{"answer": "Yes.", "sources": ["S1"]}')))

    response = run(provider.generate(REQUEST))

    assert response.text == '{"answer": "Yes.", "sources": ["S1"]}'


def test_an_unknown_source_label_survives_to_the_application():
    """The adapter does not judge the answer; validation happens further in."""
    provider, _ = make_provider(json_response(completion('{"answer": "Yes.", "sources": ["S9"]}')))

    assert "S9" in run(provider.generate(REQUEST)).text


def test_the_reported_model_and_finish_reason_are_carried_through():
    provider, _ = make_provider(
        json_response(
            {
                "model": "mistral-small-2506",
                "choices": [
                    {"message": {"content": "{}"}, "finish_reason": "stop"},
                ],
            }
        )
    )

    response = run(provider.generate(REQUEST))

    assert response.model == "mistral-small-2506"
    assert response.finish_reason == "stop"


def test_token_usage_is_reported_when_the_provider_sends_it():
    provider, _ = make_provider(
        json_response(completion(usage={"prompt_tokens": 120, "completion_tokens": 30}))
    )

    usage = run(provider.generate(REQUEST)).usage

    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (120, 30)


@pytest.mark.parametrize(
    "usage",
    ["not-a-dict", {"prompt_tokens": "many"}, {"prompt_tokens": -1, "completion_tokens": 2}, {}],
)
def test_unusable_usage_costs_a_diagnostic_and_never_an_answer(usage: Any):
    provider, _ = make_provider(json_response(completion(usage=usage)))

    response = run(provider.generate(REQUEST))

    assert response.usage is None
    assert response.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": "not a list"},
        {"choices": ["not an object"]},
        {"choices": [{}]},
        {"choices": [{"message": "not an object"}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 42}}]},
    ],
)
def test_a_response_without_usable_text_is_a_provider_error(payload: dict[str, Any]):
    provider, _ = make_provider(json_response(payload))

    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))


def test_a_body_that_is_not_json_is_a_provider_error():
    provider, _ = make_provider(lambda _: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))


def test_json_that_is_not_an_object_is_a_provider_error():
    provider, _ = make_provider(json_response(["nope"]))

    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))


def test_an_empty_completion_is_left_for_the_application_to_reject():
    """The transport succeeded. Whether "" is an answer is not this layer's call."""
    provider, _ = make_provider(json_response(completion("")))

    assert run(provider.generate(REQUEST)).text == ""


# --- failures ----------------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_a_rejected_credential_is_reported_without_being_retried(status_code: int):
    provider, recorder = make_provider(status_response(status_code))

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert len(recorder.requests) == 1, "asking again will not produce a different answer"
    assert not caught.value.retryable


def test_a_rate_limit_is_retried_and_then_given_up_on():
    provider, recorder = make_provider(status_response(429), max_attempts=3)

    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))

    assert len(recorder.requests) == 3


def test_a_rate_limit_reports_how_long_the_provider_asked_for():
    """Reported, not obeyed: this adapter's own delay is unchanged by it.

    A caller with a browser waiting cannot sit out a rate-limit window; a batch
    run can, and is the one that reads this.
    """
    provider, _ = make_provider(throttled("12"), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds == 12.0


def test_the_adapter_does_not_lengthen_its_own_waits_to_match_the_header():
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    provider, _ = make_provider(throttled("30"), max_attempts=2, sleeper=record)
    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))

    assert slept == [1.0]


@pytest.mark.parametrize("header", ["nan", "NaN", "-nan"])
def test_a_not_a_number_retry_after_is_rejected(header: str):
    """NaN does not fail a range check — it fails every comparison in one.

    `nan < 0` and `nan > 300` are both false, so a bounds check alone lets it
    through and hands a caller a wait it can neither compare nor sleep for.
    """
    provider, _ = make_provider(throttled(header), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds is None


@pytest.mark.parametrize("header", ["inf", "Infinity", "-inf", "-Infinity"])
def test_an_infinite_retry_after_is_rejected(header: str):
    provider, _ = make_provider(throttled(header), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds is None


@pytest.mark.parametrize(
    ("header", "expected"), [("0", 0.0), ("1", 1.0), ("12", 12.0), ("30.5", 30.5), ("300", 300.0)]
)
def test_a_finite_non_negative_retry_after_is_still_reported_unchanged(
    header: str, expected: float
):
    """The values that were already accepted must behave exactly as before."""
    provider, _ = make_provider(throttled(header), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds == expected


@pytest.mark.parametrize(
    "header",
    [
        "",
        "soon",
        "Wed, 21 Oct 2015 07:28:00 GMT",
        "-5",
        "99999",
        "nan",
        "NaN",
        "inf",
        "Infinity",
        "-inf",
        "-Infinity",
    ],
)
def test_an_unusable_retry_after_is_reported_as_no_advice_at_all(header: str):
    """A header is upstream input. Unreadable or absurd means the caller decides."""
    provider, _ = make_provider(throttled(header), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds is None


def test_a_failure_without_the_header_advises_nothing():
    provider, _ = make_provider(status_response(503), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds is None


def test_a_transient_failure_that_clears_produces_an_answer():
    provider, recorder = make_provider(
        status_response(503), json_response(completion()), max_attempts=3
    )

    # The recorder replays by position, so the second attempt gets the success.
    response = run(provider.generate(REQUEST))

    assert len(recorder.requests) == 2
    assert response.text


@pytest.mark.parametrize("status_code", [408, 500, 502, 503, 504])
def test_transient_statuses_are_retried(status_code: int):
    provider, recorder = make_provider(status_response(status_code), max_attempts=2)

    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))

    assert len(recorder.requests) == 2


@pytest.mark.parametrize("status_code", [400, 404, 422])
def test_permanent_statuses_are_not_retried(status_code: int):
    provider, recorder = make_provider(status_response(status_code), max_attempts=3)

    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))

    assert len(recorder.requests) == 1


def test_a_timeout_is_a_retryable_provider_error():
    provider, recorder = make_provider(raising(httpx.ConnectTimeout("timed out")), max_attempts=2)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retryable
    assert len(recorder.requests) == 2


def test_a_network_error_is_a_retryable_provider_error():
    provider, _ = make_provider(raising(httpx.ConnectError("no route")), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retryable


def test_no_upstream_body_is_ever_repeated_in_the_error():
    provider, _ = make_provider(status_response(500), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert "sk-live" not in caught.value.message
    assert "500" in caught.value.message


def test_the_credential_never_appears_in_an_error():
    provider, _ = make_provider(status_response(401))

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert FAKE_KEY not in caught.value.message


def test_retries_do_not_sleep_in_tests():
    """The injected sleeper is what keeps the retry tests instant."""
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    provider, _ = make_provider(status_response(429), max_attempts=3, sleeper=record)
    with pytest.raises(LLMProviderError):
        run(provider.generate(REQUEST))

    assert slept == [1.0, 1.0]


# --- construction ------------------------------------------------------------


def test_a_provider_without_a_key_is_refused():
    with pytest.raises(ValueError, match="API key"):
        MistralChatProvider(api_key="")


def test_a_provider_without_a_model_is_refused():
    with pytest.raises(ValueError, match="model"):
        MistralChatProvider(api_key=FAKE_KEY, model="")


def test_a_nonsensical_retry_budget_is_refused():
    with pytest.raises(ValueError, match="max_attempts"):
        MistralChatProvider(api_key=FAKE_KEY, max_attempts=0)


def test_the_default_model_is_a_chat_model_not_an_embedding_model():
    assert "embed" not in DEFAULT_MODEL
    assert DEFAULT_MODEL == "mistral-small-latest"
