"""The Cloudflare Workers AI chat adapter, without a network and without a key.

Requests are served by an in-process transport, so every branch — the happy
path, each failure mode, the retry policy, and the reasoning a thinking model
emits next to its answer — is exercised deterministically and for free. The
live smoke test is separate and opt-in.

No real credential appears anywhere in this file.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Sequence
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.llm.workers_ai import (
    DEFAULT_MODEL,
    WorkersAIChatProvider,
)
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import (
    GenerationRequest,
    LLMProvider,
    MessageRole,
    PromptMessage,
    ResponseFormat,
)
from tests.contracts.llm_provider import REQUEST, LLMProviderContract
from tests.support import run

FAKE_TOKEN = "test-token-not-a-real-credential"  # noqa: S105 - a fixture value
FAKE_ACCOUNT = "test-account-id"

#: What a reasoning model thinks on its way to an answer. If this string ever
#: reaches a caller, the adapter has leaked.
REASONING = "The user asked about X. Let me check source S1 and decide..."


def completion(
    text: Any = '{"answer": "Yes.", "sources": ["S1"]}', **message_extra: Any
) -> dict[str, Any]:
    """Cloudflare's envelope around a `messages`-shaped Workers AI result."""
    message: dict[str, Any] = {"role": "assistant", "content": text}
    message.update(message_extra)
    return {
        "success": True,
        "errors": [],
        "messages": [],
        "result": {"choices": [{"index": 0, "message": message}]},
    }


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
    return lambda _: httpx.Response(
        status_code, json={"errors": [{"message": "token-should-never-be-echoed"}]}
    )


def throttled(retry_after: str) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _: httpx.Response(429, json={"errors": []}, headers={"Retry-After": retry_after})


def raising(exception: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def _raise(_: httpx.Request) -> httpx.Response:
        raise exception

    return _raise


def make_provider(
    *responses: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[WorkersAIChatProvider, _Recorder]:
    recorder = _Recorder(responses or [json_response(completion())])
    client = httpx.AsyncClient(
        base_url="https://api.cloudflare.test/client/v4",
        transport=httpx.MockTransport(recorder),
        headers={"Authorization": f"Bearer {FAKE_TOKEN}"},
    )

    async def no_sleep(_: float) -> None:
        return None

    kwargs.setdefault("sleeper", no_sleep)
    provider = WorkersAIChatProvider(
        account_id=FAKE_ACCOUNT, api_token=FAKE_TOKEN, client=client, **kwargs
    )
    return provider, recorder


class TestWorkersAIChatProviderContract(LLMProviderContract):
    def make_provider(self) -> LLMProvider:
        return make_provider()[0]

    def make_failing_provider(self) -> LLMProvider:
        return make_provider(status_response(500), max_attempts=1)[0]


# --- the request --------------------------------------------------------------


def test_it_posts_to_the_accounts_ai_run_endpoint_for_the_configured_model():
    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    assert recorder.requests[0].method == "POST"
    expected = f"/client/v4/accounts/{FAKE_ACCOUNT}/ai/run/{DEFAULT_MODEL}"
    assert recorder.requests[0].url.path == expected


def test_a_configured_model_changes_the_endpoint_and_not_the_body():
    provider, recorder = make_provider(model="@cf/meta/llama-3.1-8b-instruct")

    run(provider.generate(REQUEST))

    assert "@cf/meta/llama-3.1-8b-instruct" in str(recorder.requests[0].url)
    assert "model" not in _body(recorder), "Workers AI takes the model in the URL, not the payload"


def test_the_messages_are_sent_in_order_with_their_roles():
    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    assert _body(recorder)["messages"] == [
        {"role": message.role.value, "content": message.content} for message in REQUEST.messages
    ]


def test_the_output_budget_and_temperature_are_passed_through():
    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    body = _body(recorder)
    assert body["max_tokens"] == REQUEST.max_output_tokens
    assert body["temperature"] == REQUEST.temperature


def test_a_json_answer_is_requested_when_the_port_asked_for_one():
    provider, recorder = make_provider()

    run(provider.generate(REQUEST))

    assert _body(recorder)["response_format"] == {"type": "json_object"}


def test_a_plain_text_request_asks_for_no_particular_format():
    provider, recorder = make_provider()

    run(
        provider.generate(
            GenerationRequest(
                messages=(PromptMessage(role=MessageRole.USER, content="Why?"),),
                response_format=ResponseFormat.TEXT,
            )
        )
    )

    body = _body(recorder)
    assert "response_format" not in body
    assert "max_tokens" not in body


def _body(recorder: _Recorder) -> dict[str, Any]:
    import json

    return dict(json.loads(recorder.requests[0].content))


# --- the answer ---------------------------------------------------------------


def test_the_final_assistant_text_is_what_comes_back():
    provider, _ = make_provider(json_response(completion("The grounded answer.")))

    assert run(provider.generate(REQUEST)).text == "The grounded answer."


def test_reasoning_next_to_the_answer_is_never_returned():
    """A thinking model's scratchpad is not an answer, and not the user's.

    The field is not filtered out — it is never read. That is what makes this
    hold for whatever a provider names the next one.
    """
    provider, _ = make_provider(
        json_response(completion("The grounded answer.", reasoning=REASONING))
    )

    response = run(provider.generate(REQUEST))

    assert response.text == "The grounded answer."
    assert REASONING not in response.text


def test_reasoning_under_any_other_field_name_is_also_never_returned():
    provider, _ = make_provider(
        json_response(
            completion(
                "The grounded answer.",
                reasoning_content=REASONING,
                thinking=REASONING,
                analysis=REASONING,
            )
        )
    )

    assert run(provider.generate(REQUEST)).text == "The grounded answer."


def test_reasoning_carried_as_a_content_part_is_dropped():
    """Typed content parts: the text-bearing ones are kept, the rest are not."""
    provider, _ = make_provider(
        json_response(
            completion(
                [
                    {"type": "reasoning", "text": REASONING},
                    {"type": "output_text", "text": "The grounded answer."},
                ]
            )
        )
    )

    response = run(provider.generate(REQUEST))

    assert response.text == "The grounded answer."
    assert REASONING not in response.text


def test_several_text_parts_are_joined_in_order():
    provider, _ = make_provider(
        json_response(
            completion([{"type": "text", "text": "One. "}, {"type": "text", "text": "Two."}])
        )
    )

    assert run(provider.generate(REQUEST)).text == "One. Two."


def test_provider_metadata_around_the_answer_is_not_carried_into_the_response():
    provider, _ = make_provider(
        json_response(
            {
                "success": True,
                "messages": [{"code": 1, "message": "internal-note"}],
                "result": {
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Text."}}],
                    "annotations": [{"note": "internal-annotation"}],
                },
            }
        )
    )

    response = run(provider.generate(REQUEST))

    assert response.text == "Text."
    assert "internal" not in response.model_dump_json()


def test_the_configured_model_is_reported_as_the_one_that_answered():
    """Workers AI does not echo the model; the adapter knows which one it asked."""
    provider, _ = make_provider(model="@cf/openai/gpt-oss-120b")

    assert run(provider.generate(REQUEST)).model == "@cf/openai/gpt-oss-120b"


def test_a_finish_reason_is_reported_when_the_provider_gives_one():
    provider, _ = make_provider(
        json_response(
            {
                "success": True,
                "result": {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Text."},
                            "finish_reason": "stop",
                        }
                    ]
                },
            }
        )
    )

    assert run(provider.generate(REQUEST)).finish_reason == "stop"


def test_token_usage_is_reported_when_it_is_present_and_sane():
    provider, _ = make_provider(
        json_response(
            {
                "success": True,
                "result": {
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "T."}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
                },
            }
        )
    )

    usage = run(provider.generate(REQUEST)).usage

    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (120, 30)


@pytest.mark.parametrize(
    "usage",
    ["not-an-object", {"prompt_tokens": "many"}, {"prompt_tokens": -1, "completion_tokens": 2}],
)
def test_unusable_usage_costs_a_diagnostic_and_never_an_answer(usage: Any):
    provider, _ = make_provider(
        json_response(
            {
                "success": True,
                "result": {
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "T."}}],
                    "usage": usage,
                },
            }
        )
    )

    response = run(provider.generate(REQUEST))

    assert response.text == "T."
    assert response.usage is None


def test_an_empty_completion_is_left_for_the_application_to_reject():
    """The transport succeeded. Whether "" is an answer is not this layer's call."""
    provider, _ = make_provider(json_response(completion("")))

    assert run(provider.generate(REQUEST)).text == ""


# --- malformed responses ------------------------------------------------------


def test_a_body_that_is_not_json_is_a_provider_error():
    provider, _ = make_provider(lambda _: httpx.Response(200, content=b"<html>gateway</html>"))

    with pytest.raises(LLMProviderError, match="not JSON"):
        run(provider.generate(REQUEST))


def test_a_json_body_that_is_not_an_object_is_a_provider_error():
    provider, _ = make_provider(json_response(["unexpected"]))

    with pytest.raises(LLMProviderError, match="not an object"):
        run(provider.generate(REQUEST))


def test_a_declared_failure_is_a_failure_even_with_http_200():
    provider, _ = make_provider(
        json_response({"success": False, "errors": [{"code": 7001}], "result": None})
    )

    with pytest.raises(LLMProviderError, match="failed generation"):
        run(provider.generate(REQUEST))


def test_a_body_without_a_result_object_is_a_provider_error():
    provider, _ = make_provider(json_response({"success": True, "result": None}))

    with pytest.raises(LLMProviderError, match="no result object"):
        run(provider.generate(REQUEST))


@pytest.mark.parametrize("choices", [None, [], "one", {}])
def test_a_result_without_choices_is_a_provider_error(choices: Any):
    provider, _ = make_provider(json_response({"success": True, "result": {"choices": choices}}))

    with pytest.raises(LLMProviderError, match="no choices"):
        run(provider.generate(REQUEST))


def test_a_choice_that_is_not_an_object_is_a_provider_error():
    provider, _ = make_provider(json_response({"success": True, "result": {"choices": ["text"]}}))

    with pytest.raises(LLMProviderError, match="not an object"):
        run(provider.generate(REQUEST))


def test_a_choice_without_a_message_is_a_provider_error():
    provider, _ = make_provider(
        json_response({"success": True, "result": {"choices": [{"index": 0}]}})
    )

    with pytest.raises(LLMProviderError, match="no message"):
        run(provider.generate(REQUEST))


@pytest.mark.parametrize(
    "content",
    [
        None,
        123,
        [],
        [{"type": "reasoning", "text": REASONING}],
        [{"type": "output_text"}],
    ],
)
def test_a_message_without_usable_text_is_a_provider_error(content: Any):
    """Including the case that matters most: reasoning, and nothing else.

    An answer that is only a scratchpad must fail loudly, never be published.
    """
    provider, _ = make_provider(json_response(completion(content)))

    with pytest.raises(LLMProviderError, match="no text content"):
        run(provider.generate(REQUEST))


def test_a_reasoning_only_message_never_reaches_the_caller_as_text():
    provider, _ = make_provider(
        json_response(completion([{"type": "reasoning", "text": REASONING}]))
    )

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert REASONING not in caught.value.message


# --- failures -----------------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_a_rejected_credential_is_reported_without_being_retried(status_code: int):
    provider, recorder = make_provider(status_response(status_code))

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert len(recorder.requests) == 1, "asking again will not produce a different answer"
    assert not caught.value.retryable


def test_a_rate_limit_is_retried_and_then_given_up_on():
    provider, recorder = make_provider(status_response(429), max_attempts=3)

    with pytest.raises(LLMProviderError, match="rate limit"):
        run(provider.generate(REQUEST))

    assert len(recorder.requests) == 3


def test_a_rate_limit_reports_how_long_the_provider_asked_for():
    """Reported, not obeyed — the eval pacer is what can afford to wait."""
    provider, _ = make_provider(throttled("20"), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds == 20.0
    assert caught.value.retryable


def test_the_adapter_does_not_lengthen_its_own_waits_to_match_the_header():
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    provider, _ = make_provider(throttled("45"), max_attempts=2, sleeper=record)
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
    "header", ["", "soon", "-5", "99999", "nan", "NaN", "inf", "Infinity", "-inf", "-Infinity"]
)
def test_an_unusable_retry_after_is_reported_as_no_advice_at_all(header: str):
    provider, _ = make_provider(throttled(header), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retry_after_seconds is None


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_server_errors_are_retried_within_the_budget(status_code: int):
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


def test_a_transient_failure_that_clears_produces_an_answer():
    provider, recorder = make_provider(
        status_response(503), json_response(completion()), max_attempts=3
    )

    response = run(provider.generate(REQUEST))

    assert len(recorder.requests) == 2
    assert response.text


def test_a_timeout_is_a_retryable_provider_error():
    provider, recorder = make_provider(raising(httpx.ConnectTimeout("timed out")), max_attempts=2)

    with pytest.raises(LLMProviderError, match="timed out") as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retryable
    assert len(recorder.requests) == 2


def test_a_network_error_is_a_retryable_provider_error():
    provider, _ = make_provider(raising(httpx.ConnectError("no route")), max_attempts=1)

    with pytest.raises(LLMProviderError, match="could not be reached") as caught:
        run(provider.generate(REQUEST))

    assert caught.value.retryable


def test_no_upstream_body_is_ever_repeated_in_the_error():
    provider, _ = make_provider(status_response(500), max_attempts=1)

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert "token-should-never-be-echoed" not in caught.value.message
    assert "500" in caught.value.message


def test_the_credential_never_appears_in_an_error():
    provider, _ = make_provider(status_response(401))

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert FAKE_TOKEN not in caught.value.message


# --- construction --------------------------------------------------------------


def test_a_provider_without_an_account_is_refused():
    with pytest.raises(ValueError, match="account id"):
        WorkersAIChatProvider(account_id="", api_token=FAKE_TOKEN)


def test_a_provider_without_a_token_is_refused():
    with pytest.raises(ValueError, match="token"):
        WorkersAIChatProvider(account_id=FAKE_ACCOUNT, api_token="")


def test_a_provider_without_a_model_is_refused():
    with pytest.raises(ValueError, match="model"):
        WorkersAIChatProvider(account_id=FAKE_ACCOUNT, api_token=FAKE_TOKEN, model="")


def test_a_nonsensical_retry_budget_is_refused():
    with pytest.raises(ValueError, match="max_attempts"):
        WorkersAIChatProvider(account_id=FAKE_ACCOUNT, api_token=FAKE_TOKEN, max_attempts=0)


def test_the_default_model_is_the_one_this_project_migrated_to():
    assert DEFAULT_MODEL == "@cf/openai/gpt-oss-120b"


# --- what "unreachable" may and may not mean ------------------------------------
#
# A request Workers AI never received must not be reported as Workers AI being
# unreachable. Seen in practice: a token with a trailing newline made the HTTP
# client refuse the Authorization header locally, every attempt failed before a
# byte was sent, and the run reported `retryable_provider_error / unreachable`
# after three attempts — while the endpoint answered a hand-made request fine.


@pytest.mark.parametrize(
    ("status", "kind", "retryable", "requests"),
    [
        (400, ProviderFailureKind.HTTP_STATUS, False, 1),
        (401, ProviderFailureKind.HTTP_STATUS, False, 1),
        (403, ProviderFailureKind.HTTP_STATUS, False, 1),
        (429, ProviderFailureKind.RATE_LIMITED, True, 3),
    ],
)
def test_an_answering_endpoint_is_never_unreachable(
    status: int, kind: ProviderFailureKind, retryable: bool, requests: int
):
    provider, recorder = make_provider(status_response(status))

    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    error = caught.value
    assert error.kind is kind and error.kind is not ProviderFailureKind.UNREACHABLE
    assert (error.status_code, error.retryable) == (status, retryable)
    assert len(recorder.requests) == error.attempts == requests


def test_a_200_completion_is_a_success():
    provider, _ = make_provider(json_response(completion("Hello.")))

    assert run(provider.generate(REQUEST)).text == "Hello."


def test_a_real_connection_failure_is_unreachable_and_names_its_class():
    provider, recorder = make_provider(raising(httpx.ConnectError("no route")))

    with pytest.raises(
        LLMProviderError, match=r"could not be reached .*\(ConnectError\)"
    ) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.kind is ProviderFailureKind.UNREACHABLE
    assert caught.value.retryable and len(recorder.requests) == 3


@pytest.mark.parametrize(
    "exception",
    [
        httpx.LocalProtocolError("Illegal header value b'Bearer secret-token\\n'"),
        httpx.UnsupportedProtocol("Request URL has an unsupported protocol 'ftp://'."),
    ],
)
def test_a_request_that_cannot_be_sent_is_not_unreachable_and_not_retried(
    exception: Exception,
):
    provider, recorder = make_provider(raising(exception))

    with pytest.raises(LLMProviderError, match="invalid locally") as caught:
        run(provider.generate(REQUEST))

    error = caught.value
    assert error.kind is ProviderFailureKind.UNSPECIFIED
    assert not error.retryable
    assert error.attempts == 1 and len(recorder.requests) == 1
    assert type(exception).__name__ in error.message
    assert "secret-token" not in error.message  # the header value is never quoted


def test_a_token_with_a_trailing_newline_fails_locally_without_a_network_call():
    """The real client and transport, no mock: h11 refuses the header itself.

    A loopback socket that accepts connections and never answers: the client
    connects, then refuses to send. Nothing leaves the machine.
    """
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    provider = WorkersAIChatProvider(
        account_id="test-account-id",
        api_token="test-token-not-a-real-credential\n",  # noqa: S106 - a fixture value
        base_url=f"http://127.0.0.1:{port}/client/v4",
        max_attempts=3,
        timeout_seconds=2.0,
    )

    async def generate() -> None:
        try:
            await provider.generate(REQUEST)
        finally:
            await provider.aclose()

    try:
        with pytest.raises(LLMProviderError) as caught:
            run(generate())
    finally:
        listener.close()

    assert caught.value.kind is ProviderFailureKind.UNSPECIFIED
    assert caught.value.attempts == 1
    assert "LocalProtocolError" in caught.value.message
    assert "test-token" not in caught.value.message
