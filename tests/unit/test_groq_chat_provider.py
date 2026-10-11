"""The Groq adapter, offline: request shape, reading the reply, every failure.

Groq is answered by an `httpx.MockTransport`; nothing here reaches a network.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.llm import GroqChatProvider
from portfolio_rag.infrastructure.llm.groq import DEFAULT_MODEL
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import (
    GenerationRequest,
    LLMProvider,
    MessageRole,
    PromptMessage,
    ResponseFormat,
)
from portfolio_rag.rag.errors import GenerationFailureCategory, provider_failure
from tests.contracts.llm_provider import LLMProviderContract
from tests.support import run

FAKE_KEY = "gsk_test_not_a_real_key_0123456789"
REASONING = "PRIVATE CHAIN OF THOUGHT"
REQUEST = GenerationRequest(
    messages=(
        PromptMessage(role=MessageRole.SYSTEM, content="You check one thing."),
        PromptMessage(role=MessageRole.USER, content="EVIDENCE\n…\n\nANSWER\n…"),
    ),
    max_output_tokens=800,
    temperature=0.2,
    response_format=ResponseFormat.JSON_OBJECT,
)


def completion(content: Any = '{"verdict": "supported"}', **choice: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "reasoning": REASONING},
                "finish_reason": "stop",
                **choice,
            }
        ],
        "usage": {"prompt_tokens": 640, "completion_tokens": 180},
    }


Handler = Callable[[httpx.Request], httpx.Response]


def make(*responses: Handler, **kwargs: Any) -> tuple[GroqChatProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    script = list(responses) or [lambda _: httpx.Response(200, json=completion())]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return script[min(len(seen) - 1, len(script) - 1)](request)

    client = httpx.AsyncClient(
        base_url="https://api.groq.test/openai/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": f"Bearer {FAKE_KEY}"},
    )

    async def no_sleep(_: float) -> None:
        return None

    kwargs.setdefault("sleeper", no_sleep)
    return GroqChatProvider(api_key=FAKE_KEY, client=client, **kwargs), seen


def status(code: int, **headers: str) -> Handler:
    body = {"error": {"message": "UPSTREAM PROSE", "type": "invalid_request_error"}}
    return lambda _: httpx.Response(code, json=body, headers=headers)


def failure_of(provider: GroqChatProvider) -> LLMProviderError:
    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))
    return caught.value


class TestGroqHonoursTheContract(LLMProviderContract):
    def make_provider(self) -> LLMProvider:
        return make()[0]

    def make_failing_provider(self) -> LLMProvider:
        return make(status(500), max_attempts=1)[0]


# --- the request ------------------------------------------------------------------


def test_the_request_is_the_grounding_request_in_openai_shape():
    provider, seen = make()
    run(provider.generate(REQUEST))

    (sent,) = seen
    body = json.loads(sent.content)
    assert sent.url.path.endswith("/chat/completions")
    assert body == {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": "You check one thing."},
            {"role": "user", "content": "EVIDENCE\n…\n\nANSWER\n…"},
        ],
        "include_reasoning": False,
        "max_completion_tokens": 800,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }


def test_the_default_model_is_the_one_the_comparison_measured():
    assert DEFAULT_MODEL == "openai/gpt-oss-20b"
    assert make()[0].model == "openai/gpt-oss-20b"


def test_a_key_is_required():
    with pytest.raises(ValueError, match="key"):
        GroqChatProvider(api_key="")


def test_the_key_appears_only_in_the_authorization_header():
    provider = GroqChatProvider(api_key=FAKE_KEY)
    assert FAKE_KEY not in repr(provider)
    run(provider.aclose())


# --- the reply ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "content", ['{"verdict": "supported"}', '{"verdict": "not_supported"}', "not json at all"]
)
def test_the_reply_text_is_returned_uninterpreted(content: str):
    """Judging it is the parser's job — the same parser for every provider."""
    provider, _ = make(lambda _: httpx.Response(200, json=completion(content)))

    assert run(provider.generate(REQUEST)).text == content


def test_reasoning_is_never_returned():
    provider, _ = make()

    response = run(provider.generate(REQUEST))

    assert REASONING not in response.text
    assert REASONING not in repr(response)


def test_usage_and_finish_reason_are_reported():
    provider, _ = make()

    response = run(provider.generate(REQUEST))

    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (640, 180)
    assert response.finish_reason == "stop"
    assert response.model == DEFAULT_MODEL
    assert response.transport_attempts == 1


@pytest.mark.parametrize("content", [None, "", "   "])
def test_a_reply_without_text_is_malformed_and_keeps_its_metadata(content: Any):
    provider, _ = make(
        lambda _: httpx.Response(200, json=completion(content, finish_reason="length"))
    )

    error = failure_of(provider)

    assert error.kind is ProviderFailureKind.MALFORMED_RESPONSE
    assert error.finish_reason == "length"
    assert error.usage is not None and error.usage.output_tokens == 180
    # The pipeline's one existing recovery applies, at the larger cap.
    assert provider_failure(error).category is GenerationFailureCategory.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "body", [{}, {"choices": []}, {"choices": ["x"]}, {"choices": [{"finish_reason": "stop"}]}]
)
def test_a_body_that_is_not_a_completion_is_malformed(body: dict[str, Any]):
    provider, _ = make(lambda _: httpx.Response(200, json=body))

    assert failure_of(provider).kind is ProviderFailureKind.MALFORMED_RESPONSE


def test_a_body_that_is_not_json_is_malformed():
    provider, _ = make(lambda _: httpx.Response(200, text="<html>"))

    assert failure_of(provider).kind is ProviderFailureKind.MALFORMED_RESPONSE


# --- failures: bounded, classified, and never carrying upstream text ---------------------


def test_http_400_is_refused_once_and_never_recovered():
    """No error code for a JSON failure is documented, so no 400 is guessed at."""
    provider, seen = make(status(400))

    error = failure_of(provider)

    assert len(seen) == 1
    assert (error.status_code, error.retryable) == (400, False)
    assert provider_failure(error).category is GenerationFailureCategory.PROVIDER_STATUS


@pytest.mark.parametrize("code", [401, 403])
def test_refused_credentials_are_not_retried(code: int):
    provider, seen = make(status(code))

    error = failure_of(provider)

    assert len(seen) == 1
    assert "credentials" in str(error)


def test_http_429_is_not_retried_and_reports_retry_after():
    provider, seen = make(status(429, **{"Retry-After": "7"}))

    error = failure_of(provider)

    assert len(seen) == 1, "asking again a second later spends the per-minute budget"
    assert error.kind is ProviderFailureKind.RATE_LIMITED
    assert error.retry_after_seconds == 7.0
    assert error.retryable is False


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_a_5xx_is_retried_once_then_fails(code: int):
    provider, seen = make(status(code))

    error = failure_of(provider)

    assert len(seen) == 2
    assert (error.status_code, error.transport_attempts) == (code, 2)


def test_a_5xx_followed_by_an_answer_succeeds():
    provider, seen = make(status(503), lambda _: httpx.Response(200, json=completion()))

    response = run(provider.generate(REQUEST))

    assert len(seen) == 2 and response.transport_attempts == 2


def test_a_timeout_is_retried_once_then_fails_as_a_timeout():
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    provider, seen = make(slow)

    error = failure_of(provider)

    assert len(seen) == 2
    assert error.kind is ProviderFailureKind.TIMEOUT
    assert provider_failure(error).category is GenerationFailureCategory.TIMEOUT


def test_an_unreachable_endpoint_is_retried_once_then_fails_as_unreachable():
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider, seen = make(refused)

    error = failure_of(provider)

    assert len(seen) == 2
    assert error.kind is ProviderFailureKind.UNREACHABLE
    assert provider_failure(error).category is GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR
    assert FAKE_KEY not in str(error) and "connection refused" not in str(error)


def test_one_attempt_means_one_request():
    provider, seen = make(status(503), max_attempts=1)

    failure_of(provider)

    assert len(seen) == 1


@pytest.mark.parametrize("code", [400, 401, 403, 429, 500])
def test_no_error_carries_the_key_or_upstream_text(code: int):
    provider, _ = make(status(code))

    error = failure_of(provider)

    assert FAKE_KEY not in str(error) and "UPSTREAM" not in str(error)
    assert FAKE_KEY not in error.describe() and "UPSTREAM" not in error.describe()
