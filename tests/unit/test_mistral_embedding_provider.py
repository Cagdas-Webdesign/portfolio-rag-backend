"""The Mistral adapter, without a network and without a key.

Requests are served by an in-process transport, so every branch — the happy
path, each failure mode, the retry policy — is exercised deterministically and
for free. The live smoke test is separate and opt-in.

No real credential appears anywhere in this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.embedding.mistral import (
    DEFAULT_MODEL,
    MistralEmbeddingProvider,
)
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION
from portfolio_rag.ports.embeddings import EmbeddingInput
from portfolio_rag.ports.errors import EmbeddingProviderError
from tests.support import run

FAKE_KEY = "test-key-not-a-real-credential"
DIMENSIONS = 4


def _embedding(seed: float = 0.1) -> list[float]:
    return [seed + index for index in range(DIMENSIONS)]


def _ok(count: int) -> dict[str, Any]:
    return {
        "data": [{"index": index, "embedding": _embedding(float(index))} for index in range(count)],
        "model": DEFAULT_MODEL,
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
    return lambda _: httpx.Response(status_code, json={"message": "no secrets here"})


def raising(exception: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def _raise(_: httpx.Request) -> httpx.Response:
        raise exception

    return _raise


def build(
    *responses: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[MistralEmbeddingProvider, _Recorder, list[float]]:
    recorder = _Recorder(responses or (json_response(_ok(1)),))
    slept: list[float] = []

    async def sleeper(delay: float) -> None:
        slept.append(delay)

    client = httpx.AsyncClient(
        base_url="https://api.mistral.ai",
        transport=httpx.MockTransport(recorder),
        headers={"Authorization": f"Bearer {FAKE_KEY}", "Content-Type": "application/json"},
    )
    provider = MistralEmbeddingProvider(
        api_key=FAKE_KEY,
        dimensions=DIMENSIONS,
        client=client,
        sleeper=sleeper,
        **kwargs,
    )
    return provider, recorder, slept


def _inputs(count: int) -> list[EmbeddingInput]:
    return [EmbeddingInput(id=f"fp{index}", text=f"text {index}") for index in range(count)]


# --- configuration ----------------------------------------------------------


def test_the_adapter_declares_a_complete_embedding_space():
    provider, _, _ = build()

    assert provider.spec.provider == "mistral"
    assert provider.spec.model == DEFAULT_MODEL
    assert provider.spec.dimensions == DIMENSIONS
    assert provider.spec.representation_version == EMBEDDING_REPRESENTATION_VERSION


def test_an_empty_api_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="API key"):
        MistralEmbeddingProvider(api_key="")


# --- the happy path ---------------------------------------------------------


def test_a_batch_becomes_one_request_with_every_text():
    provider, recorder, _ = build(json_response(_ok(3)))

    results = run(provider.embed(_inputs(3)))

    assert len(recorder.requests) == 1
    body = json.loads(recorder.requests[0].content)
    assert body == {"model": DEFAULT_MODEL, "input": ["text 0", "text 1", "text 2"]}
    assert [result.id for result in results] == ["fp0", "fp1", "fp2"]


def test_the_request_carries_the_bearer_token_and_targets_the_embeddings_endpoint():
    provider, recorder, _ = build()

    run(provider.embed(_inputs(1)))

    request = recorder.requests[0]
    assert request.url.path == "/v1/embeddings"
    assert request.headers["Authorization"] == f"Bearer {FAKE_KEY}"


def test_results_keep_the_ids_of_the_inputs_that_produced_them():
    provider, _, _ = build(json_response(_ok(2)))

    results = run(provider.embed(_inputs(2)))

    assert [result.id for result in results] == ["fp0", "fp1"]
    assert results[0].vector == tuple(_embedding(0.0))
    assert results[1].vector == tuple(_embedding(1.0))


def test_a_large_batch_is_split_into_provider_sized_requests():
    """Batch limits belong to the adapter; the caller passes everything."""
    provider, recorder, _ = build(json_response(_ok(2)), batch_size=2)

    results = run(provider.embed(_inputs(4)))

    assert len(recorder.requests) == 2
    assert [result.id for result in results] == ["fp0", "fp1", "fp2", "fp3"]


def test_an_empty_input_list_makes_no_request():
    provider, recorder, _ = build()

    assert run(provider.embed([])) == []
    assert recorder.requests == []


# --- malformed responses ----------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({}, "missing its `data`"),
        ({"data": "not-a-list"}, "missing its `data`"),
        (
            {"data": [{"embedding": _embedding()}, {"embedding": _embedding()}]},
            "2 embeddings for 1",
        ),
        ({"data": [{"index": 0}]}, "no embedding"),
        ({"data": [{"embedding": []}]}, "no embedding"),
        ({"data": [{"embedding": [0.1, 0.2]}]}, "dimensions"),
        ({"data": [{"embedding": ["a", "b", "c", "d"]}]}, "non-numeric"),
        ({"data": ["not-an-object"]}, "not an object"),
    ],
)
def test_a_response_that_is_not_usable_is_rejected(payload: dict[str, Any], match: str):
    provider, _, _ = build(json_response(payload))

    with pytest.raises(EmbeddingProviderError, match=match):
        run(provider.embed(_inputs(1)))


def test_a_non_finite_value_is_rejected():
    """`NaN` is legal in a JSON body Python will parse, and poison in a vector."""
    provider, _, _ = build(
        lambda _: httpx.Response(200, content=b'{"data":[{"embedding":[NaN,0.0,0.0,0.0]}]}')
    )

    with pytest.raises(EmbeddingProviderError, match="non-finite"):
        run(provider.embed(_inputs(1)))


def test_a_non_json_body_is_rejected():
    provider, _, _ = build(lambda _: httpx.Response(200, content=b"not json"))

    with pytest.raises(EmbeddingProviderError, match="not JSON"):
        run(provider.embed(_inputs(1)))


def test_a_json_array_instead_of_an_object_is_rejected():
    provider, _, _ = build(json_response([1, 2, 3]))

    with pytest.raises(EmbeddingProviderError, match="not an object"):
        run(provider.embed(_inputs(1)))


# --- HTTP failures ----------------------------------------------------------


@pytest.mark.parametrize("status_code", [401, 403])
def test_an_authentication_failure_is_not_retried(status_code: int):
    """A 401 will not become a 200 by asking again."""
    provider, recorder, slept = build(status_response(status_code))

    with pytest.raises(EmbeddingProviderError, match="credentials"):
        run(provider.embed(_inputs(1)))

    assert len(recorder.requests) == 1
    assert slept == []


def test_a_rate_limit_is_retried_a_bounded_number_of_times():
    provider, recorder, slept = build(status_response(429), max_attempts=3)

    with pytest.raises(EmbeddingProviderError, match="rate limit"):
        run(provider.embed(_inputs(1)))

    assert len(recorder.requests) == 3
    assert len(slept) == 2


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_server_errors_are_retried(status_code: int):
    provider, recorder, _ = build(status_response(status_code), max_attempts=2)

    with pytest.raises(EmbeddingProviderError, match=f"HTTP {status_code}"):
        run(provider.embed(_inputs(1)))

    assert len(recorder.requests) == 2


def test_a_client_error_that_is_not_transient_is_not_retried():
    provider, recorder, _ = build(status_response(400), max_attempts=3)

    with pytest.raises(EmbeddingProviderError, match="HTTP 400"):
        run(provider.embed(_inputs(1)))

    assert len(recorder.requests) == 1


def test_a_retry_that_succeeds_returns_the_result():
    provider, recorder, slept = build(status_response(503), json_response(_ok(1)), max_attempts=3)

    results = run(provider.embed(_inputs(1)))

    assert len(results) == 1
    assert len(recorder.requests) == 2
    assert slept == [pytest.approx(1.0)]


def test_a_timeout_is_reported_as_a_provider_failure():
    provider, _, _ = build(raising(httpx.TimeoutException("too slow")), max_attempts=1)

    with pytest.raises(EmbeddingProviderError, match="timed out"):
        run(provider.embed(_inputs(1)))


def test_a_network_failure_is_reported_as_a_provider_failure():
    provider, _, _ = build(raising(httpx.ConnectError("no route")), max_attempts=1)

    with pytest.raises(EmbeddingProviderError, match="could not be reached"):
        run(provider.embed(_inputs(1)))


def test_a_transient_failure_is_retried_then_given_up_on():
    provider, recorder, slept = build(
        raising(httpx.ConnectError("no route")), max_attempts=2, retry_delay_seconds=0.25
    )

    with pytest.raises(EmbeddingProviderError):
        run(provider.embed(_inputs(1)))

    assert len(recorder.requests) == 2
    assert slept == [pytest.approx(0.25)]


# --- what never leaks -------------------------------------------------------


def test_no_error_message_ever_contains_the_api_key():
    for handler in (status_response(401), status_response(500), json_response({})):
        provider, _, _ = build(handler, max_attempts=1)
        try:
            run(provider.embed(_inputs(1)))
        except EmbeddingProviderError as error:
            assert FAKE_KEY not in str(error)
            assert "Bearer" not in str(error)
        else:  # pragma: no cover - every handler above fails
            pytest.fail("expected a provider error")


def test_the_provider_body_is_not_echoed_into_the_error():
    """Provider bodies can contain request content; only the status is reported."""
    provider, _, _ = build(
        lambda _: httpx.Response(500, json={"detail": "sensitive upstream detail"}),
        max_attempts=1,
    )

    with pytest.raises(EmbeddingProviderError) as caught:
        run(provider.embed(_inputs(1)))

    assert "sensitive upstream detail" not in str(caught.value)


def test_only_the_embedding_text_is_sent():
    """No ids, no metadata, no fingerprints — just what has to be embedded."""
    provider, recorder, _ = build(json_response(_ok(1)))

    run(provider.embed([EmbeddingInput(id="fingerprint-value", text="the text")]))

    body = json.loads(recorder.requests[0].content)
    assert body == {"model": DEFAULT_MODEL, "input": ["the text"]}
    assert "fingerprint-value" not in recorder.requests[0].content.decode()


def test_invalid_configuration_is_refused_at_construction():
    with pytest.raises(ValueError, match="batch_size"):
        MistralEmbeddingProvider(api_key=FAKE_KEY, batch_size=0)
    with pytest.raises(ValueError, match="max_attempts"):
        MistralEmbeddingProvider(api_key=FAKE_KEY, max_attempts=0)
