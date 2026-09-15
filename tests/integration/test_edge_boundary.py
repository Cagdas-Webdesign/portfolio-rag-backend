"""The guards that run before a request is allowed to cost anything.

Two questions, both answered before routing: did this come through the
gateway, and is it small enough to be a question? Everything here is about
what does *not* happen — no embedding, no Vectorize query, no generation — so
the assertions are as much about call counts on the adapters as about status
codes.

The shared secret in this file is a fixture string. No real credential appears
anywhere, and none is needed: the guard compares two strings.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from portfolio_rag.api.middleware.gate import DEFAULT_MAX_BODY_BYTES, EDGE_AUTH_HEADER
from portfolio_rag.api.schemas.chat import MAX_MESSAGE_LENGTH
from portfolio_rag.composition import ConfigurationError
from portfolio_rag.core.config import Environment, Settings
from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.main import create_app
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingProvider, EmbeddingResult
from tests.doubles import ScriptedLLMProvider, grounded

CHAT_URL = "/api/v1/chat"
HEALTH_URL = "/health"

EDGE_SECRET = "test-edge-secret-not-a-real-credential"  # noqa: S105 - a fixture value
WRONG_SECRET = "test-edge-secret-not-a-real-credentiaX"  # noqa: S105 - one character off


@pytest.fixture
def guarded_settings(rag_settings: Settings) -> Settings:
    """The production posture, without needing a production provider."""
    return rag_settings.model_copy(
        update={"require_edge_auth": True, "edge_shared_secret": SecretStr(EDGE_SECRET)}
    )


@pytest.fixture
def guarded(guarded_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(guarded_settings)) as client:
        yield client


class _CountingEmbeddings:
    """The real offline provider, plus a count of the calls it was asked for.

    A real object rather than a mock: what is being measured is whether the
    request reached the embedding step at all, and that has to be observed on
    the adapter the application actually uses.
    """

    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self.call_count = 0

    @property
    def spec(self) -> EmbeddingSpec:
        return self._inner.spec

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        self.call_count += 1
        return await self._inner.embed(inputs)


def _counted(client: TestClient) -> tuple[_CountingEmbeddings, ScriptedLLMProvider]:
    """Install counters on the running application's query stack.

    Reaching into the started service is the same trick `test_chat.py` uses,
    and for the same reason: the alternative is a second composition path that
    exists only for tests, and then the thing under test is not the thing that
    runs.
    """
    app = cast(FastAPI, client.app)
    service = app.state.answer_service
    embeddings = _CountingEmbeddings(service._retrieval._embeddings)
    llm = ScriptedLLMProvider(grounded("A scripted answer.", "S1"))
    service._retrieval._embeddings = embeddings
    service._llm = llm
    return embeddings, llm


# --- edge authentication ------------------------------------------------------


def test_a_request_without_the_edge_header_is_refused(guarded: TestClient):
    response = guarded.post(CHAT_URL, json={"message": "Which framework is used?"})

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_a_request_with_the_wrong_edge_secret_is_refused(guarded: TestClient):
    response = guarded.post(
        CHAT_URL,
        json={"message": "Which framework is used?"},
        headers={EDGE_AUTH_HEADER: WRONG_SECRET},
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    "presented",
    ["", " ", EDGE_SECRET + " ", EDGE_SECRET[:-1], EDGE_SECRET.upper(), "Bearer " + EDGE_SECRET],
)
def test_anything_that_is_not_exactly_the_secret_is_refused(guarded: TestClient, presented: str):
    """No trimming, no prefix, no case folding — an exact match or nothing."""
    response = guarded.post(
        CHAT_URL,
        json={"message": "Which framework is used?"},
        headers={EDGE_AUTH_HEADER: presented},
    )

    assert response.status_code == 403


def test_the_correct_edge_secret_reaches_the_normal_path(guarded: TestClient):
    response = guarded.post(
        CHAT_URL,
        json={"message": "Which HTTP framework is used?"},
        headers={EDGE_AUTH_HEADER: EDGE_SECRET},
    )

    assert response.status_code == 200
    assert "answer" in response.json()


def test_a_refused_request_never_reaches_a_provider(guarded: TestClient):
    """The point of the guard: a rejected caller costs nothing but a string compare."""
    embeddings, llm = _counted(guarded)

    for _ in range(5):
        guarded.post(CHAT_URL, json={"message": "expensive question"})
        guarded.post(
            CHAT_URL,
            json={"message": "expensive question"},
            headers={EDGE_AUTH_HEADER: WRONG_SECRET},
        )

    assert embeddings.call_count == 0, "no question was ever embedded"
    assert llm.call_count == 0, "no answer was ever generated"


def test_an_accepted_request_does_reach_the_providers(guarded: TestClient):
    """The other half of the previous test: these counters can move at all.

    Without this, the assertion above would hold just as well on a counter
    that is never wired up to anything.
    """
    embeddings, llm = _counted(guarded)

    guarded.post(
        CHAT_URL,
        json={"message": "Which HTTP framework is used?"},
        headers={EDGE_AUTH_HEADER: EDGE_SECRET},
    )

    assert embeddings.call_count == 1
    assert llm.call_count == 1


def test_the_refusal_says_nothing_about_the_secret(guarded: TestClient):
    response = guarded.post(
        CHAT_URL, json={"message": "hi"}, headers={EDGE_AUTH_HEADER: WRONG_SECRET}
    )

    body = response.text
    assert EDGE_SECRET not in body
    assert WRONG_SECRET not in body
    assert EDGE_AUTH_HEADER not in body, "not even the name of the header it wanted"
    assert "Traceback" not in body


def test_health_stays_public_behind_the_guard(guarded: TestClient):
    """A platform probe carries no shared secret and must still succeed."""
    assert guarded.get(HEALTH_URL).status_code == 200


def test_the_openapi_document_stays_reachable(guarded: TestClient):
    assert guarded.get("/openapi.json").status_code == 200


def test_the_guard_is_absent_by_default_so_development_is_unchanged(client: TestClient):
    response = client.post(CHAT_URL, json={"message": "Which HTTP framework is used?"})

    assert response.status_code == 200


def test_the_guard_is_on_by_default_in_production(rag_settings: Settings):
    production = rag_settings.model_copy(update={"environment": Environment.PRODUCTION})

    assert production.edge_auth_required is True


def test_the_guard_can_be_switched_on_outside_production(rag_settings: Settings):
    assert rag_settings.edge_auth_required is False
    assert rag_settings.model_copy(update={"require_edge_auth": True}).edge_auth_required is True


def test_a_production_app_without_a_secret_refuses_to_start(rag_settings: Settings):
    """Fail-closed: starting unguarded is the failure this guard exists to prevent."""
    unguarded = rag_settings.model_copy(update={"require_edge_auth": True})

    with pytest.raises(ConfigurationError, match="EDGE_SHARED_SECRET"):
        create_app(unguarded)


def test_the_secret_is_never_rendered_in_settings(guarded_settings: Settings):
    assert EDGE_SECRET not in repr(guarded_settings)
    assert EDGE_SECRET not in str(guarded_settings)


def test_the_edge_header_is_not_part_of_any_request_model(guarded: TestClient):
    """It is transport. A client has no business seeing it in a schema."""
    schema = guarded.get("/openapi.json").json()

    assert EDGE_AUTH_HEADER.lower() not in str(schema).lower()


# --- body size ----------------------------------------------------------------


def test_a_body_over_the_limit_is_refused(client: TestClient):
    oversized = "y" * (DEFAULT_MAX_BODY_BYTES + 1)

    response = client.post(CHAT_URL, json={"message": oversized})

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_an_oversized_body_never_reaches_a_provider(client: TestClient):
    embeddings, llm = _counted(client)

    client.post(CHAT_URL, json={"message": "y" * (DEFAULT_MAX_BODY_BYTES * 4)})

    assert (embeddings.call_count, llm.call_count) == (0, 0)


def test_an_oversized_body_is_refused_even_without_a_content_length(client: TestClient):
    """A chunked request makes no size claim, so the body itself is counted."""

    def chunks() -> Iterator[bytes]:
        yield b'{"message": "'
        for _ in range(40):
            yield b"y" * 1024
        yield b'"}'

    response = client.post(CHAT_URL, content=chunks(), headers={"Content-Type": "application/json"})

    assert response.status_code == 413


def test_a_lying_content_length_does_not_get_past_the_counter(client: TestClient):
    """The header is a claim; the bytes are the fact."""
    response = client.post(
        CHAT_URL,
        content=b'{"message": "' + b"y" * (DEFAULT_MAX_BODY_BYTES * 2) + b'"}',
        headers={"Content-Type": "application/json", "Transfer-Encoding": "chunked"},
    )

    assert response.status_code == 413


def test_a_normal_question_is_unaffected_by_the_size_guard(client: TestClient):
    response = client.post(CHAT_URL, json={"message": "Which HTTP framework is used?"})

    assert response.status_code == 200


def test_the_longest_allowed_question_still_fits_the_body_limit(client: TestClient):
    """The two limits must not contradict each other: 2000 characters is a
    field-level `422` at worst, never a `413`."""
    response = client.post(CHAT_URL, json={"message": "y" * MAX_MESSAGE_LENGTH})

    assert response.status_code == 200
    assert len(f'{{"message": "{"y" * MAX_MESSAGE_LENGTH}"}}') < DEFAULT_MAX_BODY_BYTES


def test_the_size_guard_does_not_apply_to_health(client: TestClient):
    assert client.get(HEALTH_URL).status_code == 200


# --- the contract -------------------------------------------------------------


def test_openapi_documents_both_refusals(client: TestClient):
    """The document promises only what the code can produce — and all of it."""
    responses = client.get("/openapi.json").json()["paths"]["/api/v1/chat"]["post"]["responses"]

    assert "403" in responses
    assert "413" in responses
    assert "422" in responses
