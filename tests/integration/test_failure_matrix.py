"""Every way the query pipeline can fail, and what a client sees when it does.

This is the failure matrix as an executable document rather than a table in a
file that drifts. Each test names one cell — a stage, a fault — and asserts the
four things that matter to a caller: the HTTP status, the stable error code,
that nothing internal is disclosed, and that no answer is invented on the way
out.

It deliberately does **not** re-test the adapters. Which HTTP statuses Mistral
retries, and how a malformed provider body is parsed, are covered where those
adapters live. What is checked here is the mapping: a fault at any stage
becomes one of four outcomes, and never a stack trace, a provider payload or a
half-built answer.

| stage             | fault                          | outcome                       |
| ----------------- | ------------------------------ | ----------------------------- |
| query boundary    | empty / oversized / invisible  | 422 VALIDATION_ERROR          |
| query embedding   | provider unreachable           | 503 RETRIEVAL_UNAVAILABLE     |
| query embedding   | wrong count / id / dimension   | 503 RETRIEVAL_UNAVAILABLE     |
| vector store      | unavailable                    | 503 RETRIEVAL_UNAVAILABLE     |
| vector store      | index space mismatch           | 500 INTERNAL_ERROR            |
| chunk resolution  | corpus unreachable             | 503 RETRIEVAL_UNAVAILABLE     |
| chunk resolution  | id missing from corpus         | 200, fewer or no sources      |
| context           | nothing fits the budget        | 500 INTERNAL_ERROR            |
| generation        | provider unreachable / 5xx     | 503 GENERATION_UNAVAILABLE    |
| generation        | malformed or empty body        | 503 GENERATION_UNAVAILABLE    |
| generation        | reachable, but says nothing    | 200, refusal, no citations    |
| citations         | unknown or forged labels       | 200, refusal, no citations    |
| citations         | duplicate labels               | 200, one citation             |
| retrieval         | nothing above threshold        | 200, refusal, no citations    |
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from portfolio_rag.api.schemas.errors import ErrorResponse
from portfolio_rag.core.config import Settings
from portfolio_rag.domain.embedding import EmbeddingSpec, VectorIndexSpec
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.main import create_app
from portfolio_rag.ports.embeddings import EmbeddingResult
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.service import GroundedAnswerService
from tests.doubles import (
    FailingChunkResolver,
    FailingEmbeddingProvider,
    MiscountingEmbeddingProvider,
    ScriptedLLMProvider,
    ScriptedReply,
    grounded,
)

CHAT_URL = "/api/v1/chat"
QUESTION = {"message": "Which HTTP framework is used?"}

#: Anything in a response body that would mean something leaked.
LEAKS = (
    "Traceback",
    "provider unreachable",
    "store unreachable",
    "corpus unavailable",
    "httpx",
    ".py",
    "portfolio_rag",
    "sk-",
    "Authorization",
)


@contextmanager
def app_client(settings: Settings, **replacements: object) -> Iterator[tuple[TestClient, FastAPI]]:
    app: FastAPI = create_app(settings)
    with TestClient(app) as client:
        service: GroundedAnswerService = app.state.answer_service
        for attribute, replacement in replacements.items():
            setattr(service, attribute, replacement)
        yield client, app


def break_retrieval(app: FastAPI, **replacements: object) -> None:
    """Swap one adapter inside the wired retrieval service."""
    retrieval = app.state.answer_service._retrieval  # a wired adapter, replaced in place
    for attribute, replacement in replacements.items():
        setattr(retrieval, attribute, replacement)


def assert_clean_failure(response: Any, *, status: int, code: str) -> None:
    """The four things a caller is entitled to, in one place."""
    assert response.status_code == status
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == code
    assert body.error.request_id is not None
    assert set(response.json()) == {"error"}, "no answer is returned alongside a failure"
    for leak in LEAKS:
        assert leak not in response.text, leak


# --- query boundary ----------------------------------------------------------


@pytest.mark.parametrize("message", ["", "   ", "​", "y" * 4001])
def test_an_unusable_question_is_a_validation_error(client: TestClient, message: str):
    assert_clean_failure(
        client.post(CHAT_URL, json={"message": message}),
        status=422,
        code="VALIDATION_ERROR",
    )


# --- query embedding ---------------------------------------------------------


def test_an_unreachable_embedding_provider_is_a_retrieval_outage(rag_settings: Settings):
    with app_client(rag_settings) as (client, app):
        spec = app.state.answer_service._retrieval.spec
        break_retrieval(app, _embeddings=FailingEmbeddingProvider(spec))

        assert_clean_failure(
            client.post(CHAT_URL, json=QUESTION),
            status=503,
            code="RETRIEVAL_UNAVAILABLE",
        )


@pytest.mark.parametrize(
    "results",
    [
        pytest.param([], id="no results"),
        pytest.param(
            [
                EmbeddingResult(id="query", vector=(1.0, 0.0)),
                EmbeddingResult(id="q", vector=(1.0,)),
            ],
            id="too many results",
        ),
        pytest.param([EmbeddingResult(id="wrong-id", vector=(1.0,) * 256)], id="wrong id"),
        pytest.param([EmbeddingResult(id="query", vector=(1.0, 0.0))], id="wrong dimensionality"),
    ],
)
def test_an_unusable_embedding_batch_is_a_retrieval_outage(
    rag_settings: Settings, results: list[EmbeddingResult]
):
    """A provider that answers with the wrong shape is as unusable as one that does not."""
    with app_client(rag_settings) as (client, app):
        spec = app.state.answer_service._retrieval.spec
        break_retrieval(app, _embeddings=MiscountingEmbeddingProvider(spec, results=results))

        assert_clean_failure(
            client.post(CHAT_URL, json=QUESTION),
            status=503,
            code="RETRIEVAL_UNAVAILABLE",
        )


# --- vector store ------------------------------------------------------------


def test_an_unreachable_vector_store_is_a_retrieval_outage(failing_store_client: TestClient):
    assert_clean_failure(
        failing_store_client.post(CHAT_URL, json=QUESTION),
        status=503,
        code="RETRIEVAL_UNAVAILABLE",
    )


def test_an_index_in_a_different_embedding_space_is_an_internal_error(rag_settings: Settings):
    """A configuration fault, not a transient one: retrying changes nothing."""
    with app_client(rag_settings) as (client, app):
        retrieval = app.state.answer_service._retrieval
        other_space = EmbeddingSpec(
            provider=retrieval.spec.provider,
            model="a-different-model",
            dimensions=retrieval.spec.dimensions,
            representation_version=retrieval.spec.representation_version,
        )
        break_retrieval(app, _store=InMemoryVectorStore(VectorIndexSpec(embedding=other_space)))

        assert_clean_failure(
            client.post(CHAT_URL, json=QUESTION),
            status=500,
            code="INTERNAL_ERROR",
        )


# --- chunk resolution --------------------------------------------------------


def test_an_unreachable_corpus_is_a_retrieval_outage(rag_settings: Settings):
    with app_client(rag_settings) as (client, app):
        break_retrieval(app, _resolver=FailingChunkResolver())

        assert_clean_failure(
            client.post(CHAT_URL, json=QUESTION),
            status=503,
            code="RETRIEVAL_UNAVAILABLE",
        )


def test_an_index_ahead_of_the_corpus_answers_honestly_rather_than_failing(
    rag_settings: Settings,
):
    """A stale record is a normal deployment window, not an outage."""
    with app_client(rag_settings) as (client, app):
        break_retrieval(app, _resolver=InMemoryChunkResolver([]))

        response = client.post(CHAT_URL, json=QUESTION)

    assert response.status_code == 200
    assert response.json()["citations"] == []


# --- context -----------------------------------------------------------------


def test_a_budget_nothing_fits_into_is_an_internal_error(rag_settings: Settings):
    """Also a configuration fault: chunks are bounded, so this means the budget is wrong."""
    with app_client(rag_settings) as (client, app):
        app.state.answer_service._context_policy = ContextPolicy(
            max_prompt_tokens=400, output_reserve_tokens=380
        )

        assert_clean_failure(
            client.post(CHAT_URL, json=QUESTION),
            status=500,
            code="INTERNAL_ERROR",
        )


# --- generation --------------------------------------------------------------


def test_an_unreachable_generation_provider_is_a_generation_outage(
    failing_generation_client: TestClient,
):
    assert_clean_failure(
        failing_generation_client.post(CHAT_URL, json=QUESTION),
        status=503,
        code="GENERATION_UNAVAILABLE",
    )


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(ScriptedReply(raw_text=""), id="empty body"),
        pytest.param(ScriptedReply(raw_text="not json at all"), id="prose instead of json"),
        pytest.param(ScriptedReply(raw_text="[1, 2, 3]"), id="json but not an object"),
        pytest.param(ScriptedReply(raw_text='{"answer": 42}'), id="answer is not text"),
        pytest.param(ScriptedReply(raw_text='{"answer": "x", "sources": "S1"}'), id="bad sources"),
    ],
)
def test_a_reply_that_breaks_the_contract_fails_closed(
    rag_settings: Settings, reply: ScriptedReply
):
    """Fail closed: no answer at all beats an answer nobody could verify."""
    with app_client(rag_settings, _llm=ScriptedLLMProvider(reply)) as (client, _):
        assert_clean_failure(
            client.post(CHAT_URL, json=QUESTION),
            status=503,
            code="GENERATION_UNAVAILABLE",
        )


# --- citations: not failures, but not answers either -------------------------


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(grounded("Confident.", "S99"), id="forged label"),
        pytest.param(grounded("Confident.", "docs/secret.md"), id="a file name"),
        pytest.param(grounded("Confident.", "https://example.invalid"), id="a url"),
        pytest.param(grounded("Confident."), id="no sources at all"),
    ],
)
def test_an_answer_with_no_verifiable_source_is_replaced_not_published(
    rag_settings: Settings, reply: ScriptedReply
):
    with app_client(rag_settings, _llm=ScriptedLLMProvider(reply)) as (client, _):
        response = client.post(CHAT_URL, json=QUESTION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert payload["answer"].startswith("I don't have enough information")
    assert "Confident." not in payload["answer"]


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(grounded(""), id="empty answer"),
        pytest.param(ScriptedReply(raw_text='{"answer": "   ", "sources": []}'), id="blank answer"),
        pytest.param(ScriptedReply(raw_text='{"sources": ["S1"]}'), id="no answer field"),
        pytest.param(ScriptedReply(raw_text='{"answer": null}'), id="null answer"),
        pytest.param(
            ScriptedReply(raw_text='{"answer": "", "sources": ["S1"]}'), id="blank + label"
        ),
    ],
)
def test_a_reachable_model_that_says_nothing_is_a_refusal_not_an_outage(
    rag_settings: Settings, reply: ScriptedReply
):
    """A 200 refusal, not a 503.

    The provider answered — it just had nothing to say about these passages.
    Observed against a real model on two unanswerable questions; reporting it
    as an upstream outage told the caller to retry something that will never
    succeed, and hid a normal knowledge gap behind an error.
    """
    with app_client(rag_settings, _llm=ScriptedLLMProvider(reply)) as (client, _):
        response = client.post(CHAT_URL, json=QUESTION)

    assert response.status_code == 200
    payload = response.json()
    assert payload["citations"] == []
    assert payload["answer"].startswith("I don't have enough information")


def test_a_repeated_label_produces_one_citation(rag_settings: Settings):
    with app_client(rag_settings, _llm=ScriptedLLMProvider(grounded("Yes.", "S1", "S1", "S1"))) as (
        client,
        _,
    ):
        payload = client.post(CHAT_URL, json=QUESTION).json()

    assert len(payload["citations"]) == 1


def test_a_mixed_claim_keeps_only_the_verifiable_half(rag_settings: Settings):
    with app_client(rag_settings, _llm=ScriptedLLMProvider(grounded("Yes.", "S1", "S99"))) as (
        client,
        _,
    ):
        payload = client.post(CHAT_URL, json=QUESTION).json()

    assert payload["answer"] == "Yes."
    assert len(payload["citations"]) == 1


# --- nothing found is not a failure ------------------------------------------


def test_nothing_above_the_threshold_is_a_successful_refusal(client: TestClient):
    """`client` runs against the empty repository corpus: nothing can be found."""
    response = client.post(CHAT_URL, json=QUESTION)

    assert response.status_code == 200
    assert response.json()["citations"] == []
