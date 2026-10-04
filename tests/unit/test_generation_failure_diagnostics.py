"""Why a generation failed, said in facts that are safe to write down.

Three layers, each adding what only it can know: an adapter says what happened
on the wire, the reply parser says which rule a reply broke, and the answering
service adds how the reply ended and what had been retrieved. None of them
changes what a client receives, and none of them may carry content.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.evaluation import GenerationPacing, PacedLLMProvider
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationRequest, MessageRole, PromptMessage
from portfolio_rag.rag.errors import (
    GenerationFailureCategory,
    GenerationUnavailableError,
    provider_failure,
)
from portfolio_rag.rag.generation import parse_generation
from tests.doubles import ScriptedLLMProvider, ScriptedReply
from tests.support import run
from tests.unit import test_mistral_chat_provider as mistral
from tests.unit import test_workers_ai_chat_provider as workers_ai
from tests.unit.test_answer_service import QUESTION, build_service

REQUEST = GenerationRequest(messages=(PromptMessage(role=MessageRole.USER, content="question"),))
ADAPTERS = pytest.mark.parametrize("adapter", [workers_ai, mistral], ids=["workers_ai", "mistral"])


def _failure_of(adapter: Any, *responses: Any, **kwargs: Any) -> LLMProviderError:
    provider, _ = adapter.make_provider(*responses, **kwargs)
    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))
    return caught.value


# --- what an adapter reports ---------------------------------------------------


@ADAPTERS
def test_a_refused_request_reports_its_status_and_is_tried_once(adapter: Any):
    error = _failure_of(adapter, adapter.status_response(400))

    assert error.kind is ProviderFailureKind.HTTP_STATUS
    assert (error.status_code, error.retryable, error.attempts) == (400, False, 1)


@ADAPTERS
def test_a_rate_limit_reports_every_request_it_cost(adapter: Any):
    error = _failure_of(adapter, adapter.throttled("2"))

    assert error.kind is ProviderFailureKind.RATE_LIMITED
    assert (error.status_code, error.retryable, error.attempts) == (429, True, 3)


@ADAPTERS
def test_a_timeout_is_told_apart_from_an_unreachable_provider(adapter: Any):
    timed_out = _failure_of(adapter, adapter.raising(httpx.ReadTimeout("slow")), max_attempts=1)
    unreachable = _failure_of(adapter, adapter.raising(httpx.ConnectError("down")), max_attempts=1)

    assert timed_out.kind is ProviderFailureKind.TIMEOUT
    assert unreachable.kind is ProviderFailureKind.UNREACHABLE
    assert timed_out.status_code is None


@ADAPTERS
def test_a_body_that_is_not_a_completion_is_a_malformed_response(adapter: Any):
    error = _failure_of(adapter, adapter.json_response({"unexpected": True}))

    assert error.kind is ProviderFailureKind.MALFORMED_RESPONSE
    assert error.retryable is False


@ADAPTERS
def test_no_diagnostic_carries_what_the_provider_sent_back(adapter: Any):
    """The error body in these tests contains a marker; it must stay there."""
    error = _failure_of(adapter, adapter.status_response(403))

    assert "token-should-never-be-echoed" not in repr(vars(error))
    assert "token-should-never-be-echoed" not in str(provider_failure(error).fields())


# --- how the query side names it ------------------------------------------------


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (
            LLMProviderError("t", retryable=True, kind=ProviderFailureKind.TIMEOUT),
            GenerationFailureCategory.TIMEOUT,
        ),
        (
            LLMProviderError(
                "r", retryable=True, kind=ProviderFailureKind.RATE_LIMITED, status_code=429
            ),
            GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR,
        ),
        (
            LLMProviderError(
                "s", retryable=True, kind=ProviderFailureKind.HTTP_STATUS, status_code=503
            ),
            GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR,
        ),
        (
            LLMProviderError("u", retryable=True, kind=ProviderFailureKind.UNREACHABLE),
            GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR,
        ),
        (
            LLMProviderError("b", kind=ProviderFailureKind.HTTP_STATUS, status_code=400),
            GenerationFailureCategory.PROVIDER_STATUS,
        ),
        (
            LLMProviderError("m", kind=ProviderFailureKind.MALFORMED_RESPONSE),
            GenerationFailureCategory.MALFORMED_RESPONSE,
        ),
        (LLMProviderError("?"), GenerationFailureCategory.UNCLASSIFIED),
    ],
)
def test_a_provider_failure_falls_into_exactly_one_category(
    error: LLMProviderError, category: GenerationFailureCategory
):
    failure = provider_failure(error)

    assert failure.category is category
    assert failure.detail == error.kind.value
    assert failure.status_code == error.status_code
    assert failure.retryable is error.retryable


@pytest.mark.parametrize(
    ("reply", "rule"),
    [
        ("", "reply_empty"),
        ("   ", "reply_empty"),
        ("The answer is FastAPI.", "reply_not_json"),
        ('{"answer": "cut off in the mid', "reply_not_json"),
        ('["S1"]', "reply_not_an_object"),
        ('{"answer": 3, "sources": []}', "answer_not_text"),
        ('{"answer": "x", "sources": "S1"}', "sources_not_a_list"),
        ('{"answer": "x", "sources": [1]}', "source_not_text"),
    ],
)
def test_an_unusable_reply_names_the_rule_it_broke(reply: str, rule: str):
    with pytest.raises(GenerationUnavailableError) as caught:
        parse_generation(reply)

    failure = caught.value.failure
    assert failure is not None
    assert failure.category is GenerationFailureCategory.UNPARSEABLE_OUTPUT
    assert failure.detail == rule


# --- what the answering service adds ---------------------------------------------

REPLY = "Certainly! FastAPI is the framework, as passage S1 says."


def test_an_unusable_reply_is_described_without_being_quoted(
    caplog: pytest.LogCaptureFixture,
):
    service, _ = build_service(llm=ScriptedLLMProvider(ScriptedReply(raw_text=REPLY)))

    with caplog.at_level(logging.WARNING), pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    failure = caught.value.failure
    assert failure is not None
    assert failure.category is GenerationFailureCategory.UNPARSEABLE_OUTPUT
    assert failure.detail == "reply_not_json"
    assert failure.finish_reason == "stop"
    assert failure.reply_characters == len(REPLY)

    record = next(r for r in caplog.records if r.getMessage() == "generation unusable")
    assert record.failure_detail == "reply_not_json"  # type: ignore[attr-defined]
    assert record.reply_characters == len(REPLY)  # type: ignore[attr-defined]
    logged = " ".join(str(vars(r)) for r in caplog.records)
    assert "Certainly" not in logged
    assert QUESTION not in logged


def test_a_provider_failure_is_logged_with_its_category(caplog: pytest.LogCaptureFixture):
    error = LLMProviderError(
        "Provider returned HTTP 400 during generation.",
        kind=ProviderFailureKind.HTTP_STATUS,
        status_code=400,
    )
    service, _ = build_service(llm=ScriptedLLMProvider(ScriptedReply(error=error)))

    with caplog.at_level(logging.WARNING), pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    failure = caught.value.failure
    assert failure is not None
    assert (failure.category, failure.status_code) == (
        GenerationFailureCategory.PROVIDER_STATUS,
        400,
    )
    record = next(r for r in caplog.records if r.getMessage() == "generation failed")
    assert record.failure_category == "provider_status"  # type: ignore[attr-defined]
    assert record.status_code == 400  # type: ignore[attr-defined]


def test_a_failed_generation_keeps_what_was_retrieved_for_it():
    service, _ = build_service(llm=ScriptedLLMProvider(ScriptedReply(raw_text=REPLY)))

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    assert caught.value.retrieval is not None
    assert caught.value.retrieval.chunks


def test_the_client_is_told_exactly_what_it_was_told_before():
    """Diagnostics ride on the exception; the message and the code do not move."""
    service, _ = build_service(llm=ScriptedLLMProvider(ScriptedReply(raw_text=REPLY)))

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    assert caught.value.message == GenerationUnavailableError.default_message
    assert caught.value.code.value == "GENERATION_UNAVAILABLE"
    assert str(caught.value) == GenerationUnavailableError.default_message


# --- the evaluation pacer --------------------------------------------------------


def test_the_pacer_reports_every_request_a_generation_cost():
    limited = LLMProviderError("limit", retryable=True, kind=ProviderFailureKind.RATE_LIMITED)

    async def no_sleep(_: float) -> None:
        return None

    paced = PacedLLMProvider(
        ScriptedLLMProvider(ScriptedReply(error=limited)),
        GenerationPacing(max_attempts=3),
        sleeper=no_sleep,
    )

    with pytest.raises(LLMProviderError) as caught:
        run(paced.generate(REQUEST))

    assert caught.value.attempts == 3


def test_a_failure_not_worth_retrying_was_tried_once():
    refused = LLMProviderError("refused", kind=ProviderFailureKind.HTTP_STATUS, status_code=400)
    paced = PacedLLMProvider(
        ScriptedLLMProvider(ScriptedReply(error=refused)), GenerationPacing(max_attempts=3)
    )

    with pytest.raises(LLMProviderError) as caught:
        run(paced.generate(REQUEST))

    assert caught.value.attempts == 1
