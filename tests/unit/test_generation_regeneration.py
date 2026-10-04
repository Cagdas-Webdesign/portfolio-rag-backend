"""One more generation, when the first reply was a bad sample — and only then.

Measured against the real provider, three questions of a 49-question run ended
as ``GENERATION_UNAVAILABLE`` after a single request each: two replies were cut
off at the output limit before the JSON object closed, and one response was not
a completion at all. The same model had answered the same questions in full,
under the same limit, in earlier runs. Nothing was wrong with the parser, which
rejected what it should have; what was missing was a second request.

What is tested here is the boundary of that second request: which failures get
one, that there is never a third, and that a regenerated reply is held to every
check the first one would have been. Everything downstream — citation
validation, the support verdict, the grounding check, the refusal wording — is
exercised through it on purpose, because the point of the change is that none
of it moved.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.llm import WorkersAIChatProvider
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.rag.errors import (
    GenerationFailureCategory,
    GenerationUnavailableError,
    ProviderCallType,
)
from portfolio_rag.rag.generation import SupportVerdict
from portfolio_rag.rag.service import (
    INSUFFICIENT_KNOWLEDGE_ANSWER,
    MAX_GENERATION_ATTEMPTS,
    AnswerOutcome,
)
from portfolio_rag.rag.verification import GroundingVerdict
from tests.doubles import ScriptedLLMProvider, ScriptedReply, checked, grounded
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run
from tests.unit.test_answer_service import QUESTION, build_service

#: A reply that stopped mid-object because the output limit was reached.
TRUNCATED = ScriptedReply(
    raw_text='{"answer": "It uses FastAPI for the HTTP API and', finish_reason="length"
)
GOOD = grounded("It uses FastAPI [S1].", "S1")
MALFORMED = ScriptedReply(
    error=LLMProviderError(
        "The provider returned no completion.", kind=ProviderFailureKind.MALFORMED_RESPONSE
    )
)


def _answer(*replies: ScriptedReply, check: ScriptedReply | None = None):
    llm = ScriptedLLMProvider(*replies, grounding_check=check)
    service, _ = build_service(llm=llm)
    return run(service.answer(QUESTION)), llm


def _failure(*replies: ScriptedReply, check: ScriptedReply | None = None):
    llm = ScriptedLLMProvider(*replies, grounding_check=check)
    service, _ = build_service(llm=llm)
    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))
    return caught.value, llm


# --- nothing changes for a reply that was fine -------------------------------------


def test_a_valid_reply_is_generated_once():
    answer, llm = _answer(GOOD)

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "It uses FastAPI [1]."
    assert answer.generation_attempts == 1
    assert llm.call_count == 1 and llm.check_count == 1


def test_a_complete_reply_is_accepted_even_when_the_provider_says_length():
    """The parser decides, not the finish reason: an object that closed is an object."""
    complete = ScriptedReply(
        raw_text=json.dumps(
            {"answer": "It uses FastAPI [S1].", "sources": ["S1"], "support": "stated"}
        ),
        finish_reason="length",
    )

    answer, llm = _answer(complete)

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 1
    assert llm.call_count == 1


def test_the_budget_is_one_extra_request():
    assert MAX_GENERATION_ATTEMPTS == 2


# --- a reply cut off at the output limit -------------------------------------------


def test_a_truncated_reply_is_rejected_by_the_parser_and_classified_as_truncated():
    """The parser rejects it as it always did; the precedence names the cause."""
    error, _ = _failure(TRUNCATED)

    assert error.failure is not None
    assert error.failure.category is GenerationFailureCategory.OUTPUT_TRUNCATED
    assert error.failure.detail == "reply_not_json", "the parser's rule is kept"
    assert error.failure.finish_reason == "length"


def test_a_truncated_reply_is_generated_again_and_the_second_reply_is_used():
    answer, llm = _answer(TRUNCATED, GOOD)

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "It uses FastAPI [1]."
    assert [citation.document_id for citation in answer.citations] == ["backend"]
    assert answer.generation_attempts == 2
    assert llm.call_count == 2
    # One answer, one grounding check: the discarded reply was never checked.
    assert llm.check_count == 1


def test_the_second_request_is_the_same_request():
    """Nothing is loosened for the retry: same prompt, same limit, same format."""
    _, llm = _answer(TRUNCATED, GOOD)

    assert llm.requests[0] == llm.requests[1]


def test_nothing_of_a_truncated_reply_reaches_the_answer():
    answer, _ = _answer(TRUNCATED, grounded("Something else entirely [S1].", "S1"))

    assert answer.answer == "Something else entirely [1]."
    assert "HTTP API and" not in answer.answer


def test_a_reply_that_stays_truncated_fails_closed_after_two_requests():
    error, llm = _failure(TRUNCATED, TRUNCATED, GOOD)

    assert llm.call_count == 2, "never a third request"
    assert llm.check_count == 0
    failure = error.failure
    assert failure is not None
    assert failure.category is GenerationFailureCategory.OUTPUT_TRUNCATED
    assert failure.detail == "reply_not_json"
    assert failure.finish_reason == "length"
    assert failure.attempts == 2
    # What was retrieved still travels with the failure.
    assert error.retrieval is not None and error.retrieval.chunks
    assert "FastAPI" not in str(error)


def test_a_regeneration_is_logged_without_quoting_the_reply(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        _answer(TRUNCATED, GOOD)

    records = [record for record in caplog.records if record.message == "generation regenerated"]
    assert len(records) == 1
    assert records[0].finish_reason == "length"  # type: ignore[attr-defined]
    assert records[0].failure_detail == "reply_not_json"  # type: ignore[attr-defined]
    assert "HTTP API and" not in caplog.text


# --- a response that was not a completion ------------------------------------------


def test_a_malformed_provider_response_is_asked_for_again():
    answer, llm = _answer(MALFORMED, GOOD)

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 2
    assert llm.call_count == 2 and llm.check_count == 1


def test_a_provider_that_stays_malformed_fails_closed_and_keeps_its_classification():
    error, llm = _failure(MALFORMED, MALFORMED, GOOD)

    assert llm.call_count == 2, "never a third request"
    failure = error.failure
    assert failure is not None
    assert failure.category is GenerationFailureCategory.MALFORMED_RESPONSE
    assert failure.detail == "malformed_response"
    assert failure.retryable is False
    assert failure.attempts == 2


def test_two_different_bad_samples_still_end_after_two_requests():
    error, llm = _failure(TRUNCATED, MALFORMED, GOOD)

    assert llm.call_count == 2
    assert error.failure is not None
    assert error.failure.detail == "malformed_response"
    assert error.failure.attempts == 2


# --- everything else is reported exactly as before ---------------------------------


@pytest.mark.parametrize(
    ("reply", "detail"),
    [
        pytest.param(
            ScriptedReply(raw_text="Sure! It uses FastAPI."), "reply_not_json", id="prose"
        ),
        pytest.param(ScriptedReply(raw_text=""), "reply_empty", id="empty"),
        pytest.param(ScriptedReply(raw_text="[1, 2]"), "reply_not_an_object", id="not an object"),
        pytest.param(ScriptedReply(raw_text='{"answer": 42}'), "answer_not_text", id="wrong type"),
        pytest.param(
            ScriptedReply(raw_text='{"answer": "x", "sources": ["S1"], "support": "maybe"}'),
            "support_not_recognized",
            id="unknown verdict",
        ),
    ],
)
def test_a_reply_the_model_finished_and_got_wrong_is_not_regenerated(
    reply: ScriptedReply, detail: str
):
    """A model that stopped on its own and ignored the contract is not a cut-off sample."""
    error, llm = _failure(reply, GOOD)

    assert llm.call_count == 1
    assert error.failure is not None
    assert error.failure.category is GenerationFailureCategory.UNPARSEABLE_OUTPUT
    assert error.failure.detail == detail
    assert error.failure.attempts == 1


@pytest.mark.parametrize(
    ("raised", "category"),
    [
        pytest.param(
            LLMProviderError("t", retryable=True, kind=ProviderFailureKind.TIMEOUT),
            GenerationFailureCategory.TIMEOUT,
            id="timeout",
        ),
        pytest.param(
            LLMProviderError("u", retryable=True, kind=ProviderFailureKind.UNREACHABLE),
            GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR,
            id="unreachable",
        ),
        pytest.param(
            LLMProviderError(
                "r", retryable=True, kind=ProviderFailureKind.RATE_LIMITED, status_code=429
            ),
            GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR,
            id="rate limited",
        ),
        pytest.param(
            LLMProviderError("a", kind=ProviderFailureKind.HTTP_STATUS, status_code=401),
            GenerationFailureCategory.PROVIDER_STATUS,
            id="refused",
        ),
        pytest.param(
            LLMProviderError("?"), GenerationFailureCategory.UNCLASSIFIED, id="unspecified"
        ),
    ],
)
def test_a_transport_failure_is_not_regenerated(
    raised: LLMProviderError, category: GenerationFailureCategory
):
    """Timeouts, outages and rate limits belong to the adapter's own budget."""
    error, llm = _failure(ScriptedReply(error=raised), GOOD)

    assert llm.call_count == 1
    assert error.failure is not None
    assert error.failure.category is category
    assert error.failure.attempts == 1


def test_the_transport_attempts_of_both_generations_are_counted_together():
    first = LLMProviderError("m", kind=ProviderFailureKind.MALFORMED_RESPONSE)
    second = LLMProviderError("t", retryable=True, kind=ProviderFailureKind.TIMEOUT)
    second.attempts = 3

    error, _ = _failure(ScriptedReply(error=first), ScriptedReply(error=second))

    assert error.failure is not None
    assert error.failure.category is GenerationFailureCategory.TIMEOUT
    assert error.failure.attempts == 4


# --- a regenerated reply is held to every check ------------------------------------


def test_a_regenerated_reply_with_an_invented_label_is_still_refused():
    answer, llm = _answer(TRUNCATED, grounded("Confident [S9].", "S9"))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert answer.citations == ()
    assert answer.unknown_labels == ("S9",)
    assert answer.generation_attempts == 2
    assert llm.check_count == 0


@pytest.mark.parametrize("support", ["inferred", "none", None])
def test_a_regenerated_reply_that_is_not_stated_is_still_refused(support: str | None):
    second = ScriptedReply(answer="Probably FastAPI [S1].", sources=("S1",), support=support)

    answer, llm = _answer(TRUNCATED, second)

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.citations == ()
    assert llm.call_count == 2 and llm.check_count == 0


def test_a_regenerated_reply_still_has_to_pass_the_grounding_check():
    answer, llm = _answer(TRUNCATED, GOOD, check=checked("not_supported"))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert answer.citations == ()
    assert answer.support is SupportVerdict.STATED
    assert answer.grounding_check is GroundingVerdict.NOT_SUPPORTED
    assert llm.call_count == 2 and llm.check_count == 1


# --- the paths this did not touch ---------------------------------------------------


def test_a_refusal_is_one_generation_and_no_check():
    answer, llm = _answer(ScriptedReply(answer="", sources=(), support="none"))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.generation_attempts == 1
    assert llm.call_count == 1 and llm.check_count == 0


def test_no_model_is_asked_when_nothing_was_retrieved():
    llm = ScriptedLLMProvider(TRUNCATED)
    service, _ = build_service([], llm=llm)

    answer = run(service.answer(QUESTION))

    assert answer.outcome is AnswerOutcome.NO_KNOWLEDGE
    assert answer.generation_attempts == 0
    assert llm.call_count == 0


@pytest.mark.parametrize(
    "check",
    [
        pytest.param(
            ScriptedReply(raw_text='{"verdict": "supp', finish_reason="length"), id="truncated"
        ),
        pytest.param(ScriptedReply(raw_text=""), id="empty"),
    ],
)
def test_the_grounding_check_is_asked_once_whatever_it_replies(check: ScriptedReply):
    """A check that gives no verdict is a technical failure, and is never asked
    again — not even when it was cut off at the limit."""
    error, llm = _failure(GOOD, check=check)

    assert llm.call_count == 1 and llm.check_count == 1
    assert error.failure is not None
    assert error.failure.step is ProviderCallType.GROUNDING_CHECK


def test_an_unreachable_grounding_check_is_still_an_outage_after_one_request():
    llm = ScriptedLLMProvider(
        GOOD,
        grounding_check=ScriptedReply(
            error=LLMProviderError("m", kind=ProviderFailureKind.MALFORMED_RESPONSE)
        ),
    )
    service, _ = build_service(llm=llm)

    with pytest.raises(GenerationUnavailableError):
        run(service.answer(QUESTION))

    assert llm.call_count == 1 and llm.check_count == 1


# --- through the real Workers AI adapter --------------------------------------------

FAKE_TOKEN = "test-token-not-a-real-credential"  # noqa: S105 - a fixture value
FAKE_ACCOUNT = "test-account-id"

_GOOD_CONTENT = json.dumps(
    {
        "answer": "The service uses FastAPI.",
        "sources": ["S1"],
        "support": "stated",
        "verdict": "supported",
    }
)


def _completion(content: Any, finish_reason: str = "stop", tokens: int = 40) -> dict[str, Any]:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 900, "completion_tokens": tokens},
    }


def _workers_ai(*results: dict[str, Any]) -> tuple[WorkersAIChatProvider, list[int]]:
    """The real adapter over an in-process transport that serves *results* in order."""
    served: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        index = min(len(served), len(results) - 1)
        served.append(index)
        return httpx.Response(200, json={"success": True, "errors": [], "result": results[index]})

    provider = WorkersAIChatProvider(
        account_id=FAKE_ACCOUNT,
        api_token=FAKE_TOKEN,
        client=httpx.AsyncClient(
            base_url="https://api.cloudflare.test/client/v4",
            transport=httpx.MockTransport(handler),
        ),
    )
    return provider, served


def test_an_ordinary_workers_ai_answer_costs_what_it_did_before():
    provider, served = _workers_ai(_completion(_GOOD_CONTENT))
    service, _, _ = build_pipeline(llm=provider)

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "The service uses FastAPI."
    assert answer.generation_attempts == 1
    assert len(served) == 2, "one generation and one grounding check"


def test_a_workers_ai_reply_cut_off_at_the_limit_is_generated_again():
    provider, served = _workers_ai(
        _completion('{"answer": "The service uses', finish_reason="length", tokens=800),
        _completion(_GOOD_CONTENT),
    )
    service, _, _ = build_pipeline(llm=provider)

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "The service uses FastAPI."
    assert answer.citations
    assert answer.generation_attempts == 2
    assert len(served) == 3, "two generations and one grounding check"


def test_a_workers_ai_response_with_no_text_is_classified_and_asked_for_again():
    """What the adapter calls malformed: a completion whose message has no content."""
    provider, served = _workers_ai(_completion(None), _completion(_GOOD_CONTENT))
    service, _, _ = build_pipeline(llm=provider)

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 2
    assert len(served) == 3


def test_a_workers_ai_reply_that_stays_cut_off_is_an_outage_after_two_requests():
    provider, served = _workers_ai(
        _completion('{"answer": "The service uses', finish_reason="length", tokens=800)
    )
    service, _, _ = build_pipeline(llm=provider)

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer("Which HTTP framework is used?"))

    assert len(served) == 2
    failure = caught.value.failure
    assert failure is not None
    assert failure.detail == "reply_not_json"
    assert failure.finish_reason == "length"
    assert failure.output_tokens == 800
    assert failure.attempts == 2
