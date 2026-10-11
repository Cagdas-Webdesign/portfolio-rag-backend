"""The grounding check on its own port: who is asked, and what still fails closed.

The generator and the checker are two scripted providers whose verdicts
disagree, so every test shows which one was actually consulted. The last group
runs the real Groq adapter behind an `httpx.MockTransport`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.llm import GroqChatProvider
from portfolio_rag.rag.errors import GenerationFailureCategory, GenerationUnavailableError
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswer, GroundedAnswerService
from portfolio_rag.rag.telemetry import ProviderCallType
from portfolio_rag.rag.verification import GROUNDING_CHECK_INSTRUCTIONS
from tests.doubles import ScriptedLLMProvider, checked, grounded
from tests.support import run
from tests.unit.test_answer_service import QUESTION, build_service

CHECKER_MODEL = "checker-model"


class _Checker(ScriptedLLMProvider):
    @property
    def model(self) -> str:
        return CHECKER_MODEL


def split_service(generator: ScriptedLLMProvider, checker: Any) -> GroundedAnswerService:
    base, _ = build_service(llm=generator)
    return GroundedAnswerService(retrieval=base._retrieval, llm=generator, grounding_llm=checker)


def generator_says(answer: str = "It uses FastAPI [S1].", *labels: str) -> ScriptedLLMProvider:
    # Its own check would refuse — so a published answer proves it was not asked.
    return ScriptedLLMProvider(
        grounded(answer, *(labels or ("S1",))), grounding_check=checked("not_supported")
    )


# --- routing ------------------------------------------------------------------------


def test_the_check_goes_to_the_grounding_port_and_only_there():
    generator, checker = generator_says(), _Checker(grounding_check=checked("supported"))

    answer = run(split_service(generator, checker).answer(QUESTION))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert (generator.call_count, generator.check_count) == (1, 0)
    assert (checker.call_count, checker.check_count) == (0, 1)


def test_the_checker_receives_the_unchanged_production_check():
    generator, checker = generator_says(), _Checker(grounding_check=checked("supported"))

    run(split_service(generator, checker).answer(QUESTION))

    (request,) = checker.check_requests
    assert request.messages[0].content == GROUNDING_CHECK_INSTRUCTIONS
    assert "FastAPI" in request.messages[1].content


def test_without_a_separate_port_the_generator_checks_as_before():
    service, llm = build_service(llm=generator_says())

    answer = run(service.answer(QUESTION))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED, "its own check said not_supported"
    assert (llm.call_count, llm.check_count) == (1, 1)


def test_every_call_record_names_the_model_that_made_it():
    generator, checker = generator_says(), _Checker(grounding_check=checked("supported"))

    answer = run(split_service(generator, checker).answer(QUESTION))

    models = {call.call_type: call.model for call in answer.provider_calls}
    assert models == {
        ProviderCallType.GENERATION: ScriptedLLMProvider.MODEL,
        ProviderCallType.GROUNDING_CHECK: CHECKER_MODEL,
    }


# --- the verdict contract, unchanged ---------------------------------------------------


def test_not_supported_from_the_checker_is_the_controlled_refusal():
    generator, checker = generator_says(), _Checker(grounding_check=checked("not_supported"))

    answer = run(split_service(generator, checker).answer(QUESTION))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.citations == ()


@pytest.mark.parametrize("reply", ["prose", '{"verdict": "probably"}', "", '{"verdict": true}'])
def test_anything_but_supported_publishes_nothing(reply: str):
    from tests.doubles import ScriptedReply

    generator = generator_says()
    checker = _Checker(grounding_check=ScriptedReply(raw_text=reply))

    with pytest.raises(GenerationUnavailableError):
        run(split_service(generator, checker).answer(QUESTION))


def test_citation_validation_still_runs_before_the_check():
    """An invented label is dropped before the checker sees anything."""
    generator = generator_says("It uses FastAPI [S1] [S9].", "S1", "S9")
    checker = _Checker(grounding_check=checked("supported"))

    answer = run(split_service(generator, checker).answer(QUESTION))

    assert answer.unknown_labels == ("S9",)
    (request,) = checker.check_requests
    assert "S9" not in request.messages[1].content.split("ANSWER")[0]


def test_an_answer_refused_before_the_check_never_reaches_the_checker():
    generator = ScriptedLLMProvider(grounded("It uses FastAPI.", "S99"))
    checker = _Checker(grounding_check=checked("supported"))

    answer = run(split_service(generator, checker).answer(QUESTION))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert checker.check_count == 0


# --- the real Groq adapter as checker -------------------------------------------------

Handler = Callable[[httpx.Request], httpx.Response]


def groq(*responses: Handler) -> tuple[GroqChatProvider, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)](request)

    async def no_sleep(_: float) -> None:
        return None

    client = httpx.AsyncClient(
        base_url="https://api.groq.test/openai/v1", transport=httpx.MockTransport(handler)
    )
    return GroqChatProvider(api_key="gsk_test", client=client, sleeper=no_sleep), seen


def says(content: Any, finish_reason: str = "stop") -> Handler:
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 600, "completion_tokens": 150},
    }
    return lambda _: httpx.Response(200, json=body)


def answer_with(checker: GroqChatProvider) -> GroundedAnswer:
    return run(split_service(generator_says(), checker).answer(QUESTION))


def test_groq_supported_publishes_the_answer_with_its_citations():
    checker, seen = groq(says(json.dumps({"verdict": "supported"})))

    answer = answer_with(checker)

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert [c.document_id for c in answer.citations] == ["backend"]
    assert len(seen) == 1
    check = next(
        c for c in answer.provider_calls if c.call_type is ProviderCallType.GROUNDING_CHECK
    )
    assert (check.model, check.input_tokens, check.output_tokens) == (
        "openai/gpt-oss-20b",
        600,
        150,
    )


def test_groq_not_supported_is_the_controlled_refusal():
    checker, _ = groq(says(json.dumps({"verdict": "not_supported"})))

    assert answer_with(checker).outcome is AnswerOutcome.NOT_GROUNDED


def test_groq_invalid_json_fails_closed_without_a_second_check():
    checker, seen = groq(says("I think it is supported."))

    with pytest.raises(GenerationUnavailableError) as caught:
        answer_with(checker)

    assert len(seen) == 1
    assert caught.value.failure is not None
    assert caught.value.failure.category is GenerationFailureCategory.UNPARSEABLE_OUTPUT


def test_groq_http_400_fails_closed_and_is_not_recovered():
    checker, seen = groq(lambda _: httpx.Response(400, json={"error": {"message": "x"}}))

    with pytest.raises(GenerationUnavailableError) as caught:
        answer_with(checker)

    assert len(seen) == 1
    assert caught.value.failure is not None
    assert caught.value.failure.category is GenerationFailureCategory.PROVIDER_STATUS


def test_a_regenerated_answer_whose_groq_check_fails_is_one_outage_and_nothing_published():
    """The observed sequence: generation regenerated, then the check failed."""
    from tests.doubles import ScriptedReply

    truncated = ScriptedReply(raw_text='{"answer": "It uses FastAPI and', finish_reason="length")
    generator = ScriptedLLMProvider(truncated, grounded("It uses FastAPI [S1].", "S1"))
    checker, seen = groq(lambda _: httpx.Response(400, json={"error": {"message": "x"}}))

    with pytest.raises(GenerationUnavailableError) as caught:
        run(split_service(generator, checker).answer(QUESTION))

    failure = caught.value.failure
    assert failure is not None
    assert failure.step is ProviderCallType.GROUNDING_CHECK
    assert failure.category is GenerationFailureCategory.PROVIDER_STATUS
    assert (generator.call_count, generator.check_count, len(seen)) == (2, 0, 1)
    assert [c.call_type for c in caught.value.provider_calls] == [
        ProviderCallType.GENERATION,
        ProviderCallType.GENERATION,
        ProviderCallType.GROUNDING_CHECK,
    ]


def test_groq_429_fails_closed_and_is_not_retried():
    checker, seen = groq(lambda _: httpx.Response(429, json={}))

    with pytest.raises(GenerationUnavailableError):
        answer_with(checker)

    assert len(seen) == 1


def test_groq_5xx_is_retried_once_beneath_the_port_then_fails_closed():
    checker, seen = groq(lambda _: httpx.Response(503, json={}))

    with pytest.raises(GenerationUnavailableError):
        answer_with(checker)

    assert len(seen) == 2


def test_groq_timeout_fails_closed():
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    checker, _ = groq(slow)

    with pytest.raises(GenerationUnavailableError) as caught:
        answer_with(checker)

    assert caught.value.failure is not None
    assert caught.value.failure.category is GenerationFailureCategory.TIMEOUT


def test_groq_empty_reply_is_recovered_once_and_never_a_third_time():
    checker, seen = groq(says("", finish_reason="length"))

    with pytest.raises(GenerationUnavailableError):
        answer_with(checker)

    assert len(seen) == 2, "one check, its one recovery"
    recovery = json.loads(seen[1].content)
    assert recovery["max_completion_tokens"] == 1500, "the existing larger cap, not a new one"


def test_groq_empty_reply_then_a_verdict_publishes():
    checker, seen = groq(says(""), says(json.dumps({"verdict": "supported"})))

    assert answer_with(checker).outcome is AnswerOutcome.ANSWERED
    assert len(seen) == 2
