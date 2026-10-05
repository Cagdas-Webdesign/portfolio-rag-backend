"""One recovery per step, with room to finish, for a reply that hit the output limit.

The v1.1.1 release acceptance (`release-acceptance-v1.1.1-final.json`) ended
two questions as pipeline errors with the same signature: ``finish_reason``
``length`` and ``output_tokens`` equal to the 800-token cap. One generation
failed twice at that cap — once cut off, once with no text at all — and one
grounding check failed once and was never asked again. The model reasons
before it answers, and the reasoning counts against the cap.

What is pinned here: the first request of each step stays at the ordinary
cap; a recovery is asked only for a response that was not a completion or a
reply cut off at the limit; it gets the larger cap only when the provider said
it stopped at the limit; there is never a third request of a step; a verdict
— ``supported`` or ``not_supported`` — is never asked for again; and both
attempts are recorded, with one terminal failure when both fail.

Everything runs through the real Workers AI adapter over an in-process
transport. No request leaves the process.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.evaluation import export_e2e, run_e2e_evaluation
from portfolio_rag.evaluation.e2e import E2EFailure
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.rag.errors import (
    GenerationFailureCategory,
    GenerationUnavailableError,
    ProviderCallType,
)
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.prompt import SYSTEM_INSTRUCTIONS
from portfolio_rag.rag.service import (
    DEADLINE_EXCEEDED,
    MAX_GENERATION_ATTEMPTS,
    MAX_GROUNDING_CHECK_ATTEMPTS,
    AnswerOutcome,
    GroundedAnswerService,
)
from portfolio_rag.rag.telemetry import CallResult
from portfolio_rag.rag.verification import GROUNDING_CHECK_INSTRUCTIONS, GroundingVerdict
from tests.doubles import DelayedLLMProvider, ScriptedLLMProvider, ScriptedReply, grounded
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run
from tests.unit.test_answer_service import QUESTION as SCRIPTED_QUESTION
from tests.unit.test_answer_service import build_service
from tests.workers_ai_transport import LIMIT, SUPPORTED, VALID, Served, workers_ai

QUESTION = "Which HTTP framework is used?"
POLICY = ContextPolicy()
RECOVERY = POLICY.recovery_output_tokens
NOT_SUPPORTED = json.dumps({"verdict": "not_supported"})
C = GenerationFailureCategory


def completion(content: Any, finish_reason: str = "stop", tokens: int = 40) -> dict[str, Any]:
    """A Workers AI body. ``content=None`` is a response with no text at all —
    what a reasoning model returns when the cap ran out before it answered."""
    return {
        "success": True,
        "errors": [],
        "result": {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 1500, "completion_tokens": tokens},
        },
    }


#: The two production signatures, at the cap.
NO_TEXT_AT_LIMIT = completion(None, "length", LIMIT)
CUT_OFF_AT_LIMIT = completion('{"answer": "The service uses FastAPI for', "length", LIMIT)
CHECK_CUT_OFF_AT_LIMIT = completion('{"verdict": "supp', "length", LIMIT)


def caps(seen: Served, instructions: str) -> list[int]:
    """The ``max_tokens`` of every request of one step, in order."""
    return [body["max_tokens"] for body in seen if body["messages"][0]["content"] == instructions]


def generation_caps(seen: Served) -> list[int]:
    return caps(seen, SYSTEM_INSTRUCTIONS)


def check_caps(seen: Served) -> list[int]:
    return caps(seen, GROUNDING_CHECK_INSTRUCTIONS)


def answer_with(*responses: Any):
    provider, seen = workers_ai(*responses)
    service, _, _ = build_pipeline(llm=provider)
    return run(service.answer(QUESTION)), seen


def failure_with(*responses: Any):
    provider, seen = workers_ai(*responses)
    service, _, _ = build_pipeline(llm=provider)
    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))
    return caught.value, seen


def test_the_recovery_cap_is_larger_than_the_ordinary_one():
    assert RECOVERY > POLICY.output_reserve_tokens == LIMIT
    assert MAX_GENERATION_ATTEMPTS == MAX_GROUNDING_CHECK_ATTEMPTS == 2


def test_a_recovery_cap_below_the_reserve_is_refused():
    with pytest.raises(ValueError, match="recovery_output_tokens"):
        ContextPolicy(output_reserve_tokens=800, recovery_output_tokens=799)


# --- generation ------------------------------------------------------------------------


@pytest.mark.parametrize("first", [CUT_OFF_AT_LIMIT, NO_TEXT_AT_LIMIT], ids=["cut-off", "no-text"])
def test_1_a_generation_at_the_limit_is_asked_again_with_room_and_answers(first: Any):
    answer, seen = answer_with(first, completion(VALID), completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 2
    assert generation_caps(seen) == [LIMIT, RECOVERY]
    assert check_caps(seen) == [LIMIT], "the check is a first request of its own step"
    first_call, second_call, check = answer.provider_calls
    assert (first_call.attempt, first_call.max_output_tokens) == (1, LIMIT)
    assert first_call.finish_reason == "length" and first_call.output_tokens == LIMIT
    assert (second_call.attempt, second_call.max_output_tokens) == (2, RECOVERY)
    assert second_call.result is CallResult.PARSED and second_call.is_recovery
    assert check.max_output_tokens == LIMIT and not check.is_recovery


def test_2_a_generation_at_the_limit_twice_is_one_terminal_failure_after_two_requests():
    error, seen = failure_with(CUT_OFF_AT_LIMIT, NO_TEXT_AT_LIMIT, completion(SUPPORTED))

    assert generation_caps(seen) == [LIMIT, RECOVERY], "never a third request"
    assert check_caps(seen) == [], "nothing to check"
    failure = error.failure
    assert failure is not None
    assert failure.step is ProviderCallType.GENERATION
    assert failure.category is C.MALFORMED_RESPONSE
    assert failure.finish_reason == "length"
    assert failure.attempts == 2
    assert [c.max_output_tokens for c in error.provider_calls] == [LIMIT, RECOVERY]
    assert error.provider_calls[-1].failure == failure, "the root cause is the last call's"


def test_3_an_ordinary_answer_makes_no_recovery_and_spends_the_ordinary_cap_only():
    answer, seen = answer_with(completion(VALID), completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert (answer.generation_attempts, answer.grounding_check_attempts) == (1, 1)
    assert generation_caps(seen) == [LIMIT] and check_caps(seen) == [LIMIT]
    assert all(not call.is_recovery for call in answer.provider_calls)


def test_a_reply_that_finished_and_broke_the_contract_is_still_not_asked_again():
    error, seen = failure_with(completion("Sure, it is FastAPI.", "stop"))

    assert generation_caps(seen) == [LIMIT]
    assert error.failure is not None and error.failure.category is C.UNPARSEABLE_OUTPUT


# --- grounding check -------------------------------------------------------------------


@pytest.mark.parametrize(
    "first", [NO_TEXT_AT_LIMIT, CHECK_CUT_OFF_AT_LIMIT], ids=["no-text", "cut-off"]
)
def test_4_a_check_at_the_limit_is_asked_again_with_room_and_publishes(first: Any):
    answer, seen = answer_with(completion(VALID), first, completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.grounding_check is GroundingVerdict.SUPPORTED
    assert answer.citations
    assert answer.grounding_check_attempts == 2
    assert check_caps(seen) == [LIMIT, RECOVERY]
    assert generation_caps(seen) == [LIMIT], "the answer is not generated again"


def test_5_a_recovered_check_that_reads_not_supported_refuses_and_is_no_failure():
    answer, seen = answer_with(completion(VALID), NO_TEXT_AT_LIMIT, completion(NOT_SUPPORTED))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.grounding_check is GroundingVerdict.NOT_SUPPORTED
    assert answer.citations == ()
    assert check_caps(seen) == [LIMIT, RECOVERY]


def test_6_a_check_unusable_twice_is_one_terminal_failure_after_two_requests():
    error, seen = failure_with(completion(VALID), NO_TEXT_AT_LIMIT, CHECK_CUT_OFF_AT_LIMIT)

    assert check_caps(seen) == [LIMIT, RECOVERY], "never a third check"
    failure = error.failure
    assert failure is not None
    assert failure.step is ProviderCallType.GROUNDING_CHECK
    assert failure.category is C.OUTPUT_TRUNCATED
    assert failure.finish_reason == "length"
    assert failure.attempts == 2
    assert [c.call_type for c in error.provider_calls] == [
        ProviderCallType.GENERATION,
        ProviderCallType.GROUNDING_CHECK,
        ProviderCallType.GROUNDING_CHECK,
    ]
    assert error.provider_calls[-1].failure == failure
    assert "FastAPI" not in str(error), "nothing of the answer is published"


def test_a_malformed_check_that_did_not_stop_at_the_limit_is_asked_again_unchanged():
    answer, seen = answer_with(completion(VALID), completion(None, "stop"), completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert check_caps(seen) == [LIMIT, LIMIT]


def test_7_a_readable_not_supported_is_never_asked_again():
    answer, seen = answer_with(completion(VALID), completion(NOT_SUPPORTED))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert check_caps(seen) == [LIMIT]
    assert answer.grounding_check_attempts == 1


def test_a_check_that_finished_and_broke_its_contract_is_not_asked_again():
    error, seen = failure_with(completion(VALID), completion('{"verdict": "mostly"}'))

    assert check_caps(seen) == [LIMIT]
    assert error.failure is not None and error.failure.category is C.UNPARSEABLE_OUTPUT


@pytest.mark.parametrize("status", [401, 403])
def test_8_a_refused_check_is_not_recovered(status: int):
    error, seen = failure_with(completion(VALID), httpx.Response(status))

    assert check_caps(seen) == [LIMIT], "no transport retry and no recovery"
    assert error.failure is not None
    assert error.failure.category is C.PROVIDER_STATUS
    assert error.failure.status_code == status


def test_9_a_check_timeout_keeps_its_transport_retries_and_gets_no_recovery():
    error, seen = failure_with(completion(VALID), httpx.TimeoutException("slow"))

    assert len(check_caps(seen)) == 3, "the adapter's bounded retries, and only those"
    assert error.failure is not None and error.failure.category is C.TIMEOUT
    assert len(error.provider_calls) == 2


def test_9_a_deadline_during_the_check_recovery_ends_the_request():
    no_text = ScriptedReply(
        error=LLMProviderError(
            "no text", kind=ProviderFailureKind.MALFORMED_RESPONSE, finish_reason="length"
        )
    )
    llm = DelayedLLMProvider(
        ScriptedLLMProvider(grounded("It uses FastAPI.", "S1"), grounding_check=no_text),
        delays=(0.0, 0.0, 5.0),
    )
    base, _ = build_service(llm=llm)
    service = GroundedAnswerService(
        retrieval=base._retrieval,
        llm=llm,
        context_policy=base.context_policy,
        deadline_seconds=0.05,
    )

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(SCRIPTED_QUESTION))

    failure = caught.value.failure
    assert failure is not None
    assert (failure.category, failure.detail) == (C.TIMEOUT, DEADLINE_EXCEEDED)
    assert failure.step is ProviderCallType.GROUNDING_CHECK
    assert failure.attempts == 2, "the recovery was under way"
    # The generation completed, the first check raised, the recovery was cancelled.
    assert (llm.started, llm.completed) == (3, 1)
    assert [c.attempt for c in caught.value.provider_calls] == [1, 1]


# --- telemetry and the evaluation --------------------------------------------------------


def _e2e(*responses: Any):
    provider, seen = workers_ai(*responses)
    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(provider)))
    return export_e2e(report, run_metadata(dataset_of(ANSWERABLE))), seen


def _grounded_body() -> dict[str, Any]:
    label = label_of("stack", ANSWERABLE)
    return completion(
        json.dumps({"answer": f"FastAPI [{label}].", "sources": [label], "support": "stated"})
    )


def test_10_a_recovered_check_is_an_answer_with_both_attempts_recorded():
    payload, _ = _e2e(_grounded_body(), NO_TEXT_AT_LIMIT, completion(SUPPORTED))

    (question,) = payload["questions"]
    assert question["outcome"] == "answered"
    assert question["failures"] == []
    assert question["grounding_check_attempts"] == 2
    checks = [c for c in question["provider_calls"] if c["call_type"] == "grounding_check"]
    assert [(c["attempt"], c["recovery"], c["max_output_tokens"]) for c in checks] == [
        (1, False, LIMIT),
        (2, True, RECOVERY),
    ]
    assert checks[0]["result"] == "provider_error"
    assert checks[0]["failure_category"] == "malformed_response"
    assert checks[0]["finish_reason"] == "length"
    assert checks[0]["output_tokens"] == LIMIT
    assert checks[1]["result"] == "parsed"
    assert payload["run"]["generation"]["recovery_output_tokens"] == RECOVERY
    assert payload["run"]["generation"]["grounding_check_attempt_limit"] == 2


def test_10_a_check_that_fails_twice_is_exactly_one_pipeline_error_with_its_cause():
    payload, _ = _e2e(_grounded_body(), NO_TEXT_AT_LIMIT, CHECK_CUT_OFF_AT_LIMIT)

    (question,) = payload["questions"]
    assert question["failures"] == [E2EFailure.PIPELINE_ERROR.value]
    assert question["grounding_check_attempts"] == 2
    assert question["error"]["failure_step"] == "grounding_check"
    assert question["error"]["failure_category"] == "output_truncated"
    assert question["error"]["finish_reason"] == "length"
    assert question["error"]["attempts"] == 2
    assert [c["max_output_tokens"] for c in question["provider_calls"]] == [LIMIT, LIMIT, RECOVERY]
    assert payload["gates"]["no_pipeline_errors"] is False
