"""One record per provider call: what it measured, and what it can never hold.

The records exist so that an ordinary call has a baseline. These tests pin
three things: every call the service makes leaves exactly one record and no
call is invented; each record carries the call's facts as the provider
reported them, with nothing estimated; and no record, log line or export ever
holds a prompt, a passage, a question or a reply.

Most cases run through the real Workers AI adapter over an in-process
transport, so ``usage`` and ``finish_reason`` are read the way production reads
them. No request leaves the process.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.evaluation import export_e2e, run_e2e_evaluation
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.conversation import ConversationRole, ConversationTurn
from portfolio_rag.rag.errors import GenerationUnavailableError
from portfolio_rag.rag.service import AnswerOutcome
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType
from portfolio_rag.rag.verification import GroundingVerdict
from tests.doubles import ScriptedLLMProvider, ScriptedReply, checked, grounded
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run
from tests.unit.test_answer_service import QUESTION, build_service
from tests.workers_ai_transport import DEGENERATE, LIMIT, SUPPORTED, VALID, workers_ai

PIPELINE_QUESTION = "Which HTTP framework is used?"


def completion(
    content: Any, *, finish_reason: str | None = "stop", usage: dict[str, int] | None = None
) -> dict[str, Any]:
    """A Workers AI body whose optional parts can each be left out."""
    choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    result: dict[str, Any] = {"choices": [choice]}
    if usage is not None:
        result["usage"] = usage
    return {"success": True, "errors": [], "result": result}


USAGE = {"prompt_tokens": 2400, "completion_tokens": 37}
CHECK_USAGE = {"prompt_tokens": 700, "completion_tokens": 9}


def answer_with(*responses: Any):
    provider, seen = workers_ai(*responses)
    service, _, _ = build_pipeline(llm=provider)
    return run(service.answer(PIPELINE_QUESTION)), seen


def raised_with(*responses: Any) -> GenerationUnavailableError:
    provider, _ = workers_ai(*responses)
    service, _, _ = build_pipeline(llm=provider)
    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(PIPELINE_QUESTION))
    return caught.value


# --- 1, 2, 6, 8, 10, 11. a successful generation and check -----------------------------


def test_a_successful_answer_records_one_generation_and_one_check():
    answer, seen = answer_with(
        completion(VALID, usage=USAGE), completion(SUPPORTED, usage=CHECK_USAGE)
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    generation, check = answer.provider_calls
    assert len(seen) == 2, "one record per request, and no record without one"

    assert generation.call_type is ProviderCallType.GENERATION
    assert (generation.attempt, generation.is_regeneration) == (1, False)
    assert generation.result is CallResult.PARSED
    assert generation.failure is None
    assert generation.finish_reason == "stop"
    assert (generation.input_tokens, generation.output_tokens) == (2400, 37)
    assert generation.reply_characters == len(VALID)
    assert generation.reply_visible_characters == sum(1 for c in VALID if not c.isspace())
    assert generation.response_format is ResponseFormat.JSON_OBJECT
    assert generation.elapsed_seconds >= 0.0

    assert check.call_type is ProviderCallType.GROUNDING_CHECK
    assert (check.attempt, check.result) == (1, CallResult.PARSED)
    assert (check.input_tokens, check.output_tokens) == (700, 9)
    assert check.reply_characters == len(SUPPORTED)


def test_the_model_is_the_one_the_provider_names():
    answer, _ = answer_with(completion(VALID), completion(SUPPORTED))

    assert {call.model for call in answer.provider_calls} == {"@cf/openai/gpt-oss-120b"}


def test_a_not_supported_verdict_is_a_parsed_call():
    """A verdict the backend could read is parsed, whichever way it went."""
    answer, _ = answer_with(completion(VALID), completion(json.dumps({"verdict": "not_supported"})))

    assert answer.grounding_check is GroundingVerdict.NOT_SUPPORTED
    assert answer.provider_calls[-1].result is CallResult.PARSED


# --- 7, 9. what the provider did not report stays unreported ------------------------------


def test_missing_usage_is_recorded_as_missing_never_estimated():
    answer, _ = answer_with(completion(VALID, usage=None), completion(SUPPORTED, usage=None))

    for call in answer.provider_calls:
        assert call.input_tokens is None
        assert call.output_tokens is None
        assert not call.usage_reported


def test_a_missing_finish_reason_is_recorded_as_missing():
    answer, _ = answer_with(
        completion(VALID, finish_reason=None), completion(SUPPORTED, finish_reason=None)
    )

    assert [call.finish_reason for call in answer.provider_calls] == [None, None]


# --- 3, 4, 15. attempts and regeneration ------------------------------------------------------


def test_a_regenerated_answer_records_both_attempts_in_order():
    answer, seen = answer_with(
        completion(DEGENERATE, finish_reason="length", usage={**USAGE, "completion_tokens": LIMIT}),
        completion(VALID, usage=USAGE),
        completion(SUPPORTED, usage=CHECK_USAGE),
    )

    first, second, check = answer.provider_calls
    assert len(seen) == 3
    assert (first.attempt, first.result, first.is_regeneration) == (
        1,
        CallResult.UNUSABLE_REPLY,
        False,
    )
    assert first.failure is not None and first.failure.detail == "reply_not_json"
    assert (first.finish_reason, first.output_tokens) == ("length", LIMIT)
    assert (first.reply_characters, first.reply_visible_characters) == (LIMIT, 1)
    assert (second.attempt, second.result, second.is_regeneration) == (2, CallResult.PARSED, True)
    assert check.call_type is ProviderCallType.GROUNDING_CHECK
    assert answer.generation_attempts == 2
    assert answer.regeneration_cause == first.failure


def test_a_malformed_response_keeps_its_metadata_and_has_no_reply_facts():
    answer, _ = answer_with(
        completion(None, finish_reason="length", usage={**USAGE, "completion_tokens": LIMIT}),
        completion(VALID),
        completion(SUPPORTED),
    )

    first = answer.provider_calls[0]
    assert first.result is CallResult.PROVIDER_ERROR
    assert first.failure is not None and first.failure.detail == "malformed_response"
    assert first.reply_characters is None, "no reply arrived, which is not an empty one"
    assert first.reply_visible_characters is None
    # What the response did say about itself is kept.
    assert first.finish_reason == "length"
    assert (first.input_tokens, first.output_tokens) == (2400, LIMIT)
    assert answer.provider_calls[1].is_regeneration


def test_two_failed_attempts_travel_with_the_error():
    error = raised_with(completion(DEGENERATE, finish_reason="length"))

    first, second = error.provider_calls
    assert [call.attempt for call in error.provider_calls] == [1, 2]
    assert all(call.result is CallResult.UNUSABLE_REPLY for call in error.provider_calls)
    assert second.is_regeneration
    assert error.failure == second.failure, "the error names the call that ended the question"
    assert first.failure is not None and first.failure.attempts == 1


def test_a_transport_failure_is_one_provider_error_record():
    error = raised_with(httpx.Response(401))

    (call,) = error.provider_calls
    assert call.result is CallResult.PROVIDER_ERROR
    assert call.failure is not None
    assert call.failure.status_code == 401
    assert call.input_tokens is None and call.reply_characters is None


# --- 5. the grounding check ------------------------------------------------------------------


def test_an_unusable_check_records_the_rule_it_broke():
    error = raised_with(completion(VALID), completion('{"verdict": "maybe"}', usage=CHECK_USAGE))

    generation, check = error.provider_calls
    assert generation.result is CallResult.PARSED
    assert check.result is CallResult.UNUSABLE_REPLY
    assert check.failure is not None and check.failure.detail == "verdict_not_recognized"
    assert error.failure == check.failure, "the error names the call that ended the request"
    assert (check.input_tokens, check.output_tokens) == (700, 9)


def test_an_unreachable_check_keeps_the_generation_record_before_it():
    llm = ScriptedLLMProvider(
        grounded("It uses FastAPI.", "S1"),
        grounding_check=ScriptedReply(
            error=LLMProviderError(
                "unreachable", retryable=True, kind=ProviderFailureKind.UNREACHABLE
            )
        ),
    )
    service, _ = build_service(llm=llm)

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    generation, check = caught.value.provider_calls
    assert (generation.call_type, generation.result) == (
        ProviderCallType.GENERATION,
        CallResult.PARSED,
    )
    assert (check.call_type, check.result) == (
        ProviderCallType.GROUNDING_CHECK,
        CallResult.PROVIDER_ERROR,
    )


# --- 16, 18. records mirror calls, and change nothing ------------------------------------------


@pytest.mark.parametrize(
    "replies",
    [
        (grounded("It uses FastAPI.", "S1"),),
        (ScriptedReply(answer="", sources=(), support="none"),),
        (ScriptedReply(raw_text='{"answer": "cut', finish_reason="length"), grounded("A.", "S1")),
        (grounded("It uses PostgreSQL.", "S9"),),
    ],
    ids=["answered", "declined", "regenerated", "invented-label"],
)
def test_there_is_exactly_one_record_per_call_the_provider_received(replies: tuple[Any, ...]):
    llm = ScriptedLLMProvider(*replies)
    service, _ = build_service(llm=llm)

    answer = run(service.answer(QUESTION))

    assert len(answer.provider_calls) == llm.call_count + llm.check_count
    assert answer.generation_attempts == llm.call_count


def test_no_knowledge_makes_no_call_and_has_no_record():
    llm = ScriptedLLMProvider()
    service, _ = build_service([], llm=llm)

    answer = run(service.answer(QUESTION))

    assert answer.outcome is AnswerOutcome.NO_KNOWLEDGE
    assert answer.provider_calls == ()
    assert llm.call_count == 0


def test_the_check_verdict_and_the_outcome_are_what_they_were():
    for check, outcome in (
        (checked("supported"), AnswerOutcome.ANSWERED),
        (checked("not_supported"), AnswerOutcome.NOT_GROUNDED),
    ):
        llm = ScriptedLLMProvider(grounded("It uses FastAPI.", "S1"), grounding_check=check)
        service, _ = build_service(llm=llm)
        assert run(service.answer(QUESTION)).outcome is outcome


def test_a_check_that_gives_no_verdict_ends_as_a_technical_failure():
    llm = ScriptedLLMProvider(
        grounded("It uses FastAPI.", "S1"), grounding_check=checked("perhaps")
    )
    service, _ = build_service(llm=llm)

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    assert [call.call_type for call in caught.value.provider_calls] == [
        ProviderCallType.GENERATION,
        ProviderCallType.GROUNDING_CHECK,
    ]


# --- 13, 17. no content anywhere -------------------------------------------------------------

_MARKED_QUESTION = "Which HTTP framework does the service use, MARKER-QUESTION?"
_MARKED_ANSWER = "It uses FastAPI MARKER-ANSWER."
_MARKED_TURN = "MARKER-CONVERSATION earlier turn"


def _contents() -> tuple[str, ...]:
    return (
        "MARKER-QUESTION",
        "MARKER-ANSWER",
        "MARKER-CONVERSATION",
        "FastAPI and runs under Uvicorn",  # a passage of the retrieval fixture
        "You are a knowledge assistant",  # the instructions
    )


def test_no_record_or_log_line_holds_content(caplog: pytest.LogCaptureFixture):
    llm = ScriptedLLMProvider(
        ScriptedReply(raw_text='{"answer": "MARKER-ANSWER cut', finish_reason="length"),
        grounded(_MARKED_ANSWER, "S1"),
    )
    service, _ = build_service(llm=llm)
    turns = [ConversationTurn(ConversationRole.USER, _MARKED_TURN)]

    with caplog.at_level(logging.INFO, logger="portfolio_rag.rag.service"):
        answer = run(service.answer(_MARKED_QUESTION, turns))

    assert answer.outcome is AnswerOutcome.ANSWERED
    serialized = json.dumps([call.fields() for call in answer.provider_calls])
    call_lines = [record for record in caplog.records if record.getMessage() == "provider call"]
    assert len(call_lines) == len(answer.provider_calls) == 3
    logged = json.dumps([record.__dict__ for record in caplog.records], default=str)
    for content in _contents():
        assert content not in serialized
        assert content not in logged


def test_a_record_exports_exactly_its_named_metadata():
    """A closed set of keys: adding one is a decision, never an accident."""
    record = ProviderCallRecord(
        call_type=ProviderCallType.GENERATION,
        attempt=1,
        model="m",
        response_format=ResponseFormat.JSON_OBJECT,
        elapsed_seconds=0.0,
        result=CallResult.PARSED,
    )

    assert set(record.fields()) == {
        "call_type",
        "attempt",
        "regeneration",
        "recovery",
        "model",
        "response_format",
        "elapsed_seconds",
        "result",
        "max_output_tokens",
        "failure_category",
        "failure_detail",
        "provider_error_code",
        "terminal_status_code",
        "http_attempts",
        "transport_attempts",
        "pacing_attempts",
        "retry_after_seconds",
        "finish_reason",
        "input_tokens",
        "output_tokens",
        "reply_characters",
        "reply_visible_characters",
    }


def test_a_conversation_changes_no_record_but_the_tokens_it_cost():
    llm_single = ScriptedLLMProvider(grounded("It uses FastAPI.", "S1"))
    llm_turns = ScriptedLLMProvider(grounded("It uses FastAPI.", "S1"))
    single, _ = build_service(llm=llm_single)
    with_turns, _ = build_service(llm=llm_turns)

    plain = run(single.answer(QUESTION))
    followed = run(with_turns.answer(QUESTION, [ConversationTurn(ConversationRole.USER, "Hi.")]))

    def shape(call: ProviderCallRecord) -> tuple[Any, ...]:
        return (call.call_type, call.attempt, call.result, call.reply_characters)

    assert [shape(c) for c in plain.provider_calls] == [shape(c) for c in followed.provider_calls]


# --- 14, 15. what the evaluation makes of them ---------------------------------------------------


def _exported(*replies: ScriptedReply, check: ScriptedReply | None = None) -> dict[str, Any]:
    dataset = dataset_of(ANSWERABLE)
    llm = ScriptedLLMProvider(*replies, grounding_check=check)
    return export_e2e(run(run_e2e_evaluation(dataset, e2e_service(llm))), run_metadata(dataset))


def test_the_export_lists_each_question_s_calls_in_order():
    label = label_of("stack", ANSWERABLE)
    payload = _exported(
        ScriptedReply(raw_text="{" + " " * 99, finish_reason="length"),
        grounded(f"FastAPI [{label}].", label),
    )

    (question,) = payload["questions"]
    assert [(c["call_type"], c["attempt"], c["result"]) for c in question["provider_calls"]] == [
        ("generation", 1, "unusable_reply"),
        ("generation", 2, "parsed"),
        ("grounding_check", 1, "parsed"),
    ]
    assert [c["regeneration"] for c in question["provider_calls"]] == [False, True, False]


def test_the_aggregate_separates_generation_from_grounding():
    label = label_of("stack", ANSWERABLE)
    payload = _exported(
        ScriptedReply(raw_text='{"answer": "cut', finish_reason="length"),
        grounded(f"FastAPI [{label}].", label),
    )

    calls = payload["metrics"]["provider_calls"]
    assert calls["generation"]["calls"] == 2
    assert calls["generation"]["regenerations"] == 1
    assert calls["generation"]["results"] == {
        "parsed": 1,
        "unusable_reply": 1,
        "provider_error": 0,
    }
    assert calls["generation"]["finish_reasons"] == {"length": 1, "stop": 1}
    assert calls["grounding_check"]["calls"] == 1
    assert calls["grounding_check"]["regenerations"] == 0
    assert calls["total"]["calls"] == 3
    # The scripted provider reports no usage: counted as unreported, summed as nothing.
    assert calls["total"]["usage_reported"] == 0
    assert calls["total"]["input_tokens"] is None, "unknown, not zero"


def test_the_aggregate_sums_only_reported_usage():
    provider, _ = workers_ai(completion(VALID, usage=USAGE), completion(SUPPORTED, usage=None))
    service, _, _ = build_pipeline(llm=provider)
    dataset = dataset_of(ANSWERABLE)
    payload = export_e2e(run(run_e2e_evaluation(dataset, service)), run_metadata(dataset))

    calls = payload["metrics"]["provider_calls"]
    assert (calls["generation"]["usage_reported"], calls["generation"]["input_tokens"]) == (1, 2400)
    assert calls["generation"]["output_tokens"] == 37
    assert calls["grounding_check"]["usage_reported"] == 0
    assert calls["total"]["usage_reported"] == 1


def test_an_errored_question_keeps_its_calls_in_the_export():
    payload = _exported(ScriptedReply(raw_text='{"answer": "cut', finish_reason="length"))

    (question,) = payload["questions"]
    assert question["outcome"] == "error"
    assert [c["attempt"] for c in question["provider_calls"]] == [1, 2]
    assert question["generation_attempts"] == 2
    assert payload["metrics"]["provider_calls"]["generation"]["calls"] == 2
