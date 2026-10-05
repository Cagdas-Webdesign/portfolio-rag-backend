"""The machine-readable contract, against the failure shapes production produced.

Every ``finish_reason=length`` failure recorded in ``evaluation/results/`` has
the same signature: ``output_tokens == reply_characters == 800``, the output
limit, with ``reply_not_json``. Readable JSON prose runs several characters per
token, and the same runs answered other questions with replies of up to 1419
characters under the same limit. A reply of exactly 800 characters in exactly
800 tokens, three times, was therefore not a long answer cut off — it was a
reply of single-character tokens. The fixture below is the simplest reply with
that signature: an opened object followed by blanks.

What is asserted is that the pipeline's answer to that shape does not depend on
what produced it: one more sample, never a third, a strict parser either way,
nothing published from it — and enough recorded, without a character of the
reply, to tell it apart from a long answer the next time it happens.

Everything runs through the real Workers AI adapter over an in-process
transport. No request leaves the process.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.evaluation import export_e2e, render_summary, run_e2e_evaluation
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.context import GroundedContext
from portfolio_rag.rag.errors import (
    GenerationFailureCategory,
    GenerationUnavailableError,
    ProviderCallType,
    visible_characters,
)
from portfolio_rag.rag.generation import parse_generation
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.prompt import SYSTEM_INSTRUCTIONS, build_generation_request
from portfolio_rag.rag.service import MAX_GENERATION_ATTEMPTS, AnswerOutcome
from portfolio_rag.rag.tokens import estimate_tokens
from portfolio_rag.rag.verification import (
    GROUNDING_CHECK_INSTRUCTIONS,
    GroundingVerdict,
    read_grounding_check,
)
from tests.doubles import ScriptedLLMProvider, ScriptedReply, make_chunk
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run
from tests.unit.test_answer_service import build_service
from tests.workers_ai_transport import DEGENERATE, LIMIT, SUPPORTED, VALID, Served, workers_ai

QUESTION = "Which HTTP framework is used?"

# --- the shapes production produced ------------------------------------------------

#: A reply that was genuinely long and cut off mid-object.
CUT_OFF_PROSE = '{"answer": "The service uses FastAPI for its HTTP API and'
DECLINED = json.dumps({"answer": "", "sources": [], "support": "none"})
NOT_SUPPORTED = json.dumps({"verdict": "not_supported"})

#: The longest grounded reply any recorded production run published, rebuilt
#: from `production-e2e-eval-v2-2026-10-01.json` (`multi-fullstack-role`) as the
#: object the model had to write: 1419 characters. Used as a size, not content.
LONGEST_OBSERVED_REPLY_CHARACTERS = 1419


def completion(content: Any, finish_reason: str = "stop", tokens: int = 40) -> dict[str, Any]:
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
            "usage": {"prompt_tokens": 2400, "completion_tokens": tokens},
        },
    }


def generations(seen: Served) -> int:
    return sum(1 for body in seen if body["messages"][0]["content"] == SYSTEM_INSTRUCTIONS)


def checks(seen: Served) -> int:
    return sum(1 for body in seen if body["messages"][0]["content"] == GROUNDING_CHECK_INSTRUCTIONS)


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


# --- the fixture is the production signature ------------------------------------------


def test_the_degenerate_fixture_has_the_recorded_signature():
    """chars == tokens == limit, and the parser calls it reply_not_json."""
    assert len(DEGENERATE) == LIMIT
    assert visible_characters(DEGENERATE) == 1
    with pytest.raises(GenerationUnavailableError) as caught:
        parse_generation(DEGENERATE)
    assert caught.value.failure is not None
    assert caught.value.failure.detail == "reply_not_json"


# --- A. a valid reply ---------------------------------------------------------------


def test_a_valid_first_reply_is_one_generation_and_one_check():
    answer, seen = answer_with(completion(VALID), completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 1
    assert answer.regeneration_cause is None
    assert (generations(seen), checks(seen)) == (1, 1)


# --- B, C. one degenerate sample ----------------------------------------------------


def test_a_degenerate_reply_at_the_limit_is_generated_again_and_the_second_is_used():
    answer, seen = answer_with(
        completion(DEGENERATE, "length", LIMIT), completion(VALID), completion(SUPPORTED)
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "The service uses FastAPI [1]."
    assert answer.generation_attempts == 2
    assert (generations(seen), checks(seen)) == (2, 1)


def test_the_discarded_sample_is_recorded_by_its_size_not_its_text():
    answer, _ = answer_with(
        completion(DEGENERATE, "length", LIMIT), completion(VALID), completion(SUPPORTED)
    )

    cause = answer.regeneration_cause
    assert cause is not None
    assert cause.category is GenerationFailureCategory.OUTPUT_TRUNCATED
    assert cause.detail == "reply_not_json"
    assert cause.finish_reason == "length"
    assert (cause.reply_characters, cause.reply_visible_characters, cause.output_tokens) == (
        LIMIT,
        1,
        LIMIT,
    )
    assert cause.attempts == 1


def test_a_long_answer_cut_off_is_told_apart_from_a_degenerate_one():
    """Same detail, same routing — different visible size, which is the point."""
    answer, _ = answer_with(
        completion(CUT_OFF_PROSE, "length", LIMIT), completion(VALID), completion(SUPPORTED)
    )

    cause = answer.regeneration_cause
    assert cause is not None
    assert cause.detail == "reply_not_json"
    assert cause.reply_visible_characters == visible_characters(CUT_OFF_PROSE)
    assert cause.reply_visible_characters > 1


# --- D. the same degenerate sample twice -----------------------------------------------


def test_two_degenerate_samples_fail_closed_after_exactly_two_generations():
    """The production failure, end to end: attempts=2, length, reply_not_json."""
    error, seen = failure_with(completion(DEGENERATE, "length", LIMIT))

    assert generations(seen) == MAX_GENERATION_ATTEMPTS == 2
    assert checks(seen) == 0
    assert error.code.value == "GENERATION_UNAVAILABLE"
    failure = error.failure
    assert failure is not None
    assert failure.category is GenerationFailureCategory.OUTPUT_TRUNCATED
    assert failure.detail == "reply_not_json"
    assert failure.finish_reason == "length"
    assert failure.attempts == 2
    assert (failure.reply_characters, failure.reply_visible_characters, failure.output_tokens) == (
        LIMIT,
        1,
        LIMIT,
    )


def test_a_second_bad_sample_of_another_kind_still_ends_after_two():
    error, seen = failure_with(
        completion(DEGENERATE, "length", LIMIT), completion(CUT_OFF_PROSE, "length", LIMIT)
    )

    assert generations(seen) == 2
    assert error.failure is not None and error.failure.attempts == 2


# --- E. malformed responses -----------------------------------------------------------


@pytest.mark.parametrize(
    "malformed",
    [
        completion(None),
        {"success": False, "errors": [{"code": 1}], "result": None},
        {"success": True, "result": {"choices": []}},
        httpx.Response(200, content=b"<html>not json</html>"),
    ],
    ids=["no-content", "success-false", "no-choices", "body-not-json"],
)
def test_a_malformed_response_is_asked_for_once_more(malformed: Any):
    answer, seen = answer_with(malformed, completion(VALID), completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 2
    assert answer.regeneration_cause is not None
    assert answer.regeneration_cause.detail == "malformed_response"
    assert generations(seen) == 2


def test_a_provider_that_stays_malformed_fails_closed_after_two():
    error, seen = failure_with(completion(None))

    assert generations(seen) == 2
    assert error.failure is not None
    assert error.failure.detail == "malformed_response"


# --- F, G. the grounding check ----------------------------------------------------------


def test_a_supported_verdict_publishes_and_a_not_supported_one_refuses():
    published, _ = answer_with(completion(VALID), completion(SUPPORTED))
    refused, _ = answer_with(completion(VALID), completion(NOT_SUPPORTED))

    assert published.grounding_check is GroundingVerdict.SUPPORTED
    assert published.citations
    assert refused.grounding_check is GroundingVerdict.NOT_SUPPORTED
    assert refused.outcome is AnswerOutcome.NOT_GROUNDED
    assert refused.citations == ()


@pytest.mark.parametrize(
    ("reply", "finish", "rule"),
    [
        ("{" + " " * (LIMIT - 1), "length", "reply_not_json"),
        ("", "stop", "reply_empty"),
        ("The answer is supported.", "stop", "reply_not_json"),
        ('["supported"]', "stop", "reply_not_an_object"),
        ('{"result": "supported"}', "stop", "verdict_missing"),
        ('{"verdict": "mostly"}', "stop", "verdict_not_recognized"),
        ('{"verdict": true}', "stop", "verdict_not_recognized"),
    ],
)
def test_an_unusable_check_is_a_technical_failure_that_says_which_rule_it_broke(
    reply: str, finish: str, rule: str
):
    tokens = LIMIT if finish == "length" else 12
    error, seen = failure_with(completion(VALID), completion(reply, finish, tokens))

    # Asked once more only when it stopped at the output limit; never a third time.
    assert checks(seen) == (2 if finish == "length" else 1)
    assert "FastAPI" not in str(error)
    failure = error.failure
    assert failure is not None
    assert failure.step is ProviderCallType.GROUNDING_CHECK
    assert failure.category is (
        GenerationFailureCategory.OUTPUT_TRUNCATED
        if finish == "length"
        else GenerationFailureCategory.UNPARSEABLE_OUTPUT
    )
    assert failure.detail == rule
    assert error.provider_calls[-1].failure == failure
    assert failure.finish_reason == finish
    assert failure.reply_characters == len(reply)
    assert failure.reply_visible_characters == visible_characters(reply)


def test_reading_a_check_never_changes_its_verdict():
    for reply in (SUPPORTED, NOT_SUPPORTED, "", "{", '{"verdict": "Supported "}'):
        verdict, rule = read_grounding_check(reply)
        assert (rule is None) == (verdict is not GroundingVerdict.UNUSABLE)


# --- H to K. transport failures are never regenerated -----------------------------------------


@pytest.mark.parametrize(
    ("response", "transport_attempts", "category"),
    [
        (httpx.Response(429), 3, GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR),
        (httpx.TimeoutException("slow"), 3, GenerationFailureCategory.TIMEOUT),
        (httpx.Response(401), 1, GenerationFailureCategory.PROVIDER_STATUS),
        (httpx.Response(403), 1, GenerationFailureCategory.PROVIDER_STATUS),
        (httpx.Response(400), 1, GenerationFailureCategory.PROVIDER_STATUS),
        (httpx.Response(503), 3, GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR),
        (httpx.ConnectError("down"), 3, GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR),
    ],
    ids=["429", "timeout", "401", "403", "400", "503", "unreachable"],
)
def test_a_transport_failure_is_reported_and_never_regenerated(
    response: Any, transport_attempts: int, category: GenerationFailureCategory
):
    error, seen = failure_with(response)

    # Only the adapter's own bounded transport retries: no second generation.
    assert len(seen) == transport_attempts
    assert error.failure is not None
    assert error.failure.category is category
    assert error.failure.attempts == transport_attempts


# --- L. what already worked still works the same way ----------------------------------------


def test_both_requests_ask_for_json_and_spend_the_same_bounded_budget():
    answer, seen = answer_with(completion(VALID), completion(SUPPORTED))

    assert answer.is_grounded
    for body in seen:
        assert body["response_format"] == {"type": ResponseFormat.JSON_OBJECT.value}
        assert body["max_tokens"] == LIMIT


def test_a_complete_reply_reported_as_length_is_still_accepted():
    """finish_reason alone never discards a reply the parser can read."""
    answer, seen = answer_with(completion(VALID, "length", LIMIT), completion(SUPPORTED))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 1
    assert generations(seen) == 1


def test_a_reply_finished_in_prose_is_not_regenerated():
    error, seen = failure_with(completion("Sure! The service uses FastAPI.", "stop", 9))

    assert generations(seen) == 1
    assert error.failure is not None
    assert error.failure.detail == "reply_not_json"
    assert error.failure.finish_reason == "stop"


def test_a_declined_reply_is_one_generation_and_no_check():
    answer, seen = answer_with(completion(DECLINED))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert (generations(seen), checks(seen)) == (1, 0)


# --- 14. size: what the budget has to hold ---------------------------------------------------

BROAD_SOURCES = [
    make_chunk(
        f"broad-{index}--0000",
        f"Project {index} uses technology {index}a, framework {index}b and tooling {index}c. "
        + "It is described in more detail in this passage. " * 32,
        document_id=f"broad-{index}",
        title=f"Broad Topic {index}",
        heading_path=("Technologies",),
    )
    for index in range(1, 6)
]
BROAD_QUESTION = "Which technologies, frameworks and tools are used across all projects?"


def broad_reply(answer_characters: int, *labels: str) -> str:
    sentence = "Project 1 uses technology 1a and framework 1b [S1]. "
    answer = (sentence * (answer_characters // len(sentence) + 1))[:answer_characters].strip()
    return json.dumps(
        {"answer": answer, "sources": list(labels), "support": "stated"}, ensure_ascii=False
    )


def test_the_longest_observed_reply_fits_the_output_budget_with_room():
    """By the project's own pessimistic estimate, not by a hoped-for ratio."""
    longest = broad_reply(LONGEST_OBSERVED_REPLY_CHARACTERS - 60, "S1", "S2", "S3", "S4", "S5")

    assert len(longest) >= LONGEST_OBSERVED_REPLY_CHARACTERS - 10
    assert estimate_tokens(longest) <= LIMIT * 0.6


def test_a_broad_multi_source_reply_parses_in_full():
    reply = broad_reply(LONGEST_OBSERVED_REPLY_CHARACTERS, "S1", "S2", "S3", "S4", "S5")

    draft = parse_generation(reply)

    assert draft.source_labels == ("S1", "S2", "S3", "S4", "S5")
    assert len(draft.answer) >= LONGEST_OBSERVED_REPLY_CHARACTERS - 1


def test_a_broad_question_over_long_passages_is_answered_in_one_generation():
    """A generic broad question: five full passages, five sources, one sample."""
    reply = broad_reply(1200, "S1", "S2", "S3", "S4", "S5")
    llm = ScriptedLLMProvider(_scripted(reply))
    service, _ = build_service(list[object](BROAD_SOURCES), llm=llm)

    answer = run(service.answer(BROAD_QUESTION))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.generation_attempts == 1
    assert len(answer.citations) == 5
    (request,) = llm.requests
    prompt_tokens = sum(estimate_tokens(message.content) for message in request.messages)
    policy = ContextPolicy()
    assert request.max_output_tokens == policy.output_reserve_tokens
    assert prompt_tokens + policy.output_reserve_tokens <= policy.max_prompt_tokens


def test_the_output_reserve_is_never_given_to_the_context():
    """Long passages are cut from the context, never from the answer's budget."""
    long_sources = [
        make_chunk(f"long-{index}--0000", "word " * 350, document_id=f"long-{index}")
        for index in range(12)
    ]
    llm = ScriptedLLMProvider()
    service, _ = build_service(list[object](long_sources), llm=llm)

    run(service.answer("word"))

    (request,) = llm.requests
    policy = ContextPolicy()
    assert request.max_output_tokens == policy.output_reserve_tokens
    prompt_tokens = sum(estimate_tokens(message.content) for message in request.messages)
    assert prompt_tokens + request.max_output_tokens <= policy.max_prompt_tokens


def test_the_output_contract_is_three_fields_and_asks_for_nothing_else():
    empty = GroundedContext(
        sources=(), text="", estimated_tokens=0, duplicates_removed=0, skipped_for_budget=0
    )
    request = build_generation_request(question="Q?", context=empty, max_output_tokens=LIMIT)
    shape = '{"answer": "<your answer>", "sources": ["S1", "S2"], "support": "stated"}'

    assert shape in request.messages[0].content
    assert "Reply with one JSON object and nothing else" in request.messages[0].content
    assert (
        '{"verdict": "supported"} or {"verdict": "not_supported"}' in GROUNDING_CHECK_INSTRUCTIONS
    )


# --- 15. what a run records ---------------------------------------------------------------


def test_the_export_and_summary_record_discarded_replies_by_size_only():
    label = label_of("stack", ANSWERABLE)
    degenerate = _scripted(DEGENERATE, finish_reason="length")
    good = _scripted(
        json.dumps({"answer": f"FastAPI [{label}].", "sources": [label], "support": "stated"})
    )
    dataset = dataset_of(ANSWERABLE)

    payload = export_e2e(
        run(run_e2e_evaluation(dataset, e2e_service(ScriptedLLMProvider(degenerate, good)))),
        run_metadata(dataset),
    )
    (question,) = payload["questions"]

    assert question["generation_attempts"] == 2
    first, second, check = question["provider_calls"]
    assert (first["call_type"], first["attempt"], first["result"]) == (
        "generation",
        1,
        "unusable_reply",
    )
    assert first["failure_detail"] == "reply_not_json"
    assert (first["reply_characters"], first["reply_visible_characters"]) == (LIMIT, 1)
    assert (second["attempt"], second["regeneration"], second["result"]) == (2, True, "parsed")
    assert (check["call_type"], check["result"]) == ("grounding_check", "parsed")
    assert payload["run"]["generation"]["response_format"] == "json_object"
    assert payload["run"]["generation"]["grounding_check_response_format"] == "json_object"

    summary = render_summary(payload)
    assert "### Calls that were not usable" in summary
    assert (
        f"| `{ANSWERABLE.id}` | generation | 1 | unusable_reply | reply_not_json | length "
        f"| {LIMIT} / 1 / — |" in summary
    )
    assert DEGENERATE not in json.dumps(payload)


def test_a_run_without_discarded_replies_has_no_such_section():
    label = label_of("stack", ANSWERABLE)
    good = _scripted(
        json.dumps({"answer": f"FastAPI [{label}].", "sources": [label], "support": "stated"})
    )
    dataset = dataset_of(ANSWERABLE)

    payload = export_e2e(
        run(run_e2e_evaluation(dataset, e2e_service(ScriptedLLMProvider(good)))),
        run_metadata(dataset),
    )

    assert "### Calls that were not usable" not in render_summary(payload)


# --- helpers --------------------------------------------------------------------------------


def _scripted(raw: str, *, finish_reason: str = "stop") -> ScriptedReply:
    return ScriptedReply(raw_text=raw, finish_reason=finish_reason)
