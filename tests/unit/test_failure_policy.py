"""The failure policy: one row per category, and proof that the code obeys it.

Three layers of evidence. The table itself — complete, every row technical,
recovery only where a fresh sample is the remedy. The precedence that places a
failure in exactly one row. And fault injection through the real Workers AI
adapter over an in-process transport: for every row, what happens on the wire,
how often the answer is generated, what the request ends as, and which facts
survive. Then the request deadline, cancellation, and the evaluation's line
between a classified failure and a defect in this code.

No request leaves the process.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.composition import build_query_components
from portfolio_rag.core.config import Settings
from portfolio_rag.evaluation import export_e2e, render_summary, run_e2e_evaluation
from portfolio_rag.evaluation.e2e import FAILURE_CLASSES, GATES, E2EFailure, FailureClass
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.rag.errors import (
    GenerationFailure,
    GenerationFailureCategory,
    GenerationUnavailableError,
    ProviderCallType,
    RetrievalUnavailableError,
    provider_failure,
)
from portfolio_rag.rag.failure_policy import (
    FAILURE_POLICY,
    Gate,
    Terminal,
    TransportRetry,
    classify_unusable_reply,
    may_recover,
    policy_for,
)
from portfolio_rag.rag.service import (
    DEADLINE_EXCEEDED,
    MAX_GENERATION_ATTEMPTS,
    MAX_GROUNDING_CHECK_ATTEMPTS,
    AnswerOutcome,
    GroundedAnswerService,
)
from portfolio_rag.rag.telemetry import CallResult
from tests.doubles import (
    DelayedEmbeddingProvider,
    DelayedLLMProvider,
    ScriptedLLMProvider,
    ScriptedReply,
    grounded,
)
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run
from tests.unit.test_answer_service import QUESTION, build_service
from tests.workers_ai_transport import LIMIT, SUPPORTED, VALID, workers_ai

C = GenerationFailureCategory

# --- the table --------------------------------------------------------------------


def test_every_category_has_exactly_one_policy_row():
    assert set(FAILURE_POLICY) == set(GenerationFailureCategory)


def test_every_failure_is_a_technical_error_and_an_availability_finding():
    """One root cause, one terminal outcome, one gate — for every category."""
    for policy in FAILURE_POLICY.values():
        assert policy.terminal is Terminal.TECHNICAL_ERROR
        assert policy.gate is Gate.AVAILABILITY


def test_recovery_is_allowed_only_where_a_fresh_sample_is_the_remedy():
    recoverable = {category for category, policy in FAILURE_POLICY.items() if policy.recovery}
    assert recoverable == {C.MALFORMED_RESPONSE, C.OUTPUT_TRUNCATED}


def test_transport_retries_belong_to_the_transient_categories_only():
    bounded = {
        category
        for category, policy in FAILURE_POLICY.items()
        if policy.transport_retry is TransportRetry.BOUNDED
    }
    assert bounded == {C.TIMEOUT, C.RETRYABLE_PROVIDER_ERROR}


def test_both_steps_recover_under_the_same_rule():
    for category in GenerationFailureCategory:
        recovers = {
            may_recover(GenerationFailure(category=category, detail="x", step=step))
            for step in ProviderCallType
        }
        assert recovers == {policy_for(GenerationFailure(category, "x")).recovery}


def test_every_evaluation_failure_has_exactly_one_class():
    assert set(FAILURE_CLASSES) == set(E2EFailure)
    assert FAILURE_CLASSES[E2EFailure.PIPELINE_ERROR] is FailureClass.AVAILABILITY
    quality = {f for f, c in FAILURE_CLASSES.items() if c is FailureClass.QUALITY}
    assert quality == {E2EFailure.RETRIEVAL_MISS, E2EFailure.NOT_ANSWERED}
    for gate in ("citations_verified", "unanswerable_refused", "nothing_internal_published"):
        assert {FAILURE_CLASSES[f] for f in GATES[gate]} == {FailureClass.SAFETY}


# --- precedence ---------------------------------------------------------------------


def _rule(detail: str = "reply_not_json") -> GenerationFailure:
    return GenerationFailure(category=C.UNPARSEABLE_OUTPUT, detail=detail)


@pytest.mark.parametrize(
    ("finish_reason", "category"),
    [("length", C.OUTPUT_TRUNCATED), ("stop", C.UNPARSEABLE_OUTPUT), (None, C.UNPARSEABLE_OUTPUT)],
)
def test_a_rejected_reply_is_truncated_only_when_the_provider_says_so(
    finish_reason: str | None, category: GenerationFailureCategory
):
    classified = classify_unusable_reply(_rule(), finish_reason=finish_reason)

    assert classified.category is category
    assert classified.detail == "reply_not_json", "the parser's rule is kept"


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (LLMProviderError("t", retryable=True, kind=ProviderFailureKind.TIMEOUT), C.TIMEOUT),
        # A malformed response is placed as malformed even if marked retryable.
        (
            LLMProviderError("m", retryable=True, kind=ProviderFailureKind.MALFORMED_RESPONSE),
            C.MALFORMED_RESPONSE,
        ),
        (
            LLMProviderError("r", retryable=True, kind=ProviderFailureKind.RATE_LIMITED),
            C.RETRYABLE_PROVIDER_ERROR,
        ),
        (
            LLMProviderError("s", kind=ProviderFailureKind.HTTP_STATUS, status_code=401),
            C.PROVIDER_STATUS,
        ),
        (LLMProviderError("?"), C.UNCLASSIFIED),
    ],
)
def test_a_port_failure_lands_in_exactly_one_category(
    error: LLMProviderError, category: GenerationFailureCategory
):
    assert provider_failure(error).category is category


# --- fault injection through the real adapter ------------------------------------------


def body(content: Any, finish_reason: str = "stop", tokens: int = 40) -> dict[str, Any]:
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


def _fail(*responses: Any) -> tuple[GenerationUnavailableError, list[dict[str, Any]]]:
    provider, seen = workers_ai(*responses)
    service, _, _ = build_pipeline(llm=provider)
    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer("Which HTTP framework is used?"))
    return caught.value, seen


def _assert_policy_obeyed(
    error: GenerationUnavailableError, *, port_calls: int
) -> GenerationFailure:
    """What every technical failure owes: classified, preserved, recorded, bounded."""
    failure = error.failure
    assert failure is not None
    policy = policy_for(failure)
    assert policy.terminal is Terminal.TECHNICAL_ERROR and policy.gate is Gate.AVAILABILITY
    assert len(error.provider_calls) == port_calls
    assert error.provider_calls[-1].failure == failure or failure.detail == DEADLINE_EXCEEDED
    assert error.retrieval is not None, "what was retrieved travels with the failure"
    generations = [c for c in error.provider_calls if c.call_type is ProviderCallType.GENERATION]
    assert len(generations) <= MAX_GENERATION_ATTEMPTS
    checks = [c for c in error.provider_calls if c.call_type is ProviderCallType.GROUNDING_CHECK]
    assert len(checks) <= MAX_GROUNDING_CHECK_ATTEMPTS
    return failure


@pytest.mark.parametrize(
    ("response", "wire", "category", "status"),
    [
        (httpx.TimeoutException("slow"), 3, C.TIMEOUT, None),
        (httpx.ConnectError("down"), 3, C.RETRYABLE_PROVIDER_ERROR, None),
        (httpx.Response(503), 3, C.RETRYABLE_PROVIDER_ERROR, 503),
        (httpx.Response(429, headers={"Retry-After": "7"}), 3, C.RETRYABLE_PROVIDER_ERROR, 429),
        (httpx.Response(401), 1, C.PROVIDER_STATUS, 401),
        (httpx.Response(403), 1, C.PROVIDER_STATUS, 403),
        (httpx.Response(400), 1, C.PROVIDER_STATUS, 400),
    ],
    ids=["timeout", "connect", "5xx", "429", "401", "403", "400"],
)
def test_transport_failures_follow_their_row(
    response: Any, wire: int, category: GenerationFailureCategory, status: int | None
):
    error, seen = _fail(response)

    failure = _assert_policy_obeyed(error, port_calls=1)
    assert len(seen) == wire, "the adapter's bounded retries, and only those"
    assert failure.category is category
    assert policy_for(failure).transport_retry is (
        TransportRetry.BOUNDED if wire > 1 else TransportRetry.NONE
    )
    assert failure.status_code == status
    assert failure.attempts == wire
    assert failure.step is ProviderCallType.GENERATION
    if status == 429:
        assert failure.retry_after_seconds == 7.0


@pytest.mark.parametrize(
    ("reply", "finish", "category", "detail", "generations"),
    [
        (None, "stop", C.MALFORMED_RESPONSE, "malformed_response", 2),
        ("", "stop", C.UNPARSEABLE_OUTPUT, "reply_empty", 1),
        ("", "length", C.OUTPUT_TRUNCATED, "reply_empty", 2),
        ("Sure! It is FastAPI.", "stop", C.UNPARSEABLE_OUTPUT, "reply_not_json", 1),
        ('{"answer": "It uses', "length", C.OUTPUT_TRUNCATED, "reply_not_json", 2),
        (
            '{"answer": 5, "sources": [], "support": "stated"}',
            "stop",
            C.UNPARSEABLE_OUTPUT,
            "answer_not_text",
            1,
        ),
        ("[1, 2]", "stop", C.UNPARSEABLE_OUTPUT, "reply_not_an_object", 1),
    ],
    ids=[
        "malformed",
        "empty-stop",
        "empty-length",
        "prose-stop",
        "cut-length",
        "wrong-type",
        "not-object",
    ],
)
def test_generation_output_failures_follow_their_row(
    reply: Any, finish: str, category: GenerationFailureCategory, detail: str, generations: int
):
    error, seen = _fail(body(reply, finish, LIMIT if finish == "length" else 40))

    failure = _assert_policy_obeyed(error, port_calls=generations)
    assert len(seen) == generations, "recovery exactly where the row allows it, never a third"
    assert failure.category is category
    assert failure.detail == detail
    assert policy_for(failure).recovery is (generations == 2)
    if reply is None:
        assert failure.finish_reason == "stop", "a malformed response keeps what it reported"
    else:
        assert failure.finish_reason == finish
        assert failure.reply_characters == len(reply)


def test_a_complete_reply_reported_as_length_is_not_a_failure():
    provider, seen = workers_ai(body(VALID, "length", LIMIT), body(SUPPORTED))
    service, _, _ = build_pipeline(llm=provider)

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert len(seen) == 2, "one generation, one check"


@pytest.mark.parametrize(
    ("check", "rule"),
    [
        ("", "reply_empty"),
        ("It is supported.", "reply_not_json"),
        ('["supported"]', "reply_not_an_object"),
        ('{"result": "supported"}', "verdict_missing"),
        ('{"verdict": "mostly"}', "verdict_not_recognized"),
    ],
)
def test_an_unusable_grounding_check_follows_its_row(check: str, rule: str):
    error, seen = _fail(body(VALID), body(check))

    failure = _assert_policy_obeyed(error, port_calls=2)
    assert len(seen) == 2, "one generation, one check, nothing asked again"
    assert failure.step is ProviderCallType.GROUNDING_CHECK
    assert failure.category is C.UNPARSEABLE_OUTPUT
    assert failure.detail == rule
    assert error.provider_calls[0].result is CallResult.PARSED


def test_a_grounding_check_provider_failure_is_technical_and_keeps_the_generation():
    error, seen = _fail(body(VALID), httpx.Response(503))

    failure = _assert_policy_obeyed(error, port_calls=2)
    assert len(seen) == 1 + 3, "one generation, the check with its transport retries"
    assert failure.step is ProviderCallType.GROUNDING_CHECK
    assert failure.category is C.RETRYABLE_PROVIDER_ERROR


@pytest.mark.parametrize(
    ("check", "outcome"),
    [("supported", AnswerOutcome.ANSWERED), ("not_supported", AnswerOutcome.NOT_GROUNDED)],
)
def test_a_readable_verdict_is_an_outcome_not_a_failure(check: str, outcome: AnswerOutcome):
    provider, _ = workers_ai(body(VALID), body(f'{{"verdict": "{check}"}}'))
    service, _, _ = build_pipeline(llm=provider)

    assert run(service.answer("Which HTTP framework is used?")).outcome is outcome


# --- the provider boundary ---------------------------------------------------------------


class _RaisingAdapter:
    """Breaks the port contract by raising something the port does not define."""

    model = "raising-adapter"

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    async def generate(self, request: Any) -> Any:
        self.calls += 1
        raise self._error


def test_an_unexpected_exception_at_the_port_is_unclassified_and_chained():
    original = KeyError("choices")
    adapter = _RaisingAdapter(original)
    service, _ = build_service(llm=adapter)

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))

    failure = caught.value.failure
    assert failure is not None
    assert failure.category is C.UNCLASSIFIED
    assert failure.detail == "KeyError", "the class name, never the text"
    assert caught.value.__cause__ is original
    assert adapter.calls == 1, "no recovery for an unclassified failure"
    assert caught.value.provider_calls[0].result is CallResult.PROVIDER_ERROR
    assert "choices" not in caught.value.message


def test_a_defect_outside_the_port_call_is_not_normalised(monkeypatch: pytest.MonkeyPatch):
    """Only the one awaited provider call is a boundary; a bug elsewhere stays a bug."""
    service, _ = build_service(llm=ScriptedLLMProvider(grounded("It uses FastAPI.", "S1")))

    def broken(*_: Any, **__: Any) -> Any:
        raise TypeError("a defect in this code")

    monkeypatch.setattr("portfolio_rag.rag.service.resolve_citations", broken)
    with pytest.raises(TypeError):
        run(service.answer(QUESTION))


# --- the request deadline -----------------------------------------------------------------


def _with_deadline(llm: Any, seconds: float = 0.05) -> GroundedAnswerService:
    service, _ = build_service(llm=llm)
    return GroundedAnswerService(
        retrieval=service._retrieval,
        llm=llm,
        context_policy=service.context_policy,
        deadline_seconds=seconds,
    )


TRUNCATED = ScriptedReply(raw_text='{"answer": "cut', finish_reason="length")
GOOD = grounded("It uses FastAPI.", "S1")


def _deadline_failure(service: GroundedAnswerService) -> GenerationUnavailableError:
    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(QUESTION))
    failure = caught.value.failure
    assert failure is not None
    assert failure.category is C.TIMEOUT
    assert failure.detail == DEADLINE_EXCEEDED
    assert isinstance(caught.value.__cause__, TimeoutError)
    return caught.value


def test_a_deadline_during_the_first_generation_ends_it():
    llm = DelayedLLMProvider(ScriptedLLMProvider(GOOD), delays=(5.0,))

    error = _deadline_failure(_with_deadline(llm))

    assert error.failure is not None and error.failure.step is ProviderCallType.GENERATION
    assert (llm.started, llm.completed) == (1, 0)
    assert error.provider_calls == (), "the cancelled call delivered nothing to record"


def test_a_deadline_that_passes_before_the_second_generation_stops_it():
    # The first call spends the whole deadline without yielding, then returns a
    # truncated reply; the regeneration it earns never gets to run.
    llm = DelayedLLMProvider(ScriptedLLMProvider(TRUNCATED, GOOD), delays=(0.1, 0.0), blocking=True)

    error = _deadline_failure(_with_deadline(llm))

    assert llm.completed == 1, "no second generation completed after the deadline"
    assert [c.attempt for c in error.provider_calls] == [1]
    assert error.failure is not None and error.failure.attempts == 2


def test_a_deadline_during_the_second_generation_ends_it():
    llm = DelayedLLMProvider(ScriptedLLMProvider(TRUNCATED, GOOD), delays=(0.0, 5.0))

    error = _deadline_failure(_with_deadline(llm))

    assert (llm.started, llm.completed) == (2, 1)
    assert error.failure is not None and error.failure.step is ProviderCallType.GENERATION


def test_a_deadline_during_the_grounding_check_ends_it_and_publishes_nothing():
    llm = DelayedLLMProvider(ScriptedLLMProvider(GOOD), delays=(0.0, 5.0))

    error = _deadline_failure(_with_deadline(llm))

    assert error.failure is not None and error.failure.step is ProviderCallType.GROUNDING_CHECK
    assert [c.call_type for c in error.provider_calls] == [ProviderCallType.GENERATION]
    assert "FastAPI" not in str(error)


def test_a_deadline_during_retrieval_is_a_retrieval_outage():
    llm = ScriptedLLMProvider(GOOD)
    service = _with_deadline(llm)
    service._retrieval._embeddings = DelayedEmbeddingProvider(
        service._retrieval._embeddings, delay=5.0
    )

    with pytest.raises(RetrievalUnavailableError):
        run(service.answer(QUESTION))
    assert llm.call_count == 0


def test_no_deadline_leaves_a_slow_request_alone():
    llm = DelayedLLMProvider(ScriptedLLMProvider(GOOD), delays=(0.06,))
    service, _ = build_service(llm=llm)

    assert run(service.answer(QUESTION)).outcome is AnswerOutcome.ANSWERED


def test_cancellation_from_outside_is_never_swallowed_and_leaves_no_task():
    llm = DelayedLLMProvider(ScriptedLLMProvider(GOOD), delays=(5.0,))
    service = _with_deadline(llm, seconds=30.0)

    async def scenario() -> None:
        task = asyncio.create_task(service.answer(QUESTION))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == {asyncio.current_task()}

    run(scenario())


def test_a_non_positive_deadline_is_refused():
    with pytest.raises(ValueError):
        _with_deadline(ScriptedLLMProvider(), seconds=0)


def test_the_server_gets_the_configured_deadline(settings: Settings):
    components = build_query_components(
        settings.model_copy(update={"request_deadline_seconds": 42.0})
    )

    assert components.answers.deadline_seconds == 42.0
    assert build_query_components(settings).answers.deadline_seconds == 60.0


# --- the evaluation: classified failure vs. defect ---------------------------------------------


class _DefectOnSecondQuestion:
    """An answering service that works once, then hits a defect in this code."""

    def __init__(self, service: GroundedAnswerService) -> None:
        self._service = service
        self.asked = 0

    async def answer(self, message: str) -> Any:
        self.asked += 1
        if self.asked == 2:
            raise TypeError("internal defect")
        return await self._service.answer(message)


def _second_question() -> Any:
    return ANSWERABLE.model_copy(update={"id": "q-api-again"})


def test_a_classified_provider_failure_is_recorded_and_the_run_goes_on():
    llm = ScriptedLLMProvider(ScriptedReply(error=LLMProviderError("x", retryable=True)))
    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE, _second_question()), e2e_service(llm)))

    assert report.aborted is None
    assert [record.errored for record in report.records] == [True, True]


def test_an_internal_defect_aborts_the_run_and_keeps_what_was_measured():
    label = label_of("stack", ANSWERABLE)
    answers = _DefectOnSecondQuestion(
        e2e_service(ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label)))
    )
    third = ANSWERABLE.model_copy(update={"id": "q-never-asked"})
    dataset = dataset_of(ANSWERABLE, _second_question(), third)

    report = run(run_e2e_evaluation(dataset, answers))  # type: ignore[arg-type]

    assert answers.asked == 2, "nothing is asked after the defect"
    assert [record.question.id for record in report.records] == ["q-api"]
    assert report.aborted is not None
    assert (report.aborted.question_id, report.aborted.error_type) == ("q-api-again", "TypeError")
    assert not report.passed

    payload = export_e2e(report, run_metadata(dataset))
    assert payload["run"]["complete"] is False
    assert payload["run"]["aborted"] == {
        "reason": "internal_defect",
        "question_id": "q-api-again",
        "error_type": "TypeError",
    }
    assert payload["passed"] is False
    assert "internal defect" not in str(payload)
    assert "**Run aborted** at `q-api-again`" in render_summary(payload)
