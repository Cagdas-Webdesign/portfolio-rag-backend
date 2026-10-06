"""Checkpoint, resume and the provider canary of an acceptance run — offline.

The CLI drives these in `tests/integration/test_cli_resume.py`. Here each
piece is held on its own: a record survives the checkpoint unchanged, a
checkpoint survives a crash, a changed identity refuses a resume, the
validator sees through a manipulated execution record, a resumed run asks
nothing it already asked, the budget forecasts only what is left, and the
canary classifies every way a provider can say no.
"""

from __future__ import annotations

import copy
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.evaluation import E2EReport, export_e2e, run_e2e_evaluation
from portfolio_rag.evaluation.acceptance import (
    _leak_problems,
    validate_artifact,
    with_release_acceptance,
)
from portfolio_rag.evaluation.budget import (
    COST_PROFILES,
    BudgetPolicy,
    BudgetZone,
    LedgerEntry,
    LedgerReading,
    UsageHistory,
)
from portfolio_rag.evaluation.canary import (
    CANARY_MAX_OUTPUT_TOKENS,
    CanaryStatus,
    canary_neurons_bound,
    canary_request,
    run_canary,
)
from portfolio_rag.evaluation.checkpoint import (
    IDENTITY_FIELDS,
    AcceptanceCheckpoint,
    CheckpointError,
    identity_digest,
    identity_mismatches,
    record_from_checkpoint,
    record_to_checkpoint,
    run_identity,
    segment_problems,
    segment_usage,
    write_json_atomic,
)
from portfolio_rag.evaluation.e2e import AbortReason, E2ERecord, RunAbort
from portfolio_rag.evaluation.operations import Tier, acceptance_attempts, preflight
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse, TokenUsage
from tests.doubles import FailingLLMProvider, ScriptedLLMProvider, grounded
from tests.e2e_stack import (
    ANSWERABLE,
    CHUNKS,
    dataset_of,
    e2e_service,
    label_of,
    run_metadata,
)
from tests.support import run

PROFILE = COST_PROFILES[0]
COMMIT = "c0ffee" + "0" * 34


def _questions(count: int) -> list[Any]:
    return [ANSWERABLE.model_copy(update={"id": f"q-{index:02d}"}) for index in range(1, count + 1)]


def _service() -> Any:
    label = label_of("stack", ANSWERABLE)
    return e2e_service(ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label)))


def _records(count: int = 3) -> tuple[E2ERecord, ...]:
    report = run(run_e2e_evaluation(dataset_of(*_questions(count)), _service()))
    return report.records


def _run_fields(questions: Any) -> dict[str, Any]:
    from portfolio_rag.evaluation.e2e import e2e_run_fields

    return e2e_run_fields(run_metadata(dataset_of(*questions)))


def _checkpoint(tmp_path: Path, count: int = 4) -> AcceptanceCheckpoint:
    questions = _questions(count)
    identity = run_identity(_run_fields(questions), [q.id for q in questions])
    return AcceptanceCheckpoint(
        path=tmp_path / "run.checkpoint.json",
        logical_run_id="logical-1",
        started_at="2026-10-05T10:00:00+00:00",
        identity=identity,
    )


# --- records survive the checkpoint -------------------------------------------------


def test_a_record_survives_the_checkpoint_and_exports_byte_for_byte_the_same():
    records = _records()
    dataset = dataset_of(*(record.question for record in records))
    corpus = {chunk.id: chunk for chunk in CHUNKS}

    restored = tuple(
        record_from_checkpoint(
            json.loads(json.dumps(record_to_checkpoint(record))), record.question, corpus
        )
        for record in records
    )

    metadata = run_metadata(dataset)
    assert export_e2e(E2EReport(records=restored), metadata) == export_e2e(
        E2EReport(records=records), metadata
    )


def test_an_errored_record_with_its_failure_and_calls_survives_too():
    failing = FailingLLMProvider(
        LLMProviderError("x", retryable=True, kind=ProviderFailureKind.TIMEOUT)
    )
    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(failing)))
    (record,) = report.records
    assert record.errored

    data = json.loads(json.dumps(record_to_checkpoint(record)))
    restored = record_from_checkpoint(data, ANSWERABLE, {chunk.id: chunk for chunk in CHUNKS})

    assert restored == record


def test_a_checkpoint_holds_no_passage_text_and_no_credential(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path, 3)
    checkpoint.begin_segment(run_id="seg-1", started_at="2026-10-05T10:00:00+00:00")
    records = _records()
    checkpoint.progress(
        records,
        segment_completed=[r.question.id for r in records],
        usage=segment_usage(records, interrupted=0, neurons=lambda calls: 0.0),
        at="2026-10-05T10:01:00+00:00",
    )

    text = checkpoint.path.read_text(encoding="utf-8")
    assert all(chunk.content not in text for chunk in CHUNKS)
    # The publication check an artifact faces — forbidden keys (prompt,
    # messages, passage, content, token, account_id, …), credential shapes and
    # configured secrets — finds nothing in a checkpoint either.
    problems = _leak_problems(json.loads(text), ["placeholder-credential-1234"])
    assert problems == []


# --- atomic, verified persistence ---------------------------------------------------


def test_a_crash_while_writing_leaves_the_last_valid_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = _checkpoint(tmp_path)
    checkpoint.begin_segment(run_id="seg-1", started_at="t0")
    records = _records(2)
    checkpoint.progress(
        records[:1],
        segment_completed=["q-01"],
        usage=segment_usage(records[:1], interrupted=0, neurons=lambda calls: 0.0),
        at="t1",
    )

    def crash(*args: Any) -> None:
        raise OSError("disk gone mid-write")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        checkpoint.progress(
            records,
            segment_completed=["q-01", "q-02"],
            usage=segment_usage(records, interrupted=0, neurons=lambda calls: 0.0),
            at="t2",
        )
    monkeypatch.undo()

    survivor = AcceptanceCheckpoint.load(checkpoint.path)
    assert survivor.completed_ids == ["q-01"]
    assert not list(tmp_path.glob(".*.tmp")), "the half-written file is cleaned up"


def test_write_json_atomic_replaces_the_whole_file(tmp_path: Path):
    target = tmp_path / "x.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"b": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"b": 2}


def test_an_edited_checkpoint_is_refused(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    checkpoint.begin_segment(run_id="seg-1", started_at="t0")
    data = json.loads(checkpoint.path.read_text(encoding="utf-8"))
    data["identity"]["git_revision"] = "f" * 40
    checkpoint.path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CheckpointError, match="integrity"):
        AcceptanceCheckpoint.load(checkpoint.path)


def test_a_checkpoint_that_skips_a_question_is_refused(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path)
    checkpoint.begin_segment(run_id="seg-1", started_at="t0")
    records = _records(2)
    checkpoint.records = [record_to_checkpoint(records[1])]  # q-02 without q-01
    checkpoint.segments[-1]["completed_question_ids"] = ["q-02"]
    checkpoint.save(updated_at="t1")

    with pytest.raises(CheckpointError, match="first questions in order"):
        AcceptanceCheckpoint.load(checkpoint.path)


# --- identity -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("git_revision",), "d" * 40),
        (("git_tag",), "v9.9.9"),
        (("project_version",), "9.9.9"),
        (("release_source_identity",), "9" * 64),
        (("dataset", "sha256"), "9" * 64),
        (("suite_sha256",), "9" * 64),
        (("corpus", "sha256"), "9" * 64),
        (("embedding", "model"), "other-embed"),
        (("vector_store",), "cloudflare_vectorize"),
        (("retrieval_policy", "top_k"), 3),
        (("retrieval_policy", "min_similarity"), 0.9),
        (("generation", "provider"), "mistral"),
        (("generation", "model"), "other-model"),
        (("generation", "prompt_version"), "grounded-answer-v0"),
        (("generation", "grounding_check_version"), "grounding-check-v0"),
        (("generation", "response_format"), "text"),
        (("generation", "max_output_tokens"), 4096),
        (("generation", "recovery_output_tokens"), 4096),
        (("generation", "transport_attempts"), 3),
    ],
    ids=lambda value: ".".join(value) if isinstance(value, tuple) else None,
)
def test_any_change_to_what_decides_an_answer_refuses_a_resume(path: tuple[str, ...], value: Any):
    questions = _questions(3)
    run_ = _run_fields(questions)
    ids = [q.id for q in questions]
    stored = run_identity(run_, ids)

    changed = copy.deepcopy(run_)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    problems = identity_mismatches(stored, run_identity(changed, ids))

    assert problems == [f"{'.'.join(path)}: checkpoint " + problems[0].split(": checkpoint ")[1]]
    assert path in IDENTITY_FIELDS


def test_a_different_question_order_refuses_a_resume():
    questions = _questions(3)
    run_ = _run_fields(questions)
    stored = run_identity(run_, ["q-01", "q-02", "q-03"])

    assert identity_mismatches(stored, run_identity(run_, ["q-02", "q-01", "q-03"])) == [
        "question_ids: the ordered question ids differ"
    ]


def test_pacing_is_not_identity():
    questions = _questions(2)
    run_ = _run_fields(questions)
    slower = copy.deepcopy(run_)
    slower["generation"]["delay_seconds"] = 9.0
    slower["retrieval_delay_seconds"] = 4.0

    assert identity_mismatches(run_identity(run_, ["a"]), run_identity(slower, ["a"])) == []


# --- the runner asks nothing it already asked ---------------------------------------


def test_a_resumed_run_makes_no_call_for_a_completed_question():
    questions = _questions(5)
    first = run(run_e2e_evaluation(dataset_of(*questions[:2]), _service())).records
    label = label_of("stack", ANSWERABLE)
    llm = ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label))
    asked: list[str] = []
    service = e2e_service(llm)
    answer = service.answer

    async def counting(question: str, **kwargs: Any) -> Any:
        asked.append(question)
        return await answer(question, **kwargs)

    service.answer = counting  # type: ignore[method-assign,assignment]
    report = run(run_e2e_evaluation(dataset_of(*questions), service, completed=first))

    assert [record.question.id for record in report.records] == [q.id for q in questions]
    assert report.records[:2] == first
    assert len(asked) == 3, "only q-03..q-05 reached the pipeline"
    assert llm.call_count == 3 and llm.check_count == 3


def test_completed_records_must_be_the_first_questions_in_order():
    questions = _questions(3)
    later = run(run_e2e_evaluation(dataset_of(questions[1]), _service())).records

    with pytest.raises(ValueError, match="first questions"):
        run(run_e2e_evaluation(dataset_of(*questions), _service(), completed=later))


def test_a_provider_stop_on_the_last_question_pauses_when_asked_to():
    class StopOnEverything:
        interrupted = 1

        def observe(self, record: E2ERecord) -> AbortReason | None:
            return AbortReason.RATE_LIMITED

    questions = _questions(1)
    plain = run(run_e2e_evaluation(dataset_of(*questions), _service(), guard=StopOnEverything()))
    paused = run(
        run_e2e_evaluation(
            dataset_of(*questions),
            _service(),
            guard=StopOnEverything(),
            pause_on_last_question=True,
        )
    )

    assert plain.aborted is None
    assert paused.aborted == RunAbort(question_id="q-01", reason=AbortReason.RATE_LIMITED)


# --- the validator's view of the segments -------------------------------------------


def _resumed_artifact() -> dict[str, Any]:
    """A logical run of three questions, paused by a 429 after one, then completed."""
    records = _records(3)
    dataset = dataset_of(*(record.question for record in records))
    metadata = run_metadata(dataset)
    payload = export_e2e(E2EReport(records=records), metadata)
    run_ = payload["run"]
    run_.update(tier=Tier.ACCEPTANCE.value, git_dirty=False, git_revision=COMMIT, run_id="L")
    ids = [record.question.id for record in records]
    digest = identity_digest(run_identity(run_, ids))
    calls = [len(question["provider_calls"]) for question in payload["questions"]]
    run_.update(
        {
            "logical_run_id": "L",
            "original_started_at": "t0",
            "completed_at": "t9",
            "resume_count": 1,
            "rate_limit_events": [{"segment": 1, "question_id": "q-02"}],
            "execution_segments": [
                {
                    "segment": 1,
                    "run_id": "L",
                    "identity_sha256": digest,
                    "status": "paused",
                    "stop_reason": "rate_limited",
                    "completed_question_ids": ["q-01"],
                    "interrupted_question_ids": ["q-02"],
                    "calls": calls[0] + 1,
                    "interrupted_calls": 1,
                    "estimated_neurons": 10.0,
                },
                {
                    "segment": 2,
                    "run_id": "S2",
                    "identity_sha256": digest,
                    "status": "complete",
                    "stop_reason": None,
                    "completed_question_ids": ["q-02", "q-03"],
                    "interrupted_question_ids": [],
                    "calls": calls[1] + calls[2],
                    "interrupted_calls": 0,
                    "estimated_neurons": 20.0,
                },
            ],
        }
    )
    return payload


def test_a_consistent_resumed_run_has_no_segment_problem():
    assert segment_problems(_resumed_artifact()) == []


@pytest.mark.parametrize(
    ("manipulate", "finding"),
    [
        (
            lambda p: p["run"]["execution_segments"][1]["completed_question_ids"].insert(0, "q-01"),
            "more than one segment",
        ),
        (
            lambda p: p["run"]["execution_segments"][1]["completed_question_ids"].pop(),
            "not the recorded questions in order",
        ),
        (lambda p: p["questions"].reverse(), "different identity"),
        (lambda p: p["run"].update(git_revision="d" * 40), "different identity"),
        (lambda p: p["run"]["generation"].update(model="other"), "different identity"),
        (
            lambda p: p["run"]["execution_segments"][0].update(stop_reason="internal_defect"),
            "not paused by an external stop",
        ),
        (lambda p: p["run"]["execution_segments"][1].update(calls=99), "kept provider calls"),
        (lambda p: p["run"].update(resume_count=0), "resume_count"),
        (lambda p: p["run"]["execution_segments"][1].update(status="paused"), "last segment"),
    ],
    ids=[
        "duplicate",
        "missing",
        "reordered",
        "commit",
        "model",
        "unresolved-pause",
        "usage",
        "resume-count",
        "incomplete-segment",
    ],
)
def test_the_validator_finds_a_manipulated_execution_record(manipulate: Any, finding: str):
    payload = _resumed_artifact()
    manipulate(payload)

    assert any(finding in problem for problem in segment_problems(payload))


def test_a_resumed_run_can_be_a_release_acceptance():
    payload = with_release_acceptance(_resumed_artifact())
    check = validate_artifact(payload)

    assert check.problems == ()
    assert payload["release_acceptance"]["conditions"]["execution_segments_consistent"]


def test_a_v1_artifact_is_still_judged_by_v1():
    path = Path("evaluation/results/release-acceptance-v1.1.1-final.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    check = validate_artifact(payload)

    assert check.consistent
    assert check.verdict == "FAIL"
    assert "gate no_pipeline_errors FAIL" in check.reasons


# --- budget for what is left --------------------------------------------------------


def _preflight(**overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "tier": Tier.ACCEPTANCE,
        "questions": 24,
        "provider": PROFILE.provider,
        "model": PROFILE.model,
        "profile": PROFILE,
        "history": UsageHistory(calls=(), questions=0, sources=(), newest=None, notes=()),
        "ledger": LedgerReading(entries=(), unreadable_lines=0),
        "today": date(2026, 10, 5),
        "policy": BudgetPolicy(),
        "commit": COMMIT,
        "dirty": False,
        "source_identity": "5" * 64,
    }
    return preflight(**{**arguments, **overrides})


def test_a_resume_forecasts_only_the_remaining_questions_and_the_canary():
    whole = _preflight(questions=24)
    remaining = _preflight(questions=14, extra_neurons=40.0, resuming="L")

    assert remaining.forecast.questions == 14
    assert remaining.forecast.neurons == pytest.approx(whole.forecast.neurons * 14 / 24)
    assert remaining.planned_neurons == pytest.approx(remaining.forecast.neurons + 40.0)
    assert remaining.projected_total == pytest.approx(remaining.planned_neurons)


def _spent_today(neurons: float) -> LedgerReading:
    entry = LedgerEntry(
        date_utc="2026-10-05",
        run_id="L",
        tier="acceptance",
        status="aborted",
        calls=20,
        input_tokens=0,
        output_tokens=0,
        estimated_neurons=neurons,
        usage_coverage=1.0,
        artifact=None,
        abort_reason="rate_limited",
        commit_sha=COMMIT,
        git_dirty=False,
        failed_gates=["no_pipeline_errors"],
        release_source_identity="5" * 64,
        logical_run_id="L",
        segment=1,
    )
    return LedgerReading(entries=(entry,), unreadable_lines=0)


def test_the_hard_cap_holds_on_resume_and_the_reserve_override_still_works():
    nearly_spent = _spent_today(9_000.0)
    capped = _preflight(questions=14, ledger=nearly_spent, resuming="L", override_reserve=True)
    assert capped.zone is BudgetZone.RED, "the nominal daily budget is never overruled"

    half = _spent_today(5_000.0)
    reserve = _preflight(questions=6, ledger=half, resuming="L")
    overridden = _preflight(questions=6, ledger=half, resuming="L", override_reserve=True)
    assert reserve.zone is BudgetZone.RED and not reserve.reserve_kept
    assert overridden.zone is not BudgetZone.RED and overridden.reserve_overridden


def test_a_resume_is_not_blocked_by_its_own_earlier_segment():
    ledger = _spent_today(1_000.0)

    assert _preflight(questions=14, ledger=ledger, resuming="L").blockers == ()
    fresh = _preflight(questions=24, ledger=ledger)
    assert any("already has an acceptance run" in blocker for blocker in fresh.blockers)


def test_the_segments_of_one_logical_run_are_one_attempt():
    entries = [
        replace_entry(segment=1, run_id="L", status="aborted"),
        replace_entry(segment=2, run_id="S2", status="complete", failed_gates=[]),
    ]
    (attempt,) = acceptance_attempts(LedgerReading(entries=tuple(entries), unreadable_lines=0))

    assert attempt.run_id == "L"
    assert attempt.passed


def replace_entry(**changes: Any) -> LedgerEntry:
    from dataclasses import replace

    return replace(_spent_today(100.0).entries[0], **changes)


# --- the canary ---------------------------------------------------------------------


class _Answering:
    def __init__(self, text: str, finish_reason: str = "stop") -> None:
        self.text, self.finish_reason = text, finish_reason
        self.requests: list[GenerationRequest] = []

    @property
    def model(self) -> str:
        return "canary-test"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        return GenerationResponse(
            text=self.text,
            model="canary-test",
            finish_reason=self.finish_reason,
            usage=TokenUsage(input_tokens=40, output_tokens=12),
        )


def test_a_healthy_provider_passes_the_canary_with_one_small_request():
    llm = _Answering('{"ok": true}')

    result = run(run_canary(llm))

    assert result.status is CanaryStatus.PASS
    (request,) = llm.requests
    assert request.max_output_tokens == CANARY_MAX_OUTPUT_TOKENS
    text = " ".join(message.content for message in request.messages)
    assert "ok" in text and "SOURCE" not in text, "no retrieval, no context, no portfolio question"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (
            LLMProviderError(
                "x",
                retryable=True,
                kind=ProviderFailureKind.RATE_LIMITED,
                status_code=429,
                retry_after_seconds=30.0,
            ),
            CanaryStatus.RATE_LIMITED,
        ),
        (
            LLMProviderError("x", kind=ProviderFailureKind.HTTP_STATUS, status_code=401),
            CanaryStatus.AUTH_FAILURE,
        ),
        (
            LLMProviderError("x", kind=ProviderFailureKind.HTTP_STATUS, status_code=403),
            CanaryStatus.AUTH_FAILURE,
        ),
        (
            LLMProviderError(
                "x", retryable=True, kind=ProviderFailureKind.HTTP_STATUS, status_code=503
            ),
            CanaryStatus.PROVIDER_FAILURE,
        ),
        (
            LLMProviderError("x", retryable=True, kind=ProviderFailureKind.TIMEOUT),
            CanaryStatus.TIMEOUT_NETWORK,
        ),
        (
            LLMProviderError("x", retryable=True, kind=ProviderFailureKind.UNREACHABLE),
            CanaryStatus.TIMEOUT_NETWORK,
        ),
        (
            LLMProviderError("x", kind=ProviderFailureKind.MALFORMED_RESPONSE),
            CanaryStatus.MALFORMED_RESPONSE,
        ),
    ],
    ids=["429", "401", "403", "5xx", "timeout", "network", "malformed"],
)
def test_the_canary_classifies_every_refusal(error: LLMProviderError, status: CanaryStatus):
    result = run(run_canary(FailingLLMProvider(error)))

    assert result.status is status
    assert not result.passed
    assert result.fields()["canary_status_code"] == error.status_code
    if status is CanaryStatus.RATE_LIMITED:
        assert result.retry_after_seconds == 30.0


@pytest.mark.parametrize(
    ("text", "finish", "category"),
    [
        ("not json", "stop", "reply_not_ok_json"),
        ('{"ok": false}', "stop", "reply_not_ok_json"),
        ('{"ok": tr', "length", "output_truncated"),
    ],
)
def test_a_reply_that_is_not_the_ok_object_fails_the_canary(text: str, finish: str, category: str):
    result = run(run_canary(_Answering(text, finish)))

    assert result.status is CanaryStatus.MALFORMED_RESPONSE
    assert result.failure_category == category


def test_the_canary_request_is_fixed_and_its_cost_is_bounded():
    request = canary_request()

    assert request.max_output_tokens == CANARY_MAX_OUTPUT_TOKENS
    assert canary_neurons_bound(PROFILE) < 100
