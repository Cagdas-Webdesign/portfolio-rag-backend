"""Paket 4: what a provider run costs, whether it may start, and when it stops.

Everything offline. The cost model is checked against the published rates, the
forecast against hand-made histories, the zones against the boundaries the
policy names, the ledger against the files it will meet in practice — missing,
empty, damaged, several runs a day, a new day — and the guard and the smoke
verdict against the runs they exist to stop or to pass.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.evaluation import export_e2e, render_summary, run_e2e_evaluation
from portfolio_rag.evaluation.budget import (
    COST_PROFILES,
    BudgetPolicy,
    BudgetZone,
    Confidence,
    Ledger,
    LedgerEntry,
    LedgerReading,
    UsageHistory,
    budget_zone,
    cost_profile,
    forecast_run,
    percentile,
    spent,
    structural_call_bound,
)
from portfolio_rag.evaluation.e2e import (
    E2E_FORMAT_VERSION,
    AbortReason,
    E2EFailure,
    E2ERecord,
    E2EReport,
    RunAbort,
    errored_record,
)
from portfolio_rag.evaluation.operations import (
    MAX_SMOKE_QUESTIONS,
    Preflight,
    ProfileRequirements,
    RunGuard,
    Tier,
    load_history,
    preflight,
    smoke_verdict,
)
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.errors import GenerationFailure, GenerationFailureCategory
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.service import MAX_GENERATION_ATTEMPTS, MAX_GROUNDING_CHECK_ATTEMPTS
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType
from tests.doubles import ScriptedLLMProvider, ScriptedReply, grounded
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.support import run

PROFILE = COST_PROFILES[0]
TODAY = date(2026, 10, 4)
C = GenerationFailureCategory


def call(
    call_type: ProviderCallType = ProviderCallType.GENERATION,
    *,
    tokens: tuple[int, int] | None = (2400, 300),
    attempt: int = 1,
) -> ProviderCallRecord:
    return ProviderCallRecord(
        call_type=call_type,
        attempt=attempt,
        model=PROFILE.model,
        response_format=ResponseFormat.JSON_OBJECT,
        elapsed_seconds=1.0,
        result=CallResult.PARSED,
        input_tokens=tokens[0] if tokens else None,
        output_tokens=tokens[1] if tokens else None,
    )


def history(
    *, questions: int, generations: int, checks: int, tokens: tuple[int, int] | None = (2400, 300)
) -> UsageHistory:
    calls = [call(tokens=tokens) for _ in range(generations)]
    calls += [call(ProviderCallType.GROUNDING_CHECK, tokens=(800, 10)) for _ in range(checks)]
    return UsageHistory(calls=tuple(calls), questions=questions)


EMPTY = UsageHistory(calls=(), questions=0)


# --- the cost model -------------------------------------------------------------------


def test_the_published_rates_live_in_one_profile():
    assert cost_profile("cloudflare_workers_ai", "@cf/openai/gpt-oss-120b") is PROFILE
    assert PROFILE.estimate(1_000_000, 0) == 31_818
    assert PROFILE.estimate(0, 1_000_000) == 68_182
    assert cost_profile("mistral", "mistral-small-latest") is None


def test_the_structural_bound_is_the_context_budget_and_the_output_limit():
    policy = ContextPolicy()
    expected = PROFILE.estimate(
        policy.max_prompt_tokens - policy.output_reserve_tokens, policy.output_reserve_tokens
    )
    assert structural_call_bound(PROFILE, policy) == pytest.approx(expected)


def test_a_call_without_usage_is_never_free():
    total, unreported = spent([call(tokens=None), call()], PROFILE, ContextPolicy())

    assert unreported == 1
    assert total == pytest.approx(
        structural_call_bound(PROFILE, ContextPolicy()) + PROFILE.estimate(2400, 300)
    )


def test_percentile_is_a_value_that_occurred():
    assert percentile([1, 2, 3, 4], 0.75) == 3
    assert percentile([10], 0.75) == 10
    assert percentile([5, 1, 9, 3, 7, 2, 8, 4], 0.75) == 7


# --- the forecast ---------------------------------------------------------------------


def test_without_measured_usage_the_forecast_is_the_structural_bound():
    forecast = forecast_run(49, EMPTY, PROFILE)

    policy = ContextPolicy()
    bound = structural_call_bound(PROFILE, policy)
    # A regeneration may ask for the recovery cap, and is costed at it.
    recovery = structural_call_bound(PROFILE, policy, output_tokens=policy.recovery_output_tokens)
    assert recovery > bound
    assert forecast.confidence is Confidence.LOW
    assert forecast.per_question_neurons == pytest.approx(2 * bound + 0.2 * recovery)
    assert forecast.neurons == pytest.approx(49 * (2 * bound + 0.2 * recovery))
    assert forecast.input_tokens is None, "no measured tokens are claimed"


def test_a_small_measured_history_gives_a_medium_forecast_with_its_margin():
    # Five questions, one of them regenerated: 6 generations, 5 checks.
    measured = history(questions=5, generations=6, checks=5)

    forecast = forecast_run(10, measured, PROFILE)

    per_question = 1.25 * (1.2 * PROFILE.estimate(2400, 300) + PROFILE.estimate(800, 10))
    assert forecast.confidence is Confidence.MEDIUM
    assert forecast.per_question_neurons == pytest.approx(per_question)
    assert forecast.regeneration_calls == pytest.approx(2.0)
    assert forecast.grounding_calls == 10, "every question is assumed to reach the check"


def test_a_large_well_covered_history_gives_a_high_forecast():
    forecast = forecast_run(49, history(questions=40, generations=44, checks=36), PROFILE)

    assert forecast.confidence is Confidence.HIGH
    assert "+10%" in forecast.method


def test_missing_usage_lowers_confidence_instead_of_lowering_the_cost():
    calls = [call() for _ in range(6)] + [call(tokens=None) for _ in range(6)]
    calls += [call(ProviderCallType.GROUNDING_CHECK, tokens=(800, 10)) for _ in range(6)]
    partial = UsageHistory(calls=tuple(calls), questions=6)

    forecast = forecast_run(10, partial, PROFILE)

    assert partial.usage_coverage == pytest.approx(12 / 18)
    assert forecast.confidence is Confidence.LOW, "coverage below 80% is not trusted"


def test_one_outlier_does_not_set_the_forecast():
    calls = [call(tokens=(2400, 300)) for _ in range(9)] + [call(tokens=(2400, 800))]
    calls += [call(ProviderCallType.GROUNDING_CHECK, tokens=(800, 10)) for _ in range(10)]
    forecast = forecast_run(1, UsageHistory(calls=tuple(calls), questions=10), PROFILE)

    assert forecast.output_tokens is not None
    assert forecast.output_tokens < 1.25 * (300 + 10) + 1, "p75, not the maximum"


# --- the zones --------------------------------------------------------------------------

POLICY = BudgetPolicy()


@pytest.mark.parametrize(
    ("forecast", "known", "zone"),
    [
        (4_200, 0, BudgetZone.GREEN),
        (5_300, 0, BudgetZone.YELLOW),
        (6_800, 0, BudgetZone.RED),
        (5_000, 0, BudgetZone.GREEN),  # exactly the target share
        (6_000, 0, BudgetZone.YELLOW),  # the reserve kept exactly
        (6_001, 0, BudgetZone.RED),  # the reserve missed by one
        (2_000, 4_000, BudgetZone.GREEN),  # earlier runs today count
        (2_000, 4_001, BudgetZone.RED),
    ],
)
def test_the_reserve_decides_red_and_the_target_decides_yellow(
    forecast: float, known: float, zone: BudgetZone
):
    assert budget_zone(forecast, known, POLICY) is zone


@pytest.mark.parametrize(
    "kwargs",
    [
        {"daily_budget": 0},
        {"minimum_reserve": -1},
        {"minimum_reserve": 10_000},
        {"target_share": 0},
        {"target_share": 1.5},
    ],
)
def test_a_policy_that_cannot_protect_anything_is_refused(kwargs: dict[str, float]):
    with pytest.raises(ValueError):
        BudgetPolicy(**kwargs)


# --- the ledger --------------------------------------------------------------------------


def entry(
    *, day: date = TODAY, neurons: float = 1_000, tier: str = "smoke", **extra: Any
) -> LedgerEntry:
    return LedgerEntry(
        date_utc=day.isoformat(),
        run_id=extra.pop("run_id", "r" * 32),
        tier=tier,
        status=extra.pop("status", "complete"),
        calls=10,
        input_tokens=24_000,
        output_tokens=3_000,
        estimated_neurons=neurons,
        usage_coverage=1.0,
        artifact="evaluation/results/x.json",
        **extra,
    )


def test_a_missing_or_empty_ledger_is_an_empty_ledger(tmp_path: Path):
    assert Ledger(tmp_path / "missing.jsonl").read().entries == ()
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    assert Ledger(tmp_path / "empty.jsonl").read().known_consumption(TODAY) == 0


def test_runs_of_one_utc_day_add_up_and_a_new_day_starts_at_zero(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    ledger.append(entry(neurons=1_200))
    ledger.append(entry(neurons=800, tier="experiment", status="aborted", abort_reason="max_calls"))
    ledger.append(entry(day=date(2026, 10, 3), neurons=9_000, tier="acceptance"))

    reading = ledger.read()

    assert reading.known_consumption(TODAY) == 2_000
    assert reading.known_consumption(date(2026, 10, 5)) == 0
    assert [e.status for e in reading.for_day(TODAY)] == ["complete", "aborted"]


def test_a_damaged_line_is_counted_and_does_not_hide_the_others(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    Ledger(path).append(entry(neurons=500))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
        handle.write(json.dumps({"date_utc": TODAY.isoformat()}) + "\n")  # missing fields
    Ledger(path).append(entry(neurons=700))

    reading = Ledger(path).read()

    assert reading.unreadable_lines == 2
    assert reading.known_consumption(TODAY) == 1_200


# --- preflight -----------------------------------------------------------------------------


def plan(
    tier: Tier = Tier.ACCEPTANCE,
    questions: int = 49,
    *,
    measured: UsageHistory | None = None,
    ledger: tuple[LedgerEntry, ...] = (),
    profile: Any = PROFILE,
) -> Preflight:
    return preflight(
        tier=tier,
        questions=questions,
        provider=PROFILE.provider,
        model=PROFILE.model,
        profile=profile,
        history=measured if measured is not None else EMPTY,
        ledger=LedgerReading(entries=ledger, unreadable_lines=0),
        today=TODAY,
        policy=POLICY,
    )


#: A measured history cheap enough for 49 questions to fit the target.
GOOD_HISTORY = history(questions=40, generations=44, checks=36, tokens=(1200, 150))
#: The same history at the token counts a broad production answer has: 49
#: questions no longer fit the reserve.
REALISTIC_HISTORY = history(questions=40, generations=44, checks=36)


def test_without_history_acceptance_is_red_and_a_smoke_run_needs_a_go():
    acceptance, smoke = plan(), plan(Tier.SMOKE, 5)

    assert acceptance.zone is BudgetZone.RED
    assert not acceptance.can_start(yellow_confirmed=True), "no override for RED"
    # ~2,400 neurons for five questions: far above what a health check should cost.
    assert smoke.zone is BudgetZone.YELLOW
    assert not smoke.can_start(yellow_confirmed=False)
    assert smoke.can_start(yellow_confirmed=True)
    assert any("operationally expensive" in warning for warning in smoke.warnings)
    assert any("low" in warning for warning in acceptance.warnings)


def test_at_realistic_token_counts_a_full_run_does_not_fit_the_reserve():
    """Not a test of the code's arithmetic so much as a warning it can give."""
    heavy = plan(measured=REALISTIC_HISTORY, ledger=(entry(neurons=0),))

    assert heavy.forecast is not None and heavy.forecast.neurons > POLICY.spendable
    assert heavy.zone is BudgetZone.RED


def test_with_a_measured_history_acceptance_is_forecast_from_it():
    forecast_plan = plan(measured=GOOD_HISTORY, ledger=(entry(neurons=0),))

    assert forecast_plan.forecast is not None
    assert forecast_plan.forecast.confidence is Confidence.HIGH
    assert forecast_plan.zone is BudgetZone.GREEN
    assert forecast_plan.can_start(yellow_confirmed=False)
    assert forecast_plan.fields()["actual_remaining"] is None, "never claimed"


def test_yellow_needs_an_explicit_go_and_red_never_starts():
    yellow = plan(measured=GOOD_HISTORY, ledger=(entry(neurons=0),))
    yellow = replace(yellow, zone=BudgetZone.YELLOW)
    red = replace(yellow, zone=BudgetZone.RED)

    assert yellow.confirmation_required
    assert not yellow.can_start(yellow_confirmed=False)
    assert yellow.can_start(yellow_confirmed=True)
    assert not red.can_start(yellow_confirmed=True)


def test_a_smoke_run_stays_small():
    too_big = plan(Tier.SMOKE, MAX_SMOKE_QUESTIONS + 1)

    assert too_big.blockers
    assert not too_big.can_start(yellow_confirmed=True)


def test_acceptance_after_a_failed_smoke_today_does_not_start():
    failed = entry(smoke_passed=False, run_id="s" * 32)

    blocked = plan(measured=GOOD_HISTORY, ledger=(failed,))

    assert any("smoke" in blocker for blocker in blocked.blockers)
    assert not blocked.can_start(yellow_confirmed=True)


def test_no_run_today_is_not_a_reason_for_a_smoke_run():
    """History of earlier days is the forecast's basis; today's ledger is not."""
    warnings = plan(measured=GOOD_HISTORY).warnings

    assert not any("smoke" in w for w in warnings)


def test_a_provider_without_neuron_rates_is_not_budgeted():
    free = plan(profile=None)

    assert free.forecast is None and free.zone is BudgetZone.GREEN
    assert any("budget does not apply" in w for w in free.warnings)


def test_the_bound_on_calls_is_the_failure_policy_ceiling():
    """Two generations and two grounding checks per question, never more."""
    assert MAX_GENERATION_ATTEMPTS == MAX_GROUNDING_CHECK_ATTEMPTS == 2
    assert plan().max_port_calls == 4 * 49


# --- history from earlier exports -----------------------------------------------------------


REQUIREMENTS = ProfileRequirements(
    provider=PROFILE.provider,
    model=PROFILE.model,
    prompt_version="grounded-answer-v6",
    grounding_check_version="grounding-check-v1",
    response_format="json_object",
    max_prompt_tokens=6000,
    output_reserve_tokens=800,
    top_k=5,
    min_similarity=0.25,
)


def _export(path: Path, *, fmt: str = E2E_FORMAT_VERSION, model: str = PROFILE.model) -> None:
    calls = [
        {**call().fields()},
        {**call(ProviderCallType.GROUNDING_CHECK, tokens=(800, 10)).fields()},
    ]
    payload = {
        "format": fmt,
        "run": {"generation": {"provider": PROFILE.provider, "model": model}},
        "questions": [{"id": "q", "provider_calls": calls}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_history_reads_only_exports_that_measured_this_model(tmp_path: Path):
    _export(tmp_path / "v3.json")
    _export(tmp_path / "v2.json", fmt="e2e-eval-v2")
    _export(tmp_path / "other.json", model="@cf/meta/llama-3.1-8b-instruct")
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")

    found = load_history(tmp_path, REQUIREMENTS)

    assert found.sources == ("v3.json",)
    assert (len(found.calls), found.questions) == (2, 1)
    assert found.calls[0].input_tokens == 2400


def test_a_missing_history_directory_is_no_history(tmp_path: Path):
    assert load_history(tmp_path / "nope", REQUIREMENTS).calls == ()


# --- the guard -------------------------------------------------------------------------------


#: One ordinary question at the cheap history's token counts.
ORDINARY = (call(tokens=(1200, 150)), call(ProviderCallType.GROUNDING_CHECK, tokens=(800, 10)))


def record(
    failure: GenerationFailure | None = None,
    *,
    calls: tuple[ProviderCallRecord, ...] = ORDINARY,
) -> E2ERecord:
    made = errored_record(
        ANSWERABLE, "X", duration_seconds=0.1, failure=failure, provider_calls=calls
    )
    return replace(made, error=failure)


def failure(
    category: GenerationFailureCategory, status: int | None = None, detail: str = "x"
) -> GenerationFailure:
    return GenerationFailure(category=category, detail=detail, status_code=status)


def guard(total: int = 49, *, per_question: float | None = None, known: float = 0) -> RunGuard:
    base = plan(measured=GOOD_HISTORY, ledger=(entry(neurons=known),))
    if per_question is not None:
        assert base.forecast is not None
        base = replace(base, forecast=replace(base.forecast, per_question_neurons=per_question))
    return RunGuard(preflight=base, total_questions=total)


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (failure(C.RETRYABLE_PROVIDER_ERROR, 429, "rate_limited"), AbortReason.RATE_LIMITED),
        (failure(C.PROVIDER_STATUS, 401), AbortReason.AUTH_FAILURE),
        (failure(C.PROVIDER_STATUS, 403), AbortReason.AUTH_FAILURE),
    ],
)
def test_the_provider_saying_no_more_stops_the_run_at_once(
    bad: GenerationFailure, reason: AbortReason
):
    assert guard().observe(record(bad, calls=())) is reason


def test_one_provider_failure_is_a_result_two_in_a_row_are_systemic():
    watching = guard()
    server_error = failure(C.RETRYABLE_PROVIDER_ERROR, 503)

    assert watching.observe(record(server_error)) is None
    assert watching.observe(record()) is None, "a success resets the streak"
    assert watching.observe(record(server_error)) is None
    assert watching.observe(record(failure(C.TIMEOUT))) is AbortReason.SYSTEMIC_PROVIDER_FAILURE


def test_a_classified_output_failure_is_not_a_reason_to_stop():
    watching = guard()

    for _ in range(3):
        assert watching.observe(record(failure(C.OUTPUT_TRUNCATED))) is None


def test_a_run_heading_far_past_its_forecast_is_stopped_before_the_reserve():
    # Forecast says ~50 per question; the run spends ~140 per question.
    watching = guard(per_question=50)
    expensive = (call(tokens=(4000, 200)), call(ProviderCallType.GROUNDING_CHECK, tokens=(500, 10)))

    stops = [watching.observe(record(calls=expensive)) for _ in range(10)]

    assert AbortReason.BUDGET_GUARD in stops
    first = stops.index(AbortReason.BUDGET_GUARD)
    assert first >= 4, "not before the run has shown its own cost"


def test_a_run_slightly_over_its_forecast_goes_on():
    # Each question costs a few neurons more than the forecast said.
    watching = guard(per_question=PROFILE.estimate(1200, 150) + PROFILE.estimate(800, 10) - 5)

    assert all(watching.observe(record()) is None for _ in range(49))


def test_spend_without_usage_is_counted_at_the_bound():
    watching = guard()
    watching.observe(record(calls=(call(tokens=None),)))

    assert watching.unreported_calls == 1
    assert watching.estimated_neurons == pytest.approx(
        structural_call_bound(PROFILE, ContextPolicy())
    )


# --- the guard inside a run ----------------------------------------------------------------


class _StopAfterFirst:
    def __init__(self) -> None:
        self.seen = 0

    def observe(self, record: E2ERecord) -> AbortReason | None:
        self.seen += 1
        return AbortReason.RATE_LIMITED


def _three() -> Any:
    return dataset_of(
        ANSWERABLE,
        ANSWERABLE.model_copy(update={"id": "q-2"}),
        ANSWERABLE.model_copy(update={"id": "q-3"}),
    )


def test_a_guarded_run_stops_and_keeps_what_was_measured():
    llm = ScriptedLLMProvider(grounded("FastAPI.", "S1"))
    stopper = _StopAfterFirst()

    report = run(run_e2e_evaluation(_three(), e2e_service(llm), guard=stopper))

    assert stopper.seen == 1
    assert len(report.records) == 1, "nothing asked after the stop"
    assert report.aborted == RunAbort(question_id="q-api", reason=AbortReason.RATE_LIMITED)
    assert not report.passed


def test_a_stop_after_the_last_question_leaves_the_run_complete():
    llm = ScriptedLLMProvider(grounded("FastAPI.", "S1"))

    report = run(
        run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(llm), guard=_StopAfterFirst())
    )

    assert report.aborted is None


# --- the smoke verdict -----------------------------------------------------------------------


def _smoke(
    *replies: ScriptedReply, check: ScriptedReply | None = None, questions: int = 1
) -> E2EReport:
    llm = ScriptedLLMProvider(*replies, grounding_check=check)
    dataset = dataset_of(
        *(ANSWERABLE.model_copy(update={"id": f"q-{n}"}) for n in range(questions))
    )
    return run(run_e2e_evaluation(dataset, e2e_service(llm)))


def test_a_recovered_regeneration_and_a_controlled_refusal_pass_the_smoke():
    label = label_of("stack", ANSWERABLE)
    recovered = _smoke(
        ScriptedReply(raw_text='{"answer": "cut', finish_reason="length"),
        grounded(f"FastAPI [{label}].", label),
    )
    refused = _smoke(ScriptedReply(answer="", sources=(), support="none"))

    assert smoke_verdict(recovered).passed
    assert smoke_verdict(refused).passed


def test_a_single_classified_failure_passes_with_a_note():
    label = label_of("stack", ANSWERABLE)
    truncated = ScriptedReply(raw_text='{"answer": "cut', finish_reason="length")
    # Five questions; the first is cut off twice, the others answer.
    verdict = smoke_verdict(
        _smoke(truncated, truncated, grounded(f"FastAPI [{label}].", label), questions=5)
    )

    assert verdict.passed
    assert any("output_truncated" in note for note in verdict.reasons)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMProviderError("?"), "unclassified"),
        (LLMProviderError("no", kind=ProviderFailureKind.HTTP_STATUS, status_code=401), "refused"),
    ],
)
def test_unclassified_and_refused_requests_fail_the_smoke(error: LLMProviderError, expected: str):
    verdict = smoke_verdict(_smoke(ScriptedReply(error=error)))

    assert not verdict.passed
    assert any(expected in reason for reason in verdict.reasons)


@pytest.mark.parametrize("reason", list(AbortReason))
def test_any_aborted_run_fails_the_smoke(reason: AbortReason):
    report = E2EReport(records=(), aborted=RunAbort(question_id="q", reason=reason))

    assert not smoke_verdict(report).passed


def test_a_safety_failure_fails_the_smoke():
    label = label_of("stack", ANSWERABLE)
    answered = _smoke(grounded(f"FastAPI [{label}].", label)).records[0]
    leaked = replace(answered, failures=(E2EFailure.LABEL_LEAK,))

    assert not smoke_verdict(E2EReport(records=(leaked,))).passed


# --- reporting ----------------------------------------------------------------------------


def test_the_summary_names_estimates_as_estimates():
    dataset = dataset_of(ANSWERABLE)
    report = run(run_e2e_evaluation(dataset, e2e_service(ScriptedLLMProvider())))
    operations = {
        **plan(Tier.SMOKE, 1).fields(),
        "status": "complete",
        "observed": {"estimated_neurons": 123.4, "usage_coverage": 0.96},
        "known_daily_after": 1_123.4,
        "nominal_reserve_after": 8_876.6,
        "smoke": {"passed": True, "reasons": []},
    }

    summary = render_summary(export_e2e(report, run_metadata(dataset), operations))

    assert "## Budget / Operations" in summary
    assert "| Tier | smoke |" in summary
    assert "| Observed estimate | 123 |" in summary
    assert "| Usage coverage | 96 % |" in summary
    assert "Neurons are estimates" in summary
    assert "| Smoke | pass (—) |" in summary
