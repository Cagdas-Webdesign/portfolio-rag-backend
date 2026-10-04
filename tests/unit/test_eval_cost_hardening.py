"""Paket 4B: budget scenarios, simulated offline.

Each scenario the hardening was asked to get right, as a test: the cost bands
and the zones at their boundaries, a run that turns out dearer than forecast,
one expensive question among ordinary ones, history from an earlier day, from
an incompatible run, from no run at all, and history with gaps. No provider is
called, and nothing here could call one.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.evaluation import export_e2e, render_summary, run_e2e_evaluation
from portfolio_rag.evaluation.budget import (
    COST_PROFILES,
    BudgetPolicy,
    BudgetZone,
    Confidence,
    CostBand,
    LedgerReading,
    UsageHistory,
    budget_zone,
    cost_band,
    forecast_run,
    robust_cost,
)
from portfolio_rag.evaluation.e2e import AbortReason, E2ERecord, errored_record
from portfolio_rag.evaluation.operations import (
    Preflight,
    ProfileRequirements,
    RunGuard,
    Tier,
    load_history,
    preflight,
    smoke_verdict,
)
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType
from tests.doubles import ScriptedLLMProvider, ScriptedReply, grounded
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.support import run

PROFILE = COST_PROFILES[0]
POLICY = BudgetPolicy()
TODAY = date(2026, 10, 4)
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


# --- A to D: bands and zones -------------------------------------------------------------


@pytest.mark.parametrize(
    ("forecast", "band", "zone"),
    [
        (3_800, CostBand.TARGET, BudgetZone.GREEN),  # A
        (4_700, CostBand.GOOD, BudgetZone.GREEN),  # B
        (5_800, CostBand.CAUTION, BudgetZone.YELLOW),  # C: reserve 4,200 kept
        (7_200, CostBand.EXCESSIVE, BudgetZone.RED),  # D
        (4_000, CostBand.TARGET, BudgetZone.GREEN),  # exactly 40 %
        (5_000, CostBand.GOOD, BudgetZone.GREEN),  # exactly 50 %
        (5_001, CostBand.CAUTION, BudgetZone.YELLOW),  # not a hard stop at 50 %
    ],
    ids=["A-target", "B-good", "C-caution", "D-excessive", "40%", "50%", "just-over-50%"],
)
def test_the_cost_band_describes_and_the_zone_decides(
    forecast: float, band: CostBand, zone: BudgetZone
):
    assert cost_band(forecast, POLICY.daily_budget) is band
    assert budget_zone(forecast, 0, POLICY) is zone


def test_seventy_percent_is_the_last_acceptable_share():
    lenient = BudgetPolicy(minimum_reserve=2_000)

    assert cost_band(7_000, POLICY.daily_budget) is CostBand.CAUTION
    assert budget_zone(7_000, 0, lenient) is BudgetZone.YELLOW
    assert cost_band(7_001, POLICY.daily_budget) is CostBand.EXCESSIVE
    # Even with a smaller reserve, a run above 70 % is not a normal run.
    assert budget_zone(7_001, 0, lenient) is BudgetZone.RED


def test_the_reserve_still_decides_on_its_own():
    """A cheap run on a day already spent does not start."""
    assert cost_band(2_000, POLICY.daily_budget) is CostBand.TARGET
    assert budget_zone(2_000, 4_500, POLICY) is BudgetZone.RED


# --- E and F: the guard during a run ---------------------------------------------------------


def call(
    call_type: ProviderCallType = ProviderCallType.GENERATION,
    tokens: tuple[int, int] | None = (1200, 150),
) -> ProviderCallRecord:
    return ProviderCallRecord(
        call_type=call_type,
        attempt=1,
        model=PROFILE.model,
        response_format=ResponseFormat.JSON_OBJECT,
        elapsed_seconds=1.0,
        result=CallResult.PARSED,
        input_tokens=tokens[0] if tokens else None,
        output_tokens=tokens[1] if tokens else None,
    )


def question_costing(neurons: float) -> E2ERecord:
    """A question whose one generation costs *neurons*, by input tokens alone."""
    tokens = round(neurons * 1_000_000 / PROFILE.input_neurons_per_million)
    made = errored_record(
        ANSWERABLE, "X", duration_seconds=0.1, provider_calls=(call(tokens=(tokens, 0)),)
    )
    return replace(made, error=None)


def guard_for(total_forecast: float, *, questions: int = 49, known: float = 0) -> RunGuard:
    base = _plan(history=_measured(), known=known)
    assert base.forecast is not None
    forecast = replace(
        base.forecast, neurons=total_forecast, per_question_neurons=total_forecast / questions
    )
    return RunGuard(preflight=replace(base, forecast=forecast), total_questions=questions)


def test_scenario_e_a_run_far_dearer_than_forecast_is_stopped_early():
    watching = guard_for(4_300)
    per_question = 8_000 / 49

    asked = 0
    for _ in range(49):
        asked += 1
        if watching.observe(question_costing(per_question)) is AbortReason.BUDGET_GUARD:
            break

    assert 2 <= asked <= 5, "not on the first question, and long before question 30"
    assert watching.estimated_neurons < 1_000, "stopped with most of the budget unspent"


def test_scenario_e_the_projection_moves_towards_the_run_with_every_question():
    watching = guard_for(4_300)
    projections = []
    for _ in range(3):
        watching.observe(question_costing(8_000 / 49))
        assert watching.projected_final_neurons is not None
        projections.append(watching.projected_final_neurons)

    assert projections[0] < projections[1] < projections[2]
    assert projections[0] < 5_500, "one question barely moves it"


def test_scenario_f_one_expensive_question_does_not_end_a_reasonable_run():
    watching = guard_for(4_300)
    ordinary = 4_300 / 49

    stops = [watching.observe(question_costing(700))]
    stops += [watching.observe(question_costing(ordinary)) for _ in range(48)]

    assert AbortReason.BUDGET_GUARD not in stops
    assert watching.estimated_neurons == pytest.approx(700 + 48 * ordinary, rel=1e-3), (
        "the expensive question is counted in full — just not projected onto the rest"
    )


def test_the_guard_reports_where_the_run_is_heading():
    watching = guard_for(4_300)
    for _ in range(10):
        watching.observe(question_costing(6_500 / 49))

    assert watching.projected_band is CostBand.CAUTION
    assert watching.status == "caution"


# --- G to J: history -----------------------------------------------------------------------


def _export(
    path: Path,
    *,
    made: date,
    model: str = PROFILE.model,
    prompt_version: str = "grounded-answer-v6",
    tokens: tuple[int, int] | None = (1200, 150),
    questions: int = 40,
    drop: tuple[str, ...] = (),
) -> None:
    generation = {
        "provider": PROFILE.provider,
        "model": model,
        "prompt_version": prompt_version,
        "grounding_check_version": "grounding-check-v1",
        "response_format": "json_object",
        "max_prompt_tokens": 6000,
        "output_reserve_tokens": 800,
    }
    for key in drop:
        generation.pop(key)
    calls = [
        call(tokens=tokens).fields(),
        call(ProviderCallType.GROUNDING_CHECK, (800, 10)).fields(),
    ]
    payload = {
        "format": "e2e-eval-v3",
        "run": {
            "generated_at": f"{made.isoformat()}T08:00:00+00:00",
            "generation": generation,
            "retrieval_policy": {"top_k": 5, "min_similarity": 0.25},
        },
        "questions": [{"id": f"q-{n}", "provider_calls": calls} for n in range(questions)],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _measured() -> UsageHistory:
    calls = tuple(call() for _ in range(40)) + tuple(
        call(ProviderCallType.GROUNDING_CHECK, (800, 10)) for _ in range(40)
    )
    return UsageHistory(calls=calls, questions=40, sources=("x.json",), newest=TODAY)


def _plan(
    *, history: UsageHistory, tier: Tier = Tier.ACCEPTANCE, questions: int = 49, known: float = 0
) -> Preflight:
    planned = preflight(
        tier=tier,
        questions=questions,
        provider=PROFILE.provider,
        model=PROFILE.model,
        profile=PROFILE,
        history=history,
        ledger=LedgerReading(entries=(), unreadable_lines=0),
        today=TODAY,
        policy=POLICY,
    )
    return replace(planned, known_consumption=known) if known else planned


def test_scenario_g_yesterdays_compatible_history_forecasts_today_without_a_smoke_run(
    tmp_path: Path,
):
    _export(tmp_path / "yesterday.json", made=TODAY - timedelta(days=1))

    history = load_history(tmp_path, REQUIREMENTS)
    planned = _plan(history=history)

    assert history.notes == ()
    assert planned.forecast is not None
    assert planned.forecast.confidence is Confidence.HIGH
    assert planned.forecast.profile_age_days == 1
    assert planned.zone is BudgetZone.GREEN
    assert planned.can_start(yellow_confirmed=False), "no run today is needed first"
    assert not any("smoke" in warning for warning in planned.warnings)


def test_scenario_h_another_models_history_is_not_used_at_all(tmp_path: Path):
    _export(tmp_path / "other.json", made=TODAY, model="@cf/meta/llama-3.3-70b-instruct-fp8-fast")

    planned = _plan(history=load_history(tmp_path, REQUIREMENTS))

    assert planned.forecast is not None
    assert planned.forecast.confidence is Confidence.LOW
    assert planned.forecast.input_tokens is None, "the structural bound, not foreign tokens"


def test_scenario_h_a_different_prompt_version_is_used_but_capped_and_named(tmp_path: Path):
    _export(tmp_path / "old-prompt.json", made=TODAY, prompt_version="grounded-answer-v5")

    planned = _plan(history=load_history(tmp_path, REQUIREMENTS))

    assert planned.forecast is not None
    assert planned.forecast.confidence is Confidence.MEDIUM
    assert any("prompt version" in note for note in planned.forecast.compatibility)
    assert any("history:" in warning for warning in planned.warnings)


def test_missing_metadata_is_not_discarded_but_lowers_confidence(tmp_path: Path):
    _export(tmp_path / "terse.json", made=TODAY, drop=("prompt_version", "response_format"))

    history = load_history(tmp_path, REQUIREMENTS)

    assert history.calls, "still used"
    assert any("does not record its prompt version" in note for note in history.notes)
    assert forecast_run(49, history, PROFILE, today=TODAY).confidence is Confidence.MEDIUM


def test_old_history_is_capped_at_medium(tmp_path: Path):
    _export(tmp_path / "old.json", made=TODAY - timedelta(days=30))

    forecast = forecast_run(49, load_history(tmp_path, REQUIREMENTS), PROFILE, today=TODAY)

    assert forecast.confidence is Confidence.MEDIUM
    assert forecast.profile_age_days == 30
    assert any("30 days old" in note for note in forecast.compatibility)


def test_history_spans_days_and_names_its_newest(tmp_path: Path):
    _export(tmp_path / "a.json", made=TODAY - timedelta(days=3), questions=20)
    _export(tmp_path / "b.json", made=TODAY - timedelta(days=1), questions=20)

    history = load_history(tmp_path, REQUIREMENTS)

    assert history.sources == ("a.json", "b.json")
    assert history.newest == TODAY - timedelta(days=1)
    assert history.questions == 40


def test_scenario_i_no_history_is_low_confidence_and_starts_nothing():
    planned = _plan(history=UsageHistory(calls=(), questions=0))

    assert planned.forecast is not None and planned.forecast.confidence is Confidence.LOW
    assert planned.zone is BudgetZone.RED
    assert not planned.can_start(yellow_confirmed=True)
    # Preflight is a pure function of files: it has no provider to call, and
    # no step in it starts a smoke run. (The CLI side: tests/integration/
    # test_cli_eval.py::test_acceptance_without_measured_usage_is_red_and_builds_nothing.)
    assert not any("smoke" in warning for warning in planned.warnings)


def test_scenario_j_missing_usage_lowers_coverage_and_is_never_zero(tmp_path: Path):
    _export(tmp_path / "full.json", made=TODAY, questions=10)
    _export(tmp_path / "gaps.json", made=TODAY, questions=10, tokens=None)

    history = load_history(tmp_path, REQUIREMENTS)
    forecast = forecast_run(49, history, PROFILE, today=TODAY)

    assert history.usage_coverage == pytest.approx(0.75)
    assert forecast.confidence is Confidence.LOW
    assert forecast.neurons > 0
    watching = RunGuard(preflight=_plan(history=_measured()), total_questions=49)
    watching.observe(replace(question_costing(50), provider_calls=(call(tokens=None),)))
    assert watching.unreported_calls == 1 and watching.estimated_neurons > 100


# --- robust statistics -------------------------------------------------------------------


def test_a_heavy_tail_is_not_filtered_away():
    """Three calls in twenty cost ten times as much — beyond the p75, so the
    capped mean, not the percentile, carries them."""
    values = [100.0] * 17 + [1_000.0] * 3

    assert robust_cost(values) == pytest.approx(235.0)


def test_a_tail_within_the_p75_is_carried_by_the_percentile():
    values = [100.0] * 7 + [400.0] * 3

    assert robust_cost(values) == 400.0


def test_one_extreme_call_in_a_large_sample_is_capped():
    values = [100.0] * 19 + [10_000.0]

    assert robust_cost(values) == pytest.approx(100.0)


def test_generation_and_grounding_are_costed_apart():
    forecast = forecast_run(49, _measured(), PROFILE, today=TODAY)

    assert forecast.generation_neurons > forecast.grounding_neurons > 0
    assert forecast.generation_neurons + forecast.grounding_neurons == pytest.approx(
        forecast.neurons
    )


# --- smoke --------------------------------------------------------------------------------


def _smoke(failing: int, questions: int = 5) -> Any:
    label = label_of("stack", ANSWERABLE)
    cut = ScriptedReply(raw_text='{"answer": "cut', finish_reason="length")
    replies = [cut] * (2 * failing) + [grounded(f"FastAPI [{label}].", label)]
    dataset = dataset_of(
        *(ANSWERABLE.model_copy(update={"id": f"q-{n}"}) for n in range(questions))
    )
    return run(run_e2e_evaluation(dataset, e2e_service(ScriptedLLMProvider(*replies))))


def test_one_technical_failure_in_five_passes_two_do_not():
    one, two = smoke_verdict(_smoke(1)), smoke_verdict(_smoke(2))

    assert one.passed
    assert not two.passed
    assert any("2 of 5" in reason for reason in two.reasons)


def test_a_smoke_run_is_forecast_and_held_to_its_own_share():
    expensive = _plan(history=UsageHistory(calls=(), questions=0), tier=Tier.SMOKE, questions=5)
    cheap = _plan(history=_measured(), tier=Tier.SMOKE, questions=5)

    assert expensive.policy.target_share == 0.10
    assert expensive.zone is BudgetZone.YELLOW
    assert cheap.zone is BudgetZone.GREEN


# --- the report ---------------------------------------------------------------------------


def test_the_report_carries_band_share_history_and_projection():
    planned = _plan(history=_measured())
    watching = RunGuard(preflight=planned, total_questions=1)
    dataset = dataset_of(ANSWERABLE)
    report = run(run_e2e_evaluation(dataset, e2e_service(ScriptedLLMProvider()), guard=watching))
    fields = planned.fields()
    operations = {
        **fields,
        "status": "complete",
        "observed": {"estimated_neurons": watching.estimated_neurons, "usage_coverage": 1.0},
        "known_daily_after": watching.estimated_neurons,
        "nominal_reserve_after": POLICY.daily_budget - watching.estimated_neurons,
        "guard": {
            "status": watching.status,
            "projected_final_neurons": watching.projected_final_neurons,
            "projected_final_share": 0.01,
            "projected_band": "target",
        },
        "smoke": None,
    }

    summary = render_summary(export_e2e(report, run_metadata(dataset), operations))

    assert fields["cost_band"] == "good" and fields["forecast_share"] == pytest.approx(
        0.402, abs=1e-3
    )
    assert fields["forecast"]["profile_sources"] == 1
    assert "GOOD" in summary
    assert "| History | 1 run(s), age 0 days, 0 compatibility note(s) |" in summary
    assert "| Budget guard |" in summary
    assert fields["actual_remaining"] is None
