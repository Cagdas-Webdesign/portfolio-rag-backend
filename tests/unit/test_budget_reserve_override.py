"""The minimum-reserve override: one rule set aside for one run, nothing else.

Without the flag, every decision is what it was. With it, a run may spend into
the 4,000-neuron reserve this tool keeps on top of the provider's allocation —
and still never past the nominal daily budget. The excessive-run limit, the
blockers (a failed smoke run, the rerun rule), and the guard's reactions to a
rate limit, refused credentials and a failing provider are untouched, and the
override is written into the run's audit record either way.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from portfolio_rag.evaluation.budget import (
    BudgetPolicy,
    BudgetZone,
    LedgerEntry,
    LedgerReading,
    budget_zone,
)
from portfolio_rag.evaluation.e2e import AbortReason
from portfolio_rag.evaluation.operations import (
    OVERRIDE_MINIMUM_RESERVE,
    Preflight,
    RunGuard,
    Tier,
    preflight,
)
from tests.unit.test_eval_operations import (
    GOOD_HISTORY,
    PROFILE,
    TODAY,
    C,
    entry,
    failure,
    record,
)

POLICY = BudgetPolicy()
DECIDED = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)


def plan_with(
    known: float, *, override: bool, ledger_extra: tuple[LedgerEntry, ...] = ()
) -> Preflight:
    return preflight(
        tier=Tier.ACCEPTANCE,
        questions=24,
        provider=PROFILE.provider,
        model=PROFILE.model,
        profile=PROFILE,
        history=GOOD_HISTORY,
        ledger=LedgerReading(entries=(entry(neurons=known), *ledger_extra), unreadable_lines=0),
        today=TODAY,
        policy=POLICY,
        override_reserve=override,
        decided_at=DECIDED,
    )


def forecast_of(plan: Preflight) -> float:
    assert plan.forecast is not None
    return plan.forecast.neurons


# --- the zone ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("forecast", "known"),
    [(4_200, 0), (5_300, 0), (6_000, 0), (6_001, 0), (2_000, 4_000), (2_001, 4_000), (7_001, 0)],
)
def test_without_the_override_the_zone_is_what_it_was(forecast: float, known: float):
    old = (
        BudgetZone.RED
        if POLICY.daily_budget - known - forecast < POLICY.minimum_reserve or forecast > 7_000
        else BudgetZone.YELLOW
        if forecast > 5_000
        else BudgetZone.GREEN
    )
    assert budget_zone(forecast, known, POLICY) is old


@pytest.mark.parametrize(
    ("forecast", "known", "zone"),
    [
        (2_001, 4_000, BudgetZone.GREEN),  # into the reserve, inside the nominal budget
        (4_000, 6_000, BudgetZone.GREEN),  # the nominal budget kept exactly
        (4_001, 6_000, BudgetZone.RED),  # the nominal budget missed by one
        (7_001, 0, BudgetZone.RED),  # an excessive run stays refused
        (5_300, 0, BudgetZone.YELLOW),  # the target share still asks for a go-ahead
    ],
)
def test_with_the_override_only_the_nominal_budget_must_survive(
    forecast: float, known: float, zone: BudgetZone
):
    assert budget_zone(forecast, known, POLICY, reserve_overridden=True) is zone


# --- the preflight -----------------------------------------------------------------------


def _known_that_eats_into_the_reserve() -> float:
    """Known spend today that leaves this forecast inside the nominal budget
    but not inside the reserve's edge."""
    forecast = forecast_of(plan_with(0, override=False))
    known = POLICY.spendable - forecast + 500
    assert known + forecast < POLICY.daily_budget
    return known


def test_the_reserve_blocks_a_run_without_the_override():
    plan = plan_with(_known_that_eats_into_the_reserve(), override=False)

    assert plan.zone is BudgetZone.RED
    assert not plan.can_start(yellow_confirmed=True)
    audit = plan.override_fields()
    assert audit["budget_override_used"] is False
    assert audit["override_type"] is None
    assert audit["reserve_kept_without_override"] is False


def test_the_override_lets_that_run_start_and_records_why():
    known = _known_that_eats_into_the_reserve()
    plan = plan_with(known, override=True)

    assert plan.zone is not BudgetZone.RED
    assert plan.can_start(yellow_confirmed=True)
    assert plan.fields()["budget_override"] == plan.override_fields()
    audit = plan.override_fields()
    assert audit == {
        "budget_override_used": True,
        "override_type": OVERRIDE_MINIMUM_RESERVE,
        "reserve_kept_without_override": False,
        "known_local_today": round(known, 1),
        "forecast": round(forecast_of(plan), 1),
        "projected_total": round(known + forecast_of(plan), 1),
        "nominal_daily_budget": POLICY.daily_budget,
        "minimum_reserve": POLICY.minimum_reserve,
        "decided_at": DECIDED.isoformat(),
    }


def test_the_override_never_passes_the_nominal_daily_budget():
    forecast = forecast_of(plan_with(0, override=False))
    plan = plan_with(POLICY.daily_budget - forecast + 1, override=True)

    assert plan.zone is BudgetZone.RED
    assert not plan.can_start(yellow_confirmed=True)


def test_the_override_leaves_every_blocker_standing():
    failed_smoke = entry(smoke_passed=False, run_id="s" * 32, neurons=0)
    plan = plan_with(0, override=True, ledger_extra=(failed_smoke,))

    assert any("smoke" in blocker for blocker in plan.blockers)
    assert not plan.can_start(yellow_confirmed=True)


# --- the guard ---------------------------------------------------------------------------


def _guard(known: float, *, override: bool, per_question: float) -> RunGuard:
    base = plan_with(known, override=override)
    assert base.forecast is not None
    base = replace(base, forecast=replace(base.forecast, per_question_neurons=per_question))
    return RunGuard(preflight=base, total_questions=24)


def test_the_guard_keeps_its_reaction_to_a_provider_saying_no_more():
    for bad, reason in (
        (failure(C.RETRYABLE_PROVIDER_ERROR, 429, "rate_limited"), AbortReason.RATE_LIMITED),
        (failure(C.PROVIDER_STATUS, 401), AbortReason.AUTH_FAILURE),
        (failure(C.PROVIDER_STATUS, 403), AbortReason.AUTH_FAILURE),
    ):
        assert _guard(0, override=True, per_question=1.0).observe(record(bad)) is reason

    guard = _guard(0, override=True, per_question=1.0)
    guard.observe(record(failure(C.TIMEOUT)))
    assert guard.observe(record(failure(C.TIMEOUT))) is AbortReason.SYSTEMIC_PROVIDER_FAILURE


def test_the_guard_lets_an_overridden_run_spend_into_the_reserve_but_not_past_the_budget():
    # Known spend just under the reserve's edge: every question crosses it.
    known = POLICY.spendable - 1
    kept = _guard(known, override=False, per_question=10.0)
    overridden = _guard(known, override=True, per_question=10.0)

    assert kept.observe(record()) is AbortReason.BUDGET_GUARD
    assert overridden.observe(record()) is None

    at_the_budget = _guard(POLICY.daily_budget - 1, override=True, per_question=10.0)
    assert at_the_budget.observe(record()) is AbortReason.BUDGET_GUARD
