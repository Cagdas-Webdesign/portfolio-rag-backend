"""Forecast history: which calls count, how coverage is computed, and what it decides.

Built on the shape of the four calibration runs of 2026-10-04: two smoke runs
whose three generations failed before reaching the provider (no usage), and
two that answered — four generations and four grounding checks, every one
with reported usage. The forecast must read all four, treat the failed calls
as unknown cost rather than free, and say exactly why the result is still a
low-confidence forecast.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.evaluation.budget import (
    BudgetPolicy,
    BudgetZone,
    Confidence,
    LedgerReading,
    UsageHistory,
    forecast_run,
    medium_confidence_gap,
)
from portfolio_rag.evaluation.operations import Tier, load_history, preflight
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType
from tests.unit.test_eval_operations import PROFILE, REQUIREMENTS

TODAY = date(2026, 10, 4)
GEN, CHECK = ProviderCallType.GENERATION, ProviderCallType.GROUNDING_CHECK


def _call(
    call_type: ProviderCallType, tokens: tuple[int, int] | None, result: CallResult
) -> dict[str, Any]:
    return ProviderCallRecord(
        call_type=call_type,
        attempt=1,
        model=PROFILE.model,
        response_format=ResponseFormat.JSON_OBJECT,
        elapsed_seconds=1.0,
        result=result,
        input_tokens=tokens[0] if tokens else None,
        output_tokens=tokens[1] if tokens else None,
    ).fields()


def _failed() -> dict[str, Any]:
    return _call(GEN, None, CallResult.PROVIDER_ERROR)


def _answered(generation: tuple[int, int], check: tuple[int, int]) -> list[dict[str, Any]]:
    return [_call(GEN, generation, CallResult.PARSED), _call(CHECK, check, CallResult.PARSED)]


def _export(
    path: Path,
    questions: list[list[dict[str, Any]]],
    *,
    tier: str = "smoke",
    selected: tuple[str, ...] = ("q0",),
    **generation: Any,
) -> None:
    """An e2e-eval-v3 export with the run fields the compatibility check reads."""
    payload = {
        "format": "e2e-eval-v3",
        "run": {
            "generated_at": "2026-10-04T12:00:00+00:00",
            "tier": tier,
            "question_count": len(questions),
            "selected_question_ids": list(selected),
            "generation": {
                "provider": PROFILE.provider,
                "model": PROFILE.model,
                "prompt_version": "grounded-answer-v6",
                "grounding_check_version": "grounding-check-v1",
                "response_format": "json_object",
                "max_prompt_tokens": 6000,
                "output_reserve_tokens": 800,
                **generation,
            },
            "retrieval_policy": {"top_k": 5, "min_similarity": 0.25, "visibility": "public"},
        },
        "questions": [
            {"id": f"q{i}", "provider_calls": calls} for i, calls in enumerate(questions)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def calibration(tmp_path: Path) -> Path:
    """The four calibration runs of 2026-10-04, by shape and by reported tokens."""
    _export(tmp_path / "provider-calibration-1q.json", [[_failed()]])
    _export(tmp_path / "provider-calibration-2q.json", [[_failed()], [_failed()]])
    _export(
        tmp_path / "provider-calibration-after-token-fix.json",
        [_answered((1261, 344), (660, 317))],
    )
    _export(
        tmp_path / "provider-calibration-3q.json",
        [
            _answered((1083, 218), (548, 164)),
            _answered((1342, 522), (1037, 411)),
            _answered((1339, 303), (599, 304)),
        ],
    )
    return tmp_path


def test_every_compatible_run_is_read_and_coverage_is_counted_per_call(calibration: Path):
    history = load_history(calibration, REQUIREMENTS)

    assert len(history.sources) == 4
    assert history.notes == ()  # smoke tier, selected ids and run size are not mismatches
    assert len(history.calls) == 11
    assert sum(1 for c in history.calls if c.usage_reported) == 8
    assert history.usage_coverage == pytest.approx(8 / 11)


def test_generation_and_grounding_are_counted_apart(calibration: Path):
    history = load_history(calibration, REQUIREMENTS)
    generations = [c for c in history.calls if c.call_type is GEN]
    checks = [c for c in history.calls if c.call_type is CHECK]

    assert (len(generations), sum(c.usage_reported for c in generations)) == (7, 4)
    assert (len(checks), sum(c.usage_reported for c in checks)) == (4, 4)


def test_a_failed_call_without_usage_stays_unknown_never_zero(calibration: Path):
    history = load_history(calibration, REQUIREMENTS)
    failed = [c for c in history.calls if c.result is CallResult.PROVIDER_ERROR]

    assert len(failed) == 3
    assert all(c.input_tokens is None and c.output_tokens is None for c in failed)
    assert not any(c.usage_reported for c in failed)


def test_four_measured_calls_per_kind_and_73_percent_is_still_low(calibration: Path):
    history = load_history(calibration, REQUIREMENTS)

    forecast = forecast_run(24, history, PROFILE, today=TODAY)

    assert forecast.confidence is Confidence.LOW
    assert forecast.input_tokens is None  # the structural bound, not a measured guess
    assert medium_confidence_gap(history) == (
        "calls with measured usage generation 4/5, grounding_check 4/5 needed; "
        "usage coverage 73% (8/11 calls) of 80% needed"
    )


def test_one_more_answered_question_is_not_yet_enough(calibration: Path):
    _export(calibration / "next-1.json", [_answered((1200, 300), (700, 300))])
    history = load_history(calibration, REQUIREMENTS)

    assert history.usage_coverage == pytest.approx(10 / 13)
    assert forecast_run(24, history, PROFILE, today=TODAY).confidence is Confidence.LOW


def test_two_more_answered_questions_reach_medium_from_measured_usage(calibration: Path):
    _export(
        calibration / "next-2.json",
        [_answered((1200, 300), (700, 300)), _answered((1300, 400), (650, 250))],
    )
    history = load_history(calibration, REQUIREMENTS)

    forecast = forecast_run(24, history, PROFILE, today=TODAY)

    assert history.usage_coverage == pytest.approx(12 / 15)
    assert forecast.confidence is Confidence.MEDIUM
    assert forecast.input_tokens is not None
    assert "+25%" in forecast.method


def test_a_soft_mismatch_is_used_and_named_a_hard_one_is_not_used(tmp_path: Path):
    _export(tmp_path / "soft.json", [_answered((1000, 200), (500, 100))], prompt_version="v5")
    _export(
        tmp_path / "hard.json",
        [_answered((1000, 200), (500, 100))],
        model="@cf/meta/llama-3.1-8b-instruct",
    )

    history = load_history(tmp_path, REQUIREMENTS)

    assert history.sources == ("soft.json",)
    assert any("prompt version" in note for note in history.notes)


def test_smoke_history_counts_for_an_acceptance_forecast(tmp_path: Path):
    _export(tmp_path / "smoke.json", [_answered((1000, 200), (500, 100))], tier="smoke")
    _export(tmp_path / "accept.json", [_answered((1000, 200), (500, 100))], tier="acceptance")

    history = load_history(tmp_path, REQUIREMENTS)

    assert len(history.sources) == 2 and history.notes == ()


def test_the_reserve_still_decides_and_the_preflight_names_the_gap(calibration: Path):
    plan = preflight(
        tier=Tier.ACCEPTANCE,
        questions=24,
        provider=PROFILE.provider,
        model=PROFILE.model,
        profile=PROFILE,
        history=load_history(calibration, REQUIREMENTS),
        ledger=LedgerReading(entries=(), unreadable_lines=0),
        today=TODAY,
        policy=BudgetPolicy(),
    )

    assert plan.zone is BudgetZone.RED
    assert not plan.can_start(yellow_confirmed=True)
    assert any("generation 4/5" in w and "73% (8/11 calls)" in w for w in plan.warnings)
    assert not any("no comparable measured usage" in w for w in plan.warnings)


def test_no_history_says_so():
    empty = UsageHistory(calls=(), questions=0)
    assert medium_confidence_gap(empty) == "no comparable provider calls in the history"
