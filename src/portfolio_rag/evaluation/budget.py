"""What a provider run will cost, whether it may start, and what runs have cost.

Workers AI bills in *neurons* against a free daily allocation, and reports
none of them: a response carries token usage at best. Everything here is
therefore an **estimate** — reported tokens times a published rate — and is
named as one. Nothing in this module can see the provider's own account, so
it never claims to know the actual remaining allocation; it knows the nominal
daily budget and what *this tool* has spent today, and says so.

**The forecast learns from runs that measured themselves.** End-to-end
exports since ``e2e-eval-v3`` carry every provider call with its reported
tokens (`rag.telemetry`). From those, per call type: the 75th percentile of
input and output tokens, the rate of regeneration, and how many calls reported
usage at all. A forecast built on enough of them gets a margin for its
confidence; one without enough falls back to a structural upper bound — the
context budget and output limit the pipeline actually enforces — which is
deliberately pessimistic. A run that cannot be forecast well enough to protect
the reserve is not started; a small smoke run is how the data gets measured.

**The reserve is the rule, the 50 % is a target.** A run whose forecast keeps
the minimum production reserve may start; one above half the daily budget
needs an explicit go-ahead; one that would eat into the reserve does not start.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from portfolio_rag.rag.policy import DEFAULT_CONTEXT_POLICY, ContextPolicy
from portfolio_rag.rag.telemetry import ProviderCallRecord, ProviderCallType

#: The free daily allocation, as published. Nominal: what is actually left
#: today is not visible from here.
DAILY_BUDGET_NEURONS: Final = 10_000.0

#: Neurons that must remain for the public chatbot after a run. Below half the
#: budget on purpose: with a 5,000 reserve the 50 % target would be a hard
#: limit in disguise.
MINIMUM_RESERVE_NEURONS: Final = 4_000.0

#: Share of the daily budget an acceptance run should stay within. A target: a
#: run above it needs a go-ahead, it is not refused for it.
TARGET_SHARE: Final = 0.5

#: Share of the daily budget a smoke run should stay within. A smoke run is a
#: health check, not a small acceptance run: one forecast above this needs the
#: same explicit go-ahead.
SMOKE_TARGET_SHARE: Final = 0.10

#: History older than this still informs a forecast, at no more than medium
#: confidence: the provider's behaviour may have moved.
PROFILE_MAX_AGE_DAYS: Final = 14


@dataclass(frozen=True, slots=True)
class CostProfile:
    """Published neuron rates of one model. Estimated cost, not provider-measured."""

    provider: str
    model: str
    input_neurons_per_million: float
    output_neurons_per_million: float
    source: str

    def estimate(self, input_tokens: float, output_tokens: float) -> float:
        return (
            input_tokens * self.input_neurons_per_million
            + output_tokens * self.output_neurons_per_million
        ) / 1_000_000


#: The one place the rates live. From Cloudflare's Workers AI price list
#: (https://developers.cloudflare.com/workers-ai/platform/pricing/), read
#: 2026-10-03: $0.350 / M input tokens = 31,818 neurons, $0.750 / M output
#: tokens = 68,182 neurons.
COST_PROFILES: Final = (
    CostProfile(
        provider="cloudflare_workers_ai",
        model="@cf/openai/gpt-oss-120b",
        input_neurons_per_million=31_818,
        output_neurons_per_million=68_182,
        source="https://developers.cloudflare.com/workers-ai/platform/pricing/ (2026-10-03)",
    ),
)


def cost_profile(provider: str, model: str) -> CostProfile | None:
    """The rates of *model*, or ``None`` for a provider not billed in neurons."""
    for profile in COST_PROFILES:
        if profile.provider == provider and profile.model == model:
            return profile
    return None


def call_neurons(call: ProviderCallRecord, profile: CostProfile) -> float | None:
    """Estimated neurons of one call, or ``None`` when it reported no usage."""
    if call.input_tokens is None or call.output_tokens is None:
        return None
    return profile.estimate(call.input_tokens, call.output_tokens)


def structural_call_bound(profile: CostProfile, policy: ContextPolicy) -> float:
    """The most one call can cost under the limits the pipeline enforces.

    The prompt budget less the output reserve is the largest input the context
    builder will compose — measured with a deliberately pessimistic token
    estimate, so real counts stay under it — and the output reserve is the
    ``max_tokens`` the provider is asked to honour.
    """
    return profile.estimate(
        policy.max_prompt_tokens - policy.output_reserve_tokens, policy.output_reserve_tokens
    )


# --- forecast -----------------------------------------------------------------------


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    """Too little measured usage: the forecast is the structural upper bound."""


#: Measured calls per call type, and the share of calls that reported usage,
#: needed for each confidence. Below MEDIUM the history is not used at all.
_HIGH_SAMPLES: Final = 30
_HIGH_COVERAGE: Final = 0.95
_MEDIUM_SAMPLES: Final = 5
_MEDIUM_COVERAGE: Final = 0.80

#: Added on top of a history-based forecast, by confidence.
_MARGIN: Final = {Confidence.HIGH: 0.10, Confidence.MEDIUM: 0.25}

#: Regenerations assumed per answered question when nothing is measured.
_FALLBACK_REGENERATION_RATE: Final = 0.2


@dataclass(frozen=True, slots=True)
class UsageHistory:
    """The historical cost profile: provider calls of earlier comparable runs.

    The forecast's learning base — any day's runs, not only today's. Separate
    from the ledger, which is today's consumption and nothing else.
    """

    calls: tuple[ProviderCallRecord, ...]
    questions: int
    """Questions that made at least one provider call."""

    sources: tuple[str, ...] = ()
    newest: date | None = None
    """When the most recent source run was made."""

    notes: tuple[str, ...] = ()
    """Where the sources differ from the run being forecast, or say too little
    to tell: a different prompt version, retrieval policy, missing metadata.
    Any note caps the confidence at medium."""

    @property
    def usage_coverage(self) -> float | None:
        if not self.calls:
            return None
        return sum(1 for call in self.calls if call.usage_reported) / len(self.calls)


@dataclass(frozen=True, slots=True)
class Forecast:
    """The expected cost of one run, conservatively."""

    questions: int
    generation_calls: float
    grounding_calls: float
    regeneration_calls: float
    input_tokens: float | None
    output_tokens: float | None
    """``None`` when the forecast is the structural bound rather than measured."""

    neurons: float
    per_question_neurons: float
    confidence: Confidence
    method: str
    usage_coverage: float | None
    generation_neurons: float = 0.0
    grounding_neurons: float = 0.0
    profile_sources: int = 0
    profile_age_days: int | None = None
    compatibility: tuple[str, ...] = ()

    def fields(self) -> dict[str, Any]:
        return {
            key: (round(value, 1) if isinstance(value, float) else value)
            for key, value in asdict(self).items()
        }


def percentile(values: Sequence[float], share: float) -> float:
    """Nearest-rank percentile: a value that actually occurred, never interpolated."""
    ordered = sorted(values)
    rank = max(1, math.ceil(share * len(ordered)))
    return ordered[rank - 1]


def forecast_run(
    questions: int,
    history: UsageHistory,
    profile: CostProfile,
    policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
    *,
    today: date | None = None,
) -> Forecast:
    """Forecast the cost of asking *questions* questions end to end.

    Generation and grounding check are costed separately. Every question is
    assumed to reach the check (an upper bound: a refused answer is never
    checked) and to be regenerated at the rate the history shows. Per call and
    per token direction: the larger of the 75th percentile and the mean with
    values capped at the 90th — the first keeps a typical call honest, the
    second keeps a heavy tail from being filtered away, and the cap keeps one
    extreme call from setting the whole forecast. Then a margin for the
    confidence the history earns. Calls that reported no usage are not counted
    as cheap: they lower the coverage, and with it the confidence.
    """
    by_type = {
        call_type: [call for call in history.calls if call.call_type is call_type]
        for call_type in ProviderCallType
    }
    measured = {
        call_type: [call for call in calls if call.usage_reported]
        for call_type, calls in by_type.items()
    }
    coverage = history.usage_coverage
    age = (today - history.newest).days if today and history.newest else None
    stale = age is not None and age > PROFILE_MAX_AGE_DAYS
    confidence = _confidence(measured, coverage)
    if confidence is Confidence.HIGH and (history.notes or stale):
        confidence = Confidence.MEDIUM
    compatibility = history.notes + ((f"newest source is {age} days old",) if stale else ())

    if confidence is Confidence.LOW:
        per_call = structural_call_bound(profile, policy)
        generations = 1 + _FALLBACK_REGENERATION_RATE
        per_question = (generations + 1) * per_call
        return Forecast(
            questions=questions,
            generation_calls=questions * generations,
            grounding_calls=float(questions),
            regeneration_calls=questions * _FALLBACK_REGENERATION_RATE,
            input_tokens=None,
            output_tokens=None,
            neurons=questions * per_question,
            per_question_neurons=per_question,
            confidence=confidence,
            method=(
                "structural upper bound: too little comparable measured usage; every call "
                "at the context budget and output limit"
            ),
            usage_coverage=coverage,
            generation_neurons=questions * generations * per_call,
            grounding_neurons=questions * per_call,
            profile_sources=len(history.sources),
            profile_age_days=age,
            compatibility=compatibility,
        )

    generations = len(by_type[ProviderCallType.GENERATION]) / max(history.questions, 1)
    generation_in, generation_out = _per_call(measured[ProviderCallType.GENERATION])
    check_in, check_out = _per_call(measured[ProviderCallType.GROUNDING_CHECK])
    margin = 1 + _MARGIN[confidence]
    generation_cost = margin * generations * profile.estimate(generation_in, generation_out)
    check_cost = margin * profile.estimate(check_in, check_out)
    per_question = generation_cost + check_cost
    return Forecast(
        questions=questions,
        generation_calls=questions * generations,
        grounding_calls=float(questions),
        regeneration_calls=questions * max(generations - 1, 0.0),
        input_tokens=questions * margin * (generations * generation_in + check_in),
        output_tokens=questions * margin * (generations * generation_out + check_out),
        neurons=questions * per_question,
        per_question_neurons=per_question,
        confidence=confidence,
        method=(
            f"max(p75, p90-capped mean) tokens per call from {len(history.calls)} calls in "
            f"{len(history.sources)} run(s), observed generations per question, a check for "
            f"every question, +{_MARGIN[confidence]:.0%}"
        ),
        usage_coverage=coverage,
        generation_neurons=questions * generation_cost,
        grounding_neurons=questions * check_cost,
        profile_sources=len(history.sources),
        profile_age_days=age,
        compatibility=compatibility,
    )


def medium_confidence_gap(history: UsageHistory) -> str:
    """What the history lacks for a medium-confidence forecast, in its own numbers.

    Descriptive only — the decision is :func:`_confidence`'s. Says how many
    calls of each kind reported usage against the number required, and the
    usage coverage against the share required, so a low forecast names its gap
    instead of implying there is no history at all.
    """
    if not history.calls:
        return "no comparable provider calls in the history"
    counts = ", ".join(
        f"{call_type.value} "
        f"{sum(1 for c in history.calls if c.call_type is call_type and c.usage_reported)}"
        f"/{_MEDIUM_SAMPLES}"
        for call_type in ProviderCallType
    )
    coverage = history.usage_coverage or 0.0
    reported = sum(1 for call in history.calls if call.usage_reported)
    return (
        f"calls with measured usage {counts} needed; usage coverage {coverage:.0%} "
        f"({reported}/{len(history.calls)} calls) of {_MEDIUM_COVERAGE:.0%} needed"
    )


def _confidence(
    measured: dict[ProviderCallType, list[ProviderCallRecord]], coverage: float | None
) -> Confidence:
    samples = min(len(calls) for calls in measured.values())
    if coverage is None:
        return Confidence.LOW
    if samples >= _HIGH_SAMPLES and coverage >= _HIGH_COVERAGE:
        return Confidence.HIGH
    if samples >= _MEDIUM_SAMPLES and coverage >= _MEDIUM_COVERAGE:
        return Confidence.MEDIUM
    return Confidence.LOW


def robust_cost(values: Sequence[float]) -> float:
    """The larger of the 75th percentile and the mean capped at the 90th.

    Never below what a typical call costs, never hiding a heavy tail, and never
    set by one extreme value in a sample large enough to have a 90th
    percentile below its maximum.
    """
    cap = percentile(values, 0.90)
    capped_mean = sum(min(value, cap) for value in values) / len(values)
    return max(percentile(values, 0.75), capped_mean)


def _per_call(calls: Sequence[ProviderCallRecord]) -> tuple[float, float]:
    return (
        robust_cost([float(call.input_tokens or 0) for call in calls]),
        robust_cost([float(call.output_tokens or 0) for call in calls]),
    )


# --- may it start -----------------------------------------------------------------------


class CostBand(StrEnum):
    """How expensive a run is, as a share of the daily budget. Descriptive,
    except that an excessive run is never started."""

    TARGET = "target"
    """At most 40 %: where an acceptance run should be."""

    GOOD = "good"
    """At most 50 %."""

    CAUTION = "caution"
    """At most 70 %: expensive; needs an explicit go-ahead."""

    EXCESSIVE = "excessive"
    """Above 70 %: not acceptable for a normal run."""


_BANDS: Final = ((0.40, CostBand.TARGET), (0.50, CostBand.GOOD), (0.70, CostBand.CAUTION))


def cost_band(neurons: float, daily_budget: float) -> CostBand:
    share = neurons / daily_budget
    for limit, band in _BANDS:
        if share <= limit:
            return band
    return CostBand.EXCESSIVE


class BudgetZone(StrEnum):
    GREEN = "green"
    """Within the target, and the reserve is kept: start."""

    YELLOW = "yellow"
    """Above the target, the reserve is still kept: start only on an explicit go."""

    RED = "red"
    """The reserve would not be kept: do not start."""


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    daily_budget: float = DAILY_BUDGET_NEURONS
    minimum_reserve: float = MINIMUM_RESERVE_NEURONS
    target_share: float = TARGET_SHARE

    def __post_init__(self) -> None:
        if self.daily_budget <= 0 or self.minimum_reserve < 0:
            raise ValueError("the daily budget must be positive and the reserve not negative")
        if self.minimum_reserve >= self.daily_budget:
            raise ValueError("the minimum reserve must be smaller than the daily budget")
        if not 0 < self.target_share <= 1:
            raise ValueError("the target share must be within (0, 1]")

    @property
    def spendable(self) -> float:
        """What may be spent in a day before the reserve is touched."""
        return self.daily_budget - self.minimum_reserve


def budget_zone(
    forecast_neurons: float, known_consumption: float, policy: BudgetPolicy
) -> BudgetZone:
    """Where a run with this forecast stands today.

    RED when the reserve would not survive it, or when it alone would take more
    than 70 % of the daily budget; YELLOW above the target share; GREEN
    otherwise. A forecast exactly at a boundary is on the cheaper side of it,
    and a reserve kept exactly is kept.
    """
    remaining = policy.daily_budget - known_consumption - forecast_neurons
    if remaining < policy.minimum_reserve:
        return BudgetZone.RED
    if cost_band(forecast_neurons, policy.daily_budget) is CostBand.EXCESSIVE:
        return BudgetZone.RED
    if forecast_neurons > policy.target_share * policy.daily_budget:
        return BudgetZone.YELLOW
    return BudgetZone.GREEN


# --- what has been spent ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One provider run this tool made. What it cost, as far as it could tell."""

    date_utc: str
    run_id: str
    tier: str
    status: str
    """``complete`` or ``aborted``."""

    calls: int
    input_tokens: int
    output_tokens: int
    estimated_neurons: float
    """Calls that reported usage, at the published rates, plus calls that did
    not at the structural upper bound. Conservative, and still an estimate."""

    usage_coverage: float | None
    artifact: str | None
    abort_reason: str | None = None
    smoke_passed: bool | None = None
    commit_sha: str | None = None
    """The commit the run was made from. ``None`` when unknown, and for
    entries written before it was recorded."""

    git_dirty: bool | None = None
    rerun_of: str | None = None
    """The run this one repeated under the provider-outlier rule."""

    failed_gates: list[str] | None = None
    """The gates the run broke; empty when it broke none. ``None`` for entries
    written before gates were recorded."""

    release_source_identity: str | None = None
    """The release candidate the run tested — its release-relevant content.
    ``None`` for entries written before it was recorded."""


@dataclass(frozen=True, slots=True)
class LedgerReading:
    entries: tuple[LedgerEntry, ...]
    unreadable_lines: int
    """Lines that were not an entry. Counted, never silently dropped."""

    def for_day(self, day: date) -> tuple[LedgerEntry, ...]:
        return tuple(entry for entry in self.entries if entry.date_utc == day.isoformat())

    def known_consumption(self, day: date) -> float:
        """What this tool has spent on *day*, by its own estimates. Not the
        provider's account: other use of the same allocation is invisible here."""
        return sum(entry.estimated_neurons for entry in self.for_day(day))


class Ledger:
    """An append-only JSON Lines file of the provider runs this tool made."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> LedgerReading:
        """Every readable entry. A missing or empty file is an empty ledger; a
        line that is not an entry is counted and skipped, so one damaged line
        cannot hide the runs around it."""
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return LedgerReading(entries=(), unreadable_lines=0)
        entries: list[LedgerEntry] = []
        unreadable = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                entries.append(LedgerEntry(**json.loads(line)))
            except (ValueError, TypeError):
                unreadable += 1
        return LedgerReading(entries=tuple(entries), unreadable_lines=unreadable)

    def append(self, entry: LedgerEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def spent(
    calls: Iterable[ProviderCallRecord], profile: CostProfile, policy: ContextPolicy
) -> tuple[float, int]:
    """Estimated neurons of *calls*, and how many reported no usage.

    A call without usage is counted at the structural upper bound — never as
    free.
    """
    total = 0.0
    unreported = 0
    for call in calls:
        cost = call_neurons(call, profile)
        if cost is None:
            unreported += 1
            cost = structural_call_bound(profile, policy)
        total += cost
    return total, unreported
