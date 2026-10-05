"""Running an evaluation against a real provider without wasting its allocation.

Three tiers. **Local** is everything provable without a provider — the test
suite, lint, types, knowledge validation — and costs nothing. **Smoke** asks a
handful of chosen questions to find out whether the provider path is healthy
enough today to be worth a full run. **Acceptance** asks all of them, and only
when asked for by name. There is no default tier for a run that spends.

Around a provider run: a **preflight** before the first call — what it will
cost, conservatively, and whether the production reserve survives it
(`evaluation.budget`); a **guard** after every question — stop early when the
provider has said no more (429, refused credentials), when it keeps failing,
or when the run is heading past the reserve; and an entry in the local
**ledger** afterwards, however the run ended. Nothing here retries: the
adapters own transport retries and the answering service owns regeneration
(`rag.failure_policy`). A run that ends early keeps what it measured.

A failure the pipeline classified and handled — an answer truncated twice, a
grounding check with no verdict — is a *result*, not a reason to stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from statistics import median
from typing import Any, Final

from portfolio_rag.core.logging import get_logger
from portfolio_rag.evaluation.acceptance import AcceptanceAttempt, rerun_blockers
from portfolio_rag.evaluation.budget import (
    SMOKE_TARGET_SHARE,
    BudgetPolicy,
    BudgetZone,
    Confidence,
    CostBand,
    CostProfile,
    Forecast,
    LedgerEntry,
    LedgerReading,
    UsageHistory,
    budget_zone,
    cost_band,
    forecast_run,
    medium_confidence_gap,
    spent,
)
from portfolio_rag.evaluation.dataset import EvaluationSuite
from portfolio_rag.evaluation.e2e import (
    E2E_FORMAT_VERSION,
    FAILURE_CLASSES,
    AbortReason,
    E2ERecord,
    E2EReport,
    FailureClass,
)
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.errors import GenerationFailureCategory
from portfolio_rag.rag.policy import DEFAULT_CONTEXT_POLICY, ContextPolicy
from portfolio_rag.rag.service import MAX_GENERATION_ATTEMPTS, MAX_GROUNDING_CHECK_ATTEMPTS
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType

_logger = get_logger(__name__)


class Tier(StrEnum):
    """What a provider run is for. Local runs are not a tier of this command:
    they are the test suite, and they spend nothing."""

    SMOKE = "smoke"
    ACCEPTANCE = "acceptance"


#: The suite each tier asks, unless a smoke run names its questions. The
#: acceptance tier asks the curated release-acceptance suite; the whole
#: dataset (``full``) is extended validation and is never a tier's suite.
TIER_SUITES: Final = {
    Tier.SMOKE: EvaluationSuite.PROVIDER_SMOKE,
    Tier.ACCEPTANCE: EvaluationSuite.RELEASE_ACCEPTANCE,
}

#: A smoke run that grows past this is no longer a smoke run.
MAX_SMOKE_QUESTIONS: Final = 6

#: Technical failures that make a smoke run fail, however many questions it had.
_SMOKE_FAILURE_LIMIT: Final = 2

#: Consecutive questions that may fail on the provider's side before the run
#: is stopped. One is a result; two in a row is a provider that is not well.
_SYSTEMIC_STREAK: Final = 2

#: How fast the guard trusts the run's own cost over the forecast's: after n
#: questions the run's median cost per question weighs n / (n + this).
#: One question weighs a sixth, five weigh half, fifteen three quarters.
_GUARD_PRIOR_WEIGHT: Final = 5

#: The provider calls one question may make at most: every generation and
#: every grounding check the failure policy allows. Transport retries beneath
#: the port are the adapter's and are not port calls.
MAX_PORT_CALLS_PER_QUESTION: Final = MAX_GENERATION_ATTEMPTS + MAX_GROUNDING_CHECK_ATTEMPTS

#: The one budget rule an operator may overrule for a run, by name.
OVERRIDE_MINIMUM_RESERVE: Final = "minimum_reserve"

#: Questions observed before a *projection* may stop a run. A median of three
#: is not moved by one expensive question; a median of one is that question.
#: A run whose actual spend crosses the line is stopped regardless.
_GUARD_MIN_QUESTIONS: Final = 3

_TRANSIENT: Final = frozenset(
    {GenerationFailureCategory.TIMEOUT, GenerationFailureCategory.RETRYABLE_PROVIDER_ERROR}
)
_AUTH_STATUSES: Final = frozenset({401, 403})


# --- preflight ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Preflight:
    """Everything decided about a run before it spends anything."""

    tier: Tier
    questions: int
    provider: str
    model: str
    profile: CostProfile | None
    forecast: Forecast | None
    """``None`` for a provider not billed in neurons."""

    policy: BudgetPolicy
    known_consumption: float
    zone: BudgetZone
    blockers: tuple[str, ...]
    """Reasons the run may not start at all."""

    warnings: tuple[str, ...]
    max_port_calls: int
    """The ceiling the failure policy sets: two generations and two checks per
    question. The forecast is the expectation; this is the bound."""

    reserve_overridden: bool = False
    """The operator set the minimum reserve aside for this run, by name. The
    run may spend into it, never past the nominal daily budget."""

    reserve_kept: bool = True
    """Whether the forecast keeps the minimum reserve — what the zone would
    say without an override. ``True`` for a provider not billed in neurons."""

    decided_at: str | None = None
    """When the preflight was decided, UTC, ISO 8601."""

    @property
    def confirmation_required(self) -> bool:
        return self.zone is BudgetZone.YELLOW

    @property
    def estimated_reserve_after(self) -> float | None:
        if self.forecast is None:
            return None
        return self.policy.daily_budget - self.known_consumption - self.forecast.neurons

    @property
    def forecast_share(self) -> float | None:
        if self.forecast is None:
            return None
        return self.forecast.neurons / self.policy.daily_budget

    @property
    def band(self) -> CostBand | None:
        if self.forecast is None:
            return None
        return cost_band(self.forecast.neurons, self.policy.daily_budget)

    @property
    def projected_total(self) -> float | None:
        """Today's known spend plus this run's forecast."""
        if self.forecast is None:
            return None
        return self.known_consumption + self.forecast.neurons

    def override_fields(self) -> dict[str, Any]:
        """The audit record of the reserve override. Present on every run, so
        that an export says whether it was used, not only when it was."""
        return {
            "budget_override_used": self.reserve_overridden,
            "override_type": OVERRIDE_MINIMUM_RESERVE if self.reserve_overridden else None,
            "reserve_kept_without_override": self.reserve_kept,
            "known_local_today": round(self.known_consumption, 1),
            "forecast": round(self.forecast.neurons, 1) if self.forecast else None,
            "projected_total": (
                round(self.projected_total, 1) if self.projected_total is not None else None
            ),
            "nominal_daily_budget": self.policy.daily_budget,
            "minimum_reserve": self.policy.minimum_reserve,
            "decided_at": self.decided_at,
        }

    def can_start(self, *, yellow_confirmed: bool) -> bool:
        if self.blockers or self.zone is BudgetZone.RED:
            return False
        return self.zone is BudgetZone.GREEN or yellow_confirmed

    def fields(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "questions": self.questions,
            "provider": self.provider,
            "model": self.model,
            "cost_profile": self.profile.source if self.profile else None,
            "forecast": self.forecast.fields() if self.forecast else None,
            "max_port_calls": self.max_port_calls,
            "daily_budget": self.policy.daily_budget,
            "minimum_reserve": self.policy.minimum_reserve,
            "target_share": self.policy.target_share,
            "forecast_share": (
                round(self.forecast_share, 3) if self.forecast_share is not None else None
            ),
            "cost_band": self.band.value if self.band else None,
            "known_daily_before": round(self.known_consumption, 1),
            "estimated_reserve_after": (
                round(self.estimated_reserve_after, 1)
                if self.estimated_reserve_after is not None
                else None
            ),
            "actual_remaining": None,
            "zone": self.zone.value,
            "confirmation_required": self.confirmation_required,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "budget_override": self.override_fields(),
        }


def preflight(
    *,
    tier: Tier,
    questions: int,
    provider: str,
    model: str,
    profile: CostProfile | None,
    history: UsageHistory,
    ledger: LedgerReading,
    today: date,
    policy: BudgetPolicy,
    context_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
    commit: str | None = None,
    dirty: bool | None = None,
    rerun_of: str | None = None,
    source_identity: str | None = None,
    override_reserve: bool = False,
    decided_at: datetime | None = None,
) -> Preflight:
    """Decide, before a single call, whether a run may start and what it will cost.

    Learns from *history* — earlier runs of any day — and charges against
    today's *ledger* only. No run is required first: without comparable
    measured usage the forecast is the structural upper bound, and the
    decision follows from that.

    An acceptance run is also held to the rerun rule
    (:func:`~portfolio_rag.evaluation.acceptance.rerun_blockers`) against
    every acceptance run in the ledger, of any day: *commit* and *dirty* are
    the tree the run is about to start from, and *source_identity* the
    release candidate it holds.

    *override_reserve* sets the minimum reserve aside for this run and nothing
    else: the zone is then judged against the nominal daily budget, and every
    blocker — the rerun rule, a failed smoke run, an unreadable plan — stands.
    """
    stamp = decided_at.isoformat() if decided_at is not None else None
    if tier is Tier.SMOKE:
        # A health check is held to a health check's budget.
        policy = replace(policy, target_share=SMOKE_TARGET_SHARE)
    blockers: list[str] = []
    warnings: list[str] = []
    known = ledger.known_consumption(today)
    if ledger.unreadable_lines:
        warnings.append(
            f"{ledger.unreadable_lines} ledger line(s) unreadable; today's known consumption "
            "may be understated"
        )
    if tier is Tier.SMOKE and questions > MAX_SMOKE_QUESTIONS:
        blockers.append(
            f"a smoke run asks at most {MAX_SMOKE_QUESTIONS} questions, not {questions}"
        )
    if tier is Tier.ACCEPTANCE:
        last_smoke = _latest_smoke(ledger, today)
        if last_smoke is not None and last_smoke.smoke_passed is False:
            blockers.append(
                f"today's latest smoke run ({last_smoke.run_id}) failed: the provider path "
                "is not healthy enough for an acceptance run"
            )
        blockers += rerun_blockers(
            acceptance_attempts(ledger),
            commit=commit,
            dirty=dirty,
            rerun_of=rerun_of,
            source_identity=source_identity,
        )

    if profile is None:
        warnings.append(
            f"no neuron cost profile for {provider} / {model}: the budget does not apply"
        )
        return Preflight(
            tier=tier,
            questions=questions,
            provider=provider,
            model=model,
            profile=None,
            forecast=None,
            policy=policy,
            known_consumption=known,
            zone=BudgetZone.GREEN,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
            max_port_calls=MAX_PORT_CALLS_PER_QUESTION * questions,
            reserve_overridden=override_reserve,
            decided_at=stamp,
        )

    forecast = forecast_run(questions, history, profile, context_policy, today=today)
    zone = budget_zone(forecast.neurons, known, policy, reserve_overridden=override_reserve)
    reserve_kept = known + forecast.neurons <= policy.spendable
    if forecast.confidence is Confidence.LOW:
        warnings.append(
            "forecast confidence is low, so the forecast is the structural upper bound: "
            + medium_confidence_gap(history)
        )
    for note in forecast.compatibility:
        warnings.append(f"history: {note}")
    if tier is Tier.SMOKE and zone is BudgetZone.YELLOW:
        warnings.append(
            f"this smoke run is forecast at {forecast.neurons:,.0f} neurons "
            f"({forecast.neurons / policy.daily_budget:.0%} of the daily budget): operationally "
            f"expensive for a health check, above its {SMOKE_TARGET_SHARE:.0%}"
        )
    return Preflight(
        tier=tier,
        questions=questions,
        provider=provider,
        model=model,
        profile=profile,
        forecast=forecast,
        policy=policy,
        known_consumption=known,
        zone=zone,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        max_port_calls=MAX_PORT_CALLS_PER_QUESTION * questions,
        reserve_overridden=override_reserve,
        reserve_kept=reserve_kept,
        decided_at=stamp,
    )


def acceptance_attempts(ledger: LedgerReading) -> tuple[AcceptanceAttempt, ...]:
    """Every acceptance run the ledger records, of any day, for the rerun rule."""
    return tuple(
        AcceptanceAttempt(
            run_id=entry.run_id,
            commit_sha=entry.commit_sha,
            git_dirty=entry.git_dirty,
            status=entry.status,
            abort_reason=entry.abort_reason,
            failed_gates=tuple(entry.failed_gates) if entry.failed_gates is not None else None,
            source_identity=entry.release_source_identity,
        )
        for entry in ledger.entries
        if entry.tier == Tier.ACCEPTANCE.value
    )


def _latest_smoke(ledger: LedgerReading, today: date) -> LedgerEntry | None:
    smokes = [entry for entry in ledger.for_day(today) if entry.tier == Tier.SMOKE.value]
    return smokes[-1] if smokes else None


# --- history --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileRequirements:
    """What makes an earlier run's costs comparable to the run being forecast.

    Provider, model and export format must match, or the run is not used at
    all. The rest change what a call costs without making the history useless:
    a difference — or a field the export does not record — is noted, and the
    forecast's confidence is capped at medium.
    """

    provider: str
    model: str
    prompt_version: str
    grounding_check_version: str
    response_format: str
    max_prompt_tokens: int
    output_reserve_tokens: int
    top_k: int
    min_similarity: float

    def soft(self) -> tuple[tuple[str, tuple[str, str], object], ...]:
        return (
            ("prompt version", ("generation", "prompt_version"), self.prompt_version),
            (
                "grounding check version",
                ("generation", "grounding_check_version"),
                self.grounding_check_version,
            ),
            ("response format", ("generation", "response_format"), self.response_format),
            ("prompt budget", ("generation", "max_prompt_tokens"), self.max_prompt_tokens),
            ("output limit", ("generation", "output_reserve_tokens"), self.output_reserve_tokens),
            ("top_k", ("retrieval_policy", "top_k"), self.top_k),
            ("min_similarity", ("retrieval_policy", "min_similarity"), self.min_similarity),
        )


def load_history(directory: Path, requirements: ProfileRequirements) -> UsageHistory:
    """Every provider call of earlier comparable end-to-end runs, of any day.

    Only exports that recorded their calls (``e2e-eval-v3`` and later) carry
    usage; older ones are skipped. A file that cannot be read as an export is
    skipped too — history informs a forecast, it is never a reason to fail one.
    """
    calls: list[ProviderCallRecord] = []
    questions = 0
    sources: list[str] = []
    notes: list[str] = []
    newest: date | None = None
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("format") != E2E_FORMAT_VERSION:
            continue
        run = payload.get("run") or {}
        generation = run.get("generation") or {}
        if (
            generation.get("provider") != requirements.provider
            or generation.get("model") != requirements.model
        ):
            continue
        found = False
        for question in payload.get("questions") or []:
            parsed = [_record(call) for call in question.get("provider_calls") or []]
            records = [record for record in parsed if record is not None]
            if records:
                calls.extend(records)
                questions += 1
                found = True
        if found:
            sources.append(path.name)
            notes.extend(_differences(path.name, run, requirements))
            made = _run_date(run.get("generated_at"))
            if made is not None and (newest is None or made > newest):
                newest = made
            elif made is None:
                notes.append(f"{path.name} does not record when it ran")
    return UsageHistory(
        calls=tuple(calls),
        questions=questions,
        sources=tuple(sources),
        newest=newest,
        notes=tuple(notes),
    )


def _differences(name: str, run: dict[str, Any], requirements: ProfileRequirements) -> list[str]:
    found: list[str] = []
    for label, (section, key), expected in requirements.soft():
        value = (run.get(section) or {}).get(key)
        if value is None:
            found.append(f"{name} does not record its {label}")
        elif value != expected:
            found.append(f"{name} ran with {label} {value!r}, now {expected!r}")
    return found


def _run_date(value: Any) -> date | None:
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def _record(fields: dict[str, Any]) -> ProviderCallRecord | None:
    """Enough of an exported call to forecast from: its step, result and tokens."""
    try:
        return ProviderCallRecord(
            call_type=ProviderCallType(fields["call_type"]),
            attempt=int(fields["attempt"]),
            model=str(fields["model"]),
            response_format=ResponseFormat(fields["response_format"]),
            elapsed_seconds=float(fields["elapsed_seconds"]),
            result=CallResult(fields["result"]),
            finish_reason=fields.get("finish_reason"),
            input_tokens=_int_or_none(fields.get("input_tokens")),
            output_tokens=_int_or_none(fields.get("output_tokens")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --- during the run ---------------------------------------------------------------------------


class RunGuard:
    """Decides after each question whether the run goes on. Observes; never retries."""

    def __init__(
        self,
        *,
        preflight: Preflight,
        total_questions: int,
        context_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
    ) -> None:
        self._preflight = preflight
        self._total = total_questions
        self._context_policy = context_policy
        self._transient_streak = 0
        self._per_question: list[float] = []
        self.estimated_neurons = 0.0
        self.unreported_calls = 0
        self.questions_seen = 0
        self.projected_final_neurons: float | None = (
            preflight.forecast.neurons if preflight.forecast is not None else None
        )
        self.stopped_by_budget = False

    def observe(self, record: E2ERecord) -> AbortReason | None:
        """Account for one finished question; say whether to stop before the next."""
        self.questions_seen += 1
        profile = self._preflight.profile
        if profile is not None:
            cost, unreported = spent(record.provider_calls, profile, self._context_policy)
            self.estimated_neurons += cost
            self.unreported_calls += unreported
            self._per_question.append(cost)
            self._project()

        failure = record.error
        if failure is not None:
            if failure.status_code == 429 or failure.detail == "rate_limited":
                return AbortReason.RATE_LIMITED
            if failure.status_code in _AUTH_STATUSES:
                return AbortReason.AUTH_FAILURE
            if failure.category in _TRANSIENT:
                self._transient_streak += 1
                if self._transient_streak >= _SYSTEMIC_STREAK:
                    return AbortReason.SYSTEMIC_PROVIDER_FAILURE
            else:
                self._transient_streak = 0
        else:
            self._transient_streak = 0

        if self._over_budget():
            self.stopped_by_budget = True
            return AbortReason.BUDGET_GUARD
        return None

    @property
    def projected_band(self) -> CostBand | None:
        if self.projected_final_neurons is None:
            return None
        return cost_band(self.projected_final_neurons, self._preflight.policy.daily_budget)

    @property
    def status(self) -> str:
        """``stopped`` by the budget, or the cost band the run is heading for."""
        if self.stopped_by_budget:
            return "stopped"
        band = self.projected_band
        return band.value if band is not None else "not_budgeted"

    def _project(self) -> None:
        """Where the run's spend is heading: what it has spent, plus the rest at a
        per-question cost that moves from the forecast's towards the run's own.

        The run's own cost is the *median* of its questions — one expensive
        question does not set the rate for the forty after it, though its
        neurons are counted in full — and its weight grows with every question
        observed, so the first question barely moves the projection and a run
        that stays expensive dominates it within a handful.
        """
        forecast = self._preflight.forecast
        if forecast is None:
            return
        seen = len(self._per_question)
        weight = seen / (seen + _GUARD_PRIOR_WEIGHT)
        rate = weight * median(self._per_question) + (1 - weight) * forecast.per_question_neurons
        remaining = max(self._total - self.questions_seen, 0)
        projected = self.estimated_neurons + remaining * rate
        before = self.projected_band
        self.projected_final_neurons = projected
        if self.projected_band is CostBand.EXCESSIVE and before is not CostBand.EXCESSIVE:
            _logger.warning(
                "evaluation projected excessive",
                extra={"projected_neurons": round(projected), "questions_seen": seen},
            )

    def _over_budget(self) -> bool:
        """Whether finishing the run is now expected to eat into the reserve.

        Stops only on the reserve — not on a run that is merely over its
        forecast, and not on the cost band, which is reported. With the
        reserve overridden for this run, on the nominal daily budget instead.
        """
        if self.projected_final_neurons is None or self._preflight.profile is None:
            return False
        spendable = self._preflight.policy.ceiling(
            reserve_overridden=self._preflight.reserve_overridden
        )
        if self._preflight.known_consumption + self.estimated_neurons > spendable:
            return True
        if len(self._per_question) < _GUARD_MIN_QUESTIONS:
            return False
        return self._preflight.known_consumption + self.projected_final_neurons > spendable


# --- after a smoke run ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SmokeVerdict:
    passed: bool
    reasons: tuple[str, ...]
    """Why it failed, or what it passed despite."""

    def fields(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons)}


def smoke_verdict(report: E2EReport) -> SmokeVerdict:
    """Is the provider path healthy enough today to justify a full run?

    Technical health only — a smoke run is not a small acceptance run. A
    regeneration that recovered, a controlled refusal and a single classified
    failure pass (the last with a note). It fails on: an aborted run (429,
    refused credentials, a provider failing repeatedly, the budget, a defect),
    an unclassified failure, a refused request, anything published that must
    not be, or most questions failing.
    """
    failing: list[str] = []
    notes: list[str] = []
    if report.aborted is not None:
        failing.append(f"run aborted: {report.aborted.reason.value}")
    errored = [record for record in report.records if record.error is not None]
    for record in errored:
        failure = record.error
        if failure is None:  # pragma: no cover - filtered above
            continue
        if failure.category is GenerationFailureCategory.UNCLASSIFIED:
            failing.append(f"{record.question.id}: unclassified provider failure")
        elif failure.category is GenerationFailureCategory.PROVIDER_STATUS:
            failing.append(f"{record.question.id}: provider refused the request")
        else:
            notes.append(f"{record.question.id}: {failure.category.value} (classified)")
    if len(errored) >= _SMOKE_FAILURE_LIMIT:
        # A handful of questions is a small sample: two technical failures in
        # it are not a provider path to trust with a full run.
        failing.append(f"{len(errored)} of {len(report.records)} questions failed technically")
    unsafe = [
        record.question.id
        for record in report.records
        if any(FAILURE_CLASSES[failure] is FailureClass.SAFETY for failure in record.failures)
    ]
    if unsafe:
        failing.append(f"safety: {', '.join(unsafe)}")
    if failing:
        return SmokeVerdict(passed=False, reasons=tuple(failing))
    return SmokeVerdict(passed=True, reasons=tuple(notes))


def describe_preflight(
    preflight: Preflight, *, yellow_confirmed: bool
) -> tuple[tuple[str, str], ...]:
    """Label/value pairs for the terminal."""
    forecast = preflight.forecast
    rows: list[tuple[str, str]] = [
        ("tier", preflight.tier.value),
        ("questions", str(preflight.questions)),
        ("provider / model", f"{preflight.provider} / {preflight.model}"),
        ("max port calls", str(preflight.max_port_calls)),
    ]
    if forecast is not None:
        rows += [
            ("generation calls", f"{forecast.generation_calls:.0f}"),
            ("grounding calls", f"{forecast.grounding_calls:.0f}"),
            ("regeneration calls", f"{forecast.regeneration_calls:.1f}"),
            ("input tokens", _tokens(forecast.input_tokens)),
            ("output tokens", _tokens(forecast.output_tokens)),
            ("generation neurons", f"{forecast.generation_neurons:,.0f}"),
            ("grounding neurons", f"{forecast.grounding_neurons:,.0f}"),
            ("estimated neurons", f"{forecast.neurons:,.0f} (estimate)"),
            ("forecast share", f"{forecast.neurons / preflight.policy.daily_budget:.0%} of daily"),
            ("cost band", preflight.band.value.upper() if preflight.band else "—"),
            ("forecast confidence", forecast.confidence.value),
            ("forecast method", forecast.method),
            ("history runs", str(forecast.profile_sources)),
            (
                "history age",
                "—" if forecast.profile_age_days is None else f"{forecast.profile_age_days} days",
            ),
            (
                "usage coverage",
                "—" if forecast.usage_coverage is None else f"{forecast.usage_coverage:.0%}",
            ),
        ]
    reserve = f"{preflight.policy.minimum_reserve:,.0f}"
    if preflight.reserve_overridden:
        reserve += " (manually overridden for this run)"
    rows += [
        ("daily nominal budget", f"{preflight.policy.daily_budget:,.0f}"),
        ("minimum reserve", reserve),
        ("known local today", f"{preflight.known_consumption:,.0f}"),
        ("actual remaining", "unknown (not visible to this tool)"),
    ]
    if preflight.estimated_reserve_after is not None:
        rows.append(("est. nominal reserve after", f"{preflight.estimated_reserve_after:,.0f}"))
    if preflight.projected_total is not None:
        rows.append(("projected total today", f"{preflight.projected_total:,.0f}"))
    rows.append(
        ("budget override", OVERRIDE_MINIMUM_RESERVE if preflight.reserve_overridden else "none")
    )
    rows += [
        ("budget zone", preflight.zone.value.upper()),
        ("confirmation required", "yes" if preflight.confirmation_required else "no"),
        ("can start", "yes" if preflight.can_start(yellow_confirmed=yellow_confirmed) else "NO"),
    ]
    return tuple(rows)


def _tokens(value: float | None) -> str:
    return "— (structural bound)" if value is None else f"{value:,.0f}"
