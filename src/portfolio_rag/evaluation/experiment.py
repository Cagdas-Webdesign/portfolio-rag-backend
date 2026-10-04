"""A controlled comparison of provider request configurations.

Paket 3 asks one question of the generation provider at a time — for E1:
does asking for ``response_format=json_object`` change how often a reply is
unusable, compared with asking for nothing? Answering it honestly needs the
*only* difference between two calls to be that one setting. So:

* **Retrieval runs once per question.** Its context is built into one
  :class:`~portfolio_rag.ports.llm.GenerationRequest` with the same builders
  the answering service uses, and every variant and repetition of that
  question sends that request with nothing changed but ``response_format``.
  The context's fingerprint is exported so the claim can be checked.
* **One call is one request.** No regeneration and — wired by the CLI — no
  transport retry, so a call's cost and outcome belong to that call alone.
* **The real parser judges every reply,** and every call leaves a
  :class:`~portfolio_rag.rag.telemetry.ProviderCallRecord`, exactly as in
  production. Nothing of a prompt, a passage or a reply is kept.
* **It stops early rather than spend blind:** on a rate limit, refused
  credentials, any provider failure other than a malformed response (which is
  a measured outcome, not a fault of the run), the call limit, or an estimated
  neuron budget.

This is a measuring instrument for a developer. The server never imports it,
and nothing it finds changes the system until a decision has been made on it.
"""

from __future__ import annotations

import asyncio
import hashlib
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from portfolio_rag.evaluation.dataset import EvaluationQuestion
from portfolio_rag.evaluation.e2e import summarize_provider_calls
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationRequest, LLMProvider, ResponseFormat
from portfolio_rag.rag.context import build_context
from portfolio_rag.rag.errors import GenerationUnavailableError, provider_failure
from portfolio_rag.rag.generation import parse_generation
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.prompt import (
    GROUNDED_PROMPT_VERSION,
    available_context_tokens,
    build_generation_request,
)
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.telemetry import (
    CallResult,
    ProviderCallRecord,
    ProviderCallType,
    failed_call,
    replied_call,
)

#: Identifies the shape of an experiment export.
EXPERIMENT_FORMAT_VERSION: Final = "provider-experiment-v1"

#: The configuration as shipped first, then the control.
DEFAULT_VARIANTS: Final = (ResponseFormat.JSON_OBJECT, ResponseFormat.TEXT)

#: Statuses that mean the credentials were refused. Retrying or carrying on
#: would only repeat the refusal.
_AUTH_STATUSES: Final = frozenset({401, 403})

Sleeper = Callable[[float], Awaitable[None]]


class ExperimentError(Exception):
    """The experiment cannot be prepared as asked. Nothing was generated."""


class StopReason(StrEnum):
    """Why a run ended."""

    COMPLETED = "completed"
    MAX_CALLS = "max_calls"
    NEURON_BUDGET = "neuron_budget"
    RATE_LIMITED = "rate_limited"
    AUTH_FAILURE = "auth_failure"
    PROVIDER_ERROR = "provider_error"
    """A provider failure that is not what is being measured: a timeout, an
    unreachable provider, an unexpected status. The run's conditions are no
    longer the ones it set out to compare."""


@dataclass(frozen=True, slots=True)
class PreparedQuestion:
    """One question, retrieved once, as the request every variant will send."""

    question_id: str
    request: GenerationRequest
    context_sha256: str
    """Fingerprint of the context text — evidence that every variant of this
    question was asked against the same passages, without keeping them."""

    context_sources: int
    context_estimated_tokens: int

    def request_for(self, variant: ResponseFormat) -> GenerationRequest:
        """The same request, with only ``response_format`` set to *variant*."""
        return self.request.model_copy(update={"response_format": variant})


@dataclass(frozen=True, slots=True)
class ExperimentLimits:
    """How much one run may spend before it stops on its own."""

    max_calls: int
    neuron_budget: float | None = None
    input_neurons_per_million: float | None = None
    output_neurons_per_million: float | None = None

    def __post_init__(self) -> None:
        if self.max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if self.neuron_budget is not None:
            if self.neuron_budget <= 0:
                raise ValueError("neuron_budget must be positive")
            if self.input_neurons_per_million is None or self.output_neurons_per_million is None:
                raise ValueError(
                    "a neuron budget needs both neuron rates, or the spend cannot be estimated"
                )

    def estimate_neurons(self, call: ProviderCallRecord) -> float | None:
        """Estimated neurons of *call* from the usage its provider reported.

        ``None`` when no rates were given or the provider reported no usage —
        an unknown cost is not a zero one.
        """
        if (
            self.input_neurons_per_million is None
            or self.output_neurons_per_million is None
            or call.input_tokens is None
            or call.output_tokens is None
        ):
            return None
        return (
            call.input_tokens * self.input_neurons_per_million
            + call.output_tokens * self.output_neurons_per_million
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class ExperimentCall:
    """One call of the run: which question, which variant, and what happened."""

    question_id: str
    variant: ResponseFormat
    repetition: int
    record: ProviderCallRecord

    def fields(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "variant": self.variant.value,
            "repetition": self.repetition,
            **self.record.fields(),
        }


@dataclass(frozen=True, slots=True)
class PlannedCall:
    question: PreparedQuestion
    variant: ResponseFormat
    repetition: int


async def prepare_questions(
    questions: Sequence[EvaluationQuestion],
    retrieval: PublicRetrievalService,
    policy: ContextPolicy,
) -> tuple[PreparedQuestion, ...]:
    """Retrieve each question once and build the request every variant shares.

    The same path the answering service takes — normalize, retrieve, build the
    context against the same budget, compose the request — with the public
    builders it uses, so a variant differs from production in nothing but the
    setting under test. A question that retrieves nothing cannot be asked
    about its reply format, and is refused here rather than measured.
    """
    prepared: list[PreparedQuestion] = []
    for question in questions:
        query = normalize_query(question.question)
        outcome = await retrieval.retrieve(query)
        if not outcome.is_sufficient:
            raise ExperimentError(f"question {question.id!r} retrieves nothing to generate from")
        context = build_context(
            outcome.chunks, available_tokens=available_context_tokens(query.text, policy)
        )
        prepared.append(
            PreparedQuestion(
                question_id=question.id,
                request=build_generation_request(
                    question=query.text,
                    context=context,
                    max_output_tokens=policy.output_reserve_tokens,
                ),
                context_sha256=hashlib.sha256(context.text.encode("utf-8")).hexdigest(),
                context_sources=len(context.sources),
                context_estimated_tokens=context.estimated_tokens,
            )
        )
    return tuple(prepared)


def plan_calls(
    prepared: Sequence[PreparedQuestion],
    variants: Sequence[ResponseFormat],
    repetitions: int,
) -> tuple[PlannedCall, ...]:
    """Every question, every repetition, every variant — interleaved.

    Variants alternate within a question, and the order flips on every
    repetition (A B, B A, A B, …), so neither variant is systematically asked
    first, or later in the run when the provider's state may have drifted.
    """
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not variants or len(set(variants)) != len(variants):
        raise ValueError("variants must be distinct and not empty")
    plan: list[PlannedCall] = []
    for question in prepared:
        for repetition in range(1, repetitions + 1):
            order = variants if repetition % 2 == 1 else tuple(reversed(variants))
            plan.extend(PlannedCall(question, variant, repetition) for variant in order)
    return tuple(plan)


@dataclass
class ExperimentRun:
    """Runs a plan against a provider, call by call, until done or stopped.

    The calls made so far are always readable from :attr:`calls`, so whatever
    ends the run — a stop rule or an exception — what was paid for is kept.
    """

    plan: Sequence[PlannedCall]
    llm: LLMProvider
    limits: ExperimentLimits
    delay_seconds: float = 0.0
    sleeper: Sleeper = asyncio.sleep
    clock: Callable[[], float] = time.perf_counter
    calls: list[ExperimentCall] = field(default_factory=list)
    stop_reason: StopReason | None = None
    estimated_neurons: float = 0.0
    unestimated_calls: int = 0

    async def run(self) -> StopReason:
        largest_call: float | None = None
        for position, planned in enumerate(self.plan):
            if len(self.calls) >= self.limits.max_calls:
                return self._stop(StopReason.MAX_CALLS)
            if (
                self.limits.neuron_budget is not None
                and largest_call is not None
                and self.estimated_neurons + largest_call > self.limits.neuron_budget
            ):
                # The next call is assumed to cost as much as the dearest one
                # so far; one that might cross the budget is not made.
                return self._stop(StopReason.NEURON_BUDGET)
            if position > 0 and self.delay_seconds > 0:
                await self.sleeper(self.delay_seconds)

            call, stop = await self._call(planned)
            self.calls.append(call)
            cost = self.limits.estimate_neurons(call.record)
            if cost is None:
                self.unestimated_calls += 1
            else:
                self.estimated_neurons += cost
                largest_call = cost if largest_call is None else max(largest_call, cost)
            if stop is not None:
                return self._stop(stop)
        return self._stop(StopReason.COMPLETED)

    def _stop(self, reason: StopReason) -> StopReason:
        self.stop_reason = reason
        return reason

    async def _call(self, planned: PlannedCall) -> tuple[ExperimentCall, StopReason | None]:
        request = planned.question.request_for(planned.variant)
        started = self.clock()
        try:
            response = await self.llm.generate(request)
        except LLMProviderError as exc:
            record = failed_call(
                call_type=ProviderCallType.GENERATION,
                attempt=1,
                model=self.llm.model,
                response_format=request.response_format,
                elapsed_seconds=round(self.clock() - started, 4),
                failure=provider_failure(exc),
            )
            return self._entry(planned, record), _stop_for(exc)

        elapsed = round(self.clock() - started, 4)
        try:
            parse_generation(response.text)
            failure = None
        except GenerationUnavailableError as exc:
            failure = exc.failure
        record = replied_call(
            call_type=ProviderCallType.GENERATION,
            attempt=1,
            model=self.llm.model,
            response_format=request.response_format,
            elapsed_seconds=elapsed,
            response=response,
            failure=failure,
        )
        return self._entry(planned, record), None

    @staticmethod
    def _entry(planned: PlannedCall, record: ProviderCallRecord) -> ExperimentCall:
        return ExperimentCall(
            question_id=planned.question.question_id,
            variant=planned.variant,
            repetition=planned.repetition,
            record=record,
        )


def _stop_for(error: LLMProviderError) -> StopReason | None:
    """Whether a provider failure ends the run. A malformed response does not:
    it is one of the outcomes being counted."""
    if error.kind is ProviderFailureKind.MALFORMED_RESPONSE:
        return None
    if error.kind is ProviderFailureKind.RATE_LIMITED or error.status_code == 429:
        return StopReason.RATE_LIMITED
    if error.status_code in _AUTH_STATUSES:
        return StopReason.AUTH_FAILURE
    return StopReason.PROVIDER_ERROR


# --- what a run reports ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExperimentMetadata:
    experiment_id: str
    generated_at: datetime
    git_revision: str | None
    git_dirty: bool | None
    dataset_path: str
    dataset_sha256: str
    provider: str
    model: str
    repetitions: int
    variants: tuple[ResponseFormat, ...]
    delay_seconds: float


def export_experiment(
    run: ExperimentRun,
    prepared: Sequence[PreparedQuestion],
    metadata: ExperimentMetadata,
) -> dict[str, Any]:
    """The run as plain JSON-ready data: metadata, contexts, every call, summaries."""
    request = prepared[0].request if prepared else None
    limits = run.limits
    return {
        "format": EXPERIMENT_FORMAT_VERSION,
        "experiment": {
            "experiment_id": metadata.experiment_id,
            "generated_at": metadata.generated_at.isoformat(timespec="seconds"),
            "git_revision": metadata.git_revision,
            "git_dirty": metadata.git_dirty,
            "dataset": {"path": metadata.dataset_path, "sha256": metadata.dataset_sha256},
            "provider": metadata.provider,
            "model": metadata.model,
            "prompt_version": GROUNDED_PROMPT_VERSION,
            "max_output_tokens": request.max_output_tokens if request else None,
            "temperature": request.temperature if request else None,
            "variants": [variant.value for variant in metadata.variants],
            "repetitions": metadata.repetitions,
            "delay_seconds": metadata.delay_seconds,
            "planned_calls": len(run.plan),
            "limits": {
                "max_calls": limits.max_calls,
                "neuron_budget": limits.neuron_budget,
                "input_neurons_per_million": limits.input_neurons_per_million,
                "output_neurons_per_million": limits.output_neurons_per_million,
            },
            "stop_reason": run.stop_reason.value if run.stop_reason else None,
        },
        "cost": {
            "estimated_neurons": round(run.estimated_neurons, 1)
            if limits.input_neurons_per_million is not None
            else None,
            "unestimated_calls": run.unestimated_calls,
            "note": (
                "An estimate: reported tokens times the configured rates. The provider "
                "does not report neurons; its own dashboard is the measurement."
            ),
        },
        "contexts": [
            {
                "question_id": question.question_id,
                "context_sha256": question.context_sha256,
                "context_sources": question.context_sources,
                "context_estimated_tokens": question.context_estimated_tokens,
            }
            for question in prepared
        ],
        "summary": {
            variant.value: _variant_summary(
                [call for call in run.calls if call.variant is variant], prepared
            )
            for variant in metadata.variants
        },
        "calls": [call.fields() for call in run.calls],
    }


def _variant_summary(
    calls: Sequence[ExperimentCall], prepared: Sequence[PreparedQuestion]
) -> dict[str, Any]:
    records = [call.record for call in calls]
    failed = [record for record in records if record.result is not CallResult.PARSED]
    replied = [
        record
        for record in records
        if record.reply_characters is not None and record.reply_visible_characters is not None
    ]
    sizes = [record.reply_characters for record in replied if record.reply_characters is not None]
    ratios = [
        record.reply_visible_characters / record.reply_characters
        for record in replied
        if record.reply_characters and record.reply_visible_characters is not None
    ]
    return {
        **summarize_provider_calls(records),
        "failures": len(failed),
        "failure_rate": round(len(failed) / len(records), 3) if records else None,
        "failure_details": _count(
            record.failure.detail for record in failed if record.failure is not None
        ),
        "reply_characters": _distribution(sizes),
        "visible_ratio": _distribution(ratios, digits=3),
        "failures_by_question": {
            question.question_id: sum(
                1
                for call in calls
                if call.question_id == question.question_id
                and call.record.result is not CallResult.PARSED
            )
            for question in prepared
        },
    }


def _distribution(values: Sequence[float], *, digits: int = 0) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": round(min(values), digits),
        "median": round(statistics.median(values), digits),
        "max": round(max(values), digits),
    }


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
