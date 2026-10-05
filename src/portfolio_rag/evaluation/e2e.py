"""One question, one pass through the whole pipeline, everything it did written down.

The retrieval run and the grounding run are two passes: each question is
embedded and searched once to score retrieval, and again to be answered. That
is the right shape for tuning either half. It is the wrong shape for saying
"this is what the system did for this question", because the passages that
were scored are not provably the passages the answer was generated from.

An end-to-end run asks each question **once**. Retrieval is scored from the
retrieval the answer actually used — the outcome every
:class:`~portfolio_rag.rag.service.GroundedAnswer` already carries — so the
hit, the outcome and the citations in one row all belong to the same request.

**Every check here is deterministic.** Did the pipeline answer or refuse, did
it publish citations, does every published citation trace back to a passage
that was in the context, did the model point at labels that do not exist, did
an internal label reach the answer text. None of that needs a second model.

**What it deliberately does not judge** is whether an answer's wording is
correct, complete or well put. That needs a human or a judge model, and this
harness claims neither. The closest deterministic signal is reported instead:
whether an answer cites at least one section the ground truth accepts. The
final answer text is written to the export so that the remaining judgement can
be made by reading it.

**Gates and measurements are kept apart.** Hit rates and the share of
answerable questions that were answered are measurements: numbers to read, with
no pass mark, because a pass mark chosen after seeing the numbers is tuning.
The gates are the properties that must hold at any quality level — no pipeline
error, no unverified or invented citation, no leaked label or internal passage,
and a controlled refusal with no citation for every question the corpus cannot
answer.
"""

from __future__ import annotations

import hashlib
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.domain.retrieval import RetrievedChunk, SourceCitation
from portfolio_rag.evaluation.dataset import EvaluationQuestion
from portfolio_rag.evaluation.export import RunMetadata, export_retrieval
from portfolio_rag.evaluation.metrics import (
    RetrievalOutcomeRecord,
    RetrievalReport,
    build_report,
    is_relevant,
    score_retrieval,
)
from portfolio_rag.rag.errors import GenerationFailure
from portfolio_rag.rag.generation import SupportVerdict
from portfolio_rag.rag.language import INSUFFICIENT_KNOWLEDGE_ANSWERS
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.prompt import GROUNDED_RESPONSE_FORMAT
from portfolio_rag.rag.retrieval import RetrievalOutcome
from portfolio_rag.rag.service import (
    MAX_GENERATION_ATTEMPTS,
    MAX_GROUNDING_CHECK_ATTEMPTS,
    AnswerOutcome,
    GroundedAnswer,
)
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType
from portfolio_rag.rag.verification import (
    GROUNDING_CHECK_RESPONSE_FORMAT,
    GROUNDING_CHECK_VERSION,
    GroundingVerdict,
)

#: Identifies the shape of the file. Bumped when a field changes meaning or
#: disappears, so a reader can tell which shape it is holding.
#:
#: ``v2`` changed what the retrieval metrics count. In ``v1`` a question the
#: pipeline raised on was scored as a retrieval miss; from ``v2`` its retrieval
#: is scored when it was observed and left out when it was not, and availability
#: is reported on its own. Hit rates from the two versions are not comparable
#: for a run that had errors.
#:
#: ``v3`` follows the failure policy (`rag.failure_policy`). A grounding check
#: that gave no readable verdict was ``not_grounded`` — a quality finding — in
#: ``v2``; it is a technical error, ``pipeline_error``, in ``v3``. An errored
#: question no longer *also* counts as ``not_answered``: one root cause, one
#: failure. The generation failure categories are sharper — ``output_truncated``
#: and ``malformed_response`` split out, ``other`` became ``unclassified`` —
#: and every failure names its ``failure_step``. A run can be ``aborted`` by an
#: internal defect and say so. ``v2`` files stay readable; their not_grounded,
#: not_answered and pipeline_error counts are not comparable with ``v3``.
E2E_FORMAT_VERSION: Final = "e2e-eval-v3"

#: An internal source label as the context writes it. One in a published answer
#: means presentation let something through that a reader was never meant to see.
INTERNAL_LABEL: Final = re.compile(r"\[S\d+\]|\(S\d+(?:\s*,\s*S\d+)*\)|\bSOURCE S\d+\b")

#: A citation mark as a reader sees it: ``[1]``, ``[2]``.
_PUBLIC_MARK: Final = re.compile(r"\[(\d+)\]")

_REFUSAL_WORDINGS: Final = frozenset(INSUFFICIENT_KNOWLEDGE_ANSWERS.values())


class E2EFailure(StrEnum):
    """Why a question did not do what the dataset says it should.

    Named individually, because the fix differs for each. Only
    :attr:`RETRIEVAL_MISS` and :attr:`NOT_ANSWERED` are quality findings; every
    other one is a gate — see :data:`GATES`.
    """

    RETRIEVAL_MISS = "retrieval_miss"
    """Answerable, and no expected source among the retrieved passages."""

    NOT_ANSWERED = "not_answered"
    """Answerable, and the pipeline refused."""

    ANSWERED_UNKNOWN = "answered_unknown"
    """The corpus has no public answer, and the pipeline answered anyway."""

    CITATION_ON_REFUSAL = "citation_on_refusal"
    PIPELINE_ERROR = "pipeline_error"
    EMPTY_ANSWER = "empty_answer"
    NO_CITATION = "answered_without_citation"
    UNVERIFIED_CITATION = "unverified_citation"
    """A published citation that no passage in the context stands behind."""

    INVENTED_LABEL = "invented_source_label"
    """The model cited a label that was never in the context. The backend drops
    it before anything is published; it is counted here because a model that
    invents references is a finding whether or not one got through."""

    DANGLING_MARK = "dangling_citation_mark"
    """The answer text carries a ``[n]`` with no n-th citation behind it."""

    LABEL_LEAK = "internal_label_leak"
    INTERNAL_PASSAGE = "internal_passage_retrieved"


class FailureClass(StrEnum):
    """Which kind of property a failure says something about."""

    SAFETY = "safety"
    """Something was published that must never be. Zero tolerance."""

    AVAILABILITY = "availability"
    """The pipeline could not produce an outcome. Technical, never a statement
    about the knowledge base."""

    QUALITY = "quality"
    """The pipeline worked and the result was weaker than the ground truth
    asks for: a missed passage, a refusal of an answerable question."""


#: Every failure, and the one class it belongs to. Exhaustive, which a test
#: asserts: a new failure without a class is a build failure.
FAILURE_CLASSES: Final[dict[E2EFailure, FailureClass]] = {
    E2EFailure.RETRIEVAL_MISS: FailureClass.QUALITY,
    E2EFailure.NOT_ANSWERED: FailureClass.QUALITY,
    E2EFailure.PIPELINE_ERROR: FailureClass.AVAILABILITY,
    E2EFailure.EMPTY_ANSWER: FailureClass.AVAILABILITY,
    E2EFailure.ANSWERED_UNKNOWN: FailureClass.SAFETY,
    E2EFailure.CITATION_ON_REFUSAL: FailureClass.SAFETY,
    E2EFailure.NO_CITATION: FailureClass.SAFETY,
    E2EFailure.UNVERIFIED_CITATION: FailureClass.SAFETY,
    E2EFailure.INVENTED_LABEL: FailureClass.SAFETY,
    E2EFailure.DANGLING_MARK: FailureClass.SAFETY,
    E2EFailure.LABEL_LEAK: FailureClass.SAFETY,
    E2EFailure.INTERNAL_PASSAGE: FailureClass.SAFETY,
}

#: The failures that make a run fail, grouped the way the summary reports them.
#: Everything not listed here is a measurement.
GATES: Final[dict[str, frozenset[E2EFailure]]] = {
    "no_pipeline_errors": frozenset({E2EFailure.PIPELINE_ERROR, E2EFailure.EMPTY_ANSWER}),
    "citations_verified": frozenset(
        {
            E2EFailure.NO_CITATION,
            E2EFailure.UNVERIFIED_CITATION,
            E2EFailure.INVENTED_LABEL,
            E2EFailure.DANGLING_MARK,
        }
    ),
    "unanswerable_refused": frozenset(
        {E2EFailure.ANSWERED_UNKNOWN, E2EFailure.CITATION_ON_REFUSAL}
    ),
    "nothing_internal_published": frozenset({E2EFailure.LABEL_LEAK, E2EFailure.INTERNAL_PASSAGE}),
}


@dataclass(frozen=True, slots=True)
class E2ERecord:
    """Everything one question did on its single pass through the pipeline."""

    question: EvaluationQuestion
    retrieval: RetrievalOutcomeRecord
    retrieval_observed: bool
    """Whether :attr:`retrieval` is something this run saw. ``False`` for a
    question that failed before or during retrieval: its record is empty
    because nothing is known, not because nothing was found."""

    outcome: AnswerOutcome | None
    """``None`` when the pipeline raised instead of answering."""

    error_code: str | None
    """The client-safe error code of that failure. Never its message."""

    error: GenerationFailure | None
    """Why a generation failed, when that is what failed. Technical facts only."""

    answer: str
    citations: tuple[SourceCitation, ...]
    verified_citations: int
    """Published citations that trace back to a passage in the context."""

    cites_expected_source: bool
    """Whether a published citation is a section the ground truth accepts."""

    unknown_labels: tuple[str, ...]
    context_sources: int
    generation_seconds: float
    duration_seconds: float
    failures: tuple[E2EFailure, ...]

    support: SupportVerdict | None = None
    """What the model declared about its own answer. ``None`` when no model
    replied, or when its reply carried no verdict."""

    claimed_labels: tuple[str, ...] = ()
    """The source labels as the model listed them, before validation. On a
    refusal these are the labels that were *not* published."""

    grounding_check: GroundingVerdict | None = None
    """What the grounding check said, or ``None`` when it was never asked."""

    grounding_check_seconds: float = 0.0

    generation_attempts: int = 0
    """How many times the answer was generated. ``2`` means the first reply
    was unusable and the question was asked again."""

    grounding_check_attempts: int = 0
    """How many times the grounding check was asked. ``2`` means its first
    reply was unusable and it was asked again; ``0`` that it was not asked."""

    provider_calls: tuple[ProviderCallRecord, ...] = ()
    """Every call the question made to the generation provider, in order —
    for a question that errored, up to and including the failing one.
    Metadata only."""

    @property
    def errored(self) -> bool:
        return self.outcome is None

    @property
    def is_controlled_refusal(self) -> bool:
        """A refusal with the fixed wording and nothing cited."""
        return (
            self.outcome in {AnswerOutcome.NO_KNOWLEDGE, AnswerOutcome.NOT_GROUNDED}
            and self.answer in _REFUSAL_WORDINGS
            and not self.citations
        )


def score_answer(
    question: EvaluationQuestion, answer: GroundedAnswer, *, duration_seconds: float
) -> E2ERecord:
    """Check one answer against what the dataset says should have happened."""
    retrieval = score_retrieval(question, answer.retrieval.chunks, outcome=answer.retrieval)
    sources = answer.context.sources if answer.context is not None else ()
    # Several passages can stand behind one citation: validation publishes one
    # per document and section, and two headings can share an innermost name.
    backing: dict[SourceCitation, list[RetrievedChunk]] = {}
    for source in sources:
        backing.setdefault(source.retrieved.citation(), []).append(source.retrieved)
    verified = sum(1 for citation in answer.citations if citation in backing)
    answered = answer.outcome is AnswerOutcome.ANSWERED

    failures: list[E2EFailure] = []
    if question.expects_an_answer:
        if retrieval.is_failure:
            failures.append(E2EFailure.RETRIEVAL_MISS)
        if not answered:
            failures.append(E2EFailure.NOT_ANSWERED)
    elif question.must_retrieve_nothing and answered:
        failures.append(E2EFailure.ANSWERED_UNKNOWN)

    if not answer.answer.strip():
        failures.append(E2EFailure.EMPTY_ANSWER)
    if answered and not answer.citations:
        failures.append(E2EFailure.NO_CITATION)
    if not answered and answer.citations:
        failures.append(E2EFailure.CITATION_ON_REFUSAL)
    if verified != len(answer.citations):
        failures.append(E2EFailure.UNVERIFIED_CITATION)
    if answer.unknown_labels:
        failures.append(E2EFailure.INVENTED_LABEL)
    if _has_dangling_mark(answer.answer, len(answer.citations)):
        failures.append(E2EFailure.DANGLING_MARK)
    if INTERNAL_LABEL.search(answer.answer):
        failures.append(E2EFailure.LABEL_LEAK)
    if retrieval.leaked_internal:
        failures.append(E2EFailure.INTERNAL_PASSAGE)

    return E2ERecord(
        question=question,
        retrieval=retrieval,
        retrieval_observed=True,
        outcome=answer.outcome,
        error_code=None,
        error=None,
        answer=answer.answer,
        citations=answer.citations,
        verified_citations=verified,
        cites_expected_source=any(
            is_relevant(question, passage)
            for citation in answer.citations
            for passage in backing.get(citation, ())
        ),
        unknown_labels=answer.unknown_labels,
        context_sources=len(sources),
        generation_seconds=answer.generation_seconds,
        duration_seconds=duration_seconds,
        failures=tuple(failures),
        support=answer.support,
        claimed_labels=answer.claimed_labels,
        grounding_check=answer.grounding_check,
        grounding_check_seconds=answer.grounding_check_seconds,
        generation_attempts=answer.generation_attempts,
        grounding_check_attempts=answer.grounding_check_attempts,
        provider_calls=answer.provider_calls,
    )


def errored_record(
    question: EvaluationQuestion,
    error_code: str,
    *,
    duration_seconds: float,
    retrieval: RetrievalOutcome | None = None,
    failure: GenerationFailure | None = None,
    provider_calls: tuple[ProviderCallRecord, ...] = (),
) -> E2ERecord:
    """The record of a question the pipeline raised on.

    Retrieval and generation fail separately and are scored separately. A
    generation failure carries the retrieval that preceded it, and that
    retrieval is scored exactly as if the answer had been produced: a provider
    that did not answer says nothing about whether the right passage was
    found. Only a question with no retrieval to look at — the failure came
    before or during the search — is left out of the retrieval metrics, and it
    is left out rather than counted as a miss.
    """
    observed = retrieval is not None
    scored = (
        score_retrieval(question, retrieval.chunks, outcome=retrieval)
        if retrieval is not None
        else score_retrieval(question, ())
    )
    # One root cause, one failure: the question was not answered *because* the
    # pipeline failed, so it is a pipeline error and not also a quality finding.
    # A retrieval that missed is a separate cause and is still reported.
    failures = [E2EFailure.PIPELINE_ERROR]
    if question.expects_an_answer and observed and scored.is_failure:
        failures.append(E2EFailure.RETRIEVAL_MISS)
    if scored.leaked_internal:
        failures.append(E2EFailure.INTERNAL_PASSAGE)
    return E2ERecord(
        question=question,
        retrieval=scored,
        retrieval_observed=observed,
        outcome=None,
        error_code=error_code,
        error=failure,
        answer="",
        citations=(),
        verified_citations=0,
        cites_expected_source=False,
        unknown_labels=(),
        context_sources=0,
        generation_seconds=0.0,
        duration_seconds=duration_seconds,
        failures=tuple(failures),
        generation_attempts=sum(
            1 for call in provider_calls if call.call_type is ProviderCallType.GENERATION
        ),
        grounding_check_attempts=sum(
            1 for call in provider_calls if call.call_type is ProviderCallType.GROUNDING_CHECK
        ),
        provider_calls=provider_calls,
    )


def _has_dangling_mark(answer: str, citation_count: int) -> bool:
    return any(not 1 <= int(number) <= citation_count for number in _PUBLIC_MARK.findall(answer))


class AbortReason(StrEnum):
    """Why a run stopped before its last question."""

    INTERNAL_DEFECT = "internal_defect"
    """A defect in this code — not a provider failure, and not run past."""

    RATE_LIMITED = "rate_limited"
    """The provider answered 429. Whether that was a short rate limit or the
    day's allocation cannot be told from here, so nothing more is asked."""

    AUTH_FAILURE = "auth_failure"
    """The provider refused the credentials. Every further call would too."""

    SYSTEMIC_PROVIDER_FAILURE = "systemic_provider_failure"
    """Consecutive questions failed on the provider's side."""

    BUDGET_GUARD = "budget_guard"
    """The run was heading past the production reserve."""


@dataclass(frozen=True, slots=True)
class RunAbort:
    """Why a run stopped before its last question. Technical facts only."""

    question_id: str
    """The question the run stopped at: for a defect, the one being asked; for
    an operational stop, the last one asked. Nothing after it was asked."""

    reason: AbortReason = AbortReason.INTERNAL_DEFECT
    error_type: str | None = None
    """For an internal defect, the exception's class name — never its message,
    which may hold content."""


@dataclass(frozen=True, slots=True)
class E2EReport:
    """The whole run. Aggregates are derived; the records are the result."""

    records: tuple[E2ERecord, ...]
    aborted: RunAbort | None = None
    """Set when an internal defect ended the run early. The records are then
    what was measured before it — a partial result, never an acceptance run."""

    @property
    def retrieval(self) -> RetrievalReport:
        """Retrieval metrics over the retrievals this run actually observed.

        A question whose retrieval never happened is not in here at all, so the
        denominators say how many questions the numbers are about.
        """
        return build_report(
            [record.retrieval for record in self.records if record.retrieval_observed]
        )

    @property
    def retrieval_not_observed(self) -> tuple[E2ERecord, ...]:
        return tuple(record for record in self.records if not record.retrieval_observed)

    def with_failure(self, *failures: E2EFailure) -> tuple[E2ERecord, ...]:
        wanted = set(failures)
        return tuple(record for record in self.records if wanted & set(record.failures))

    def count(self, outcome: AnswerOutcome | None) -> int:
        return sum(1 for record in self.records if record.outcome is outcome)

    @property
    def answerable(self) -> tuple[E2ERecord, ...]:
        return tuple(record for record in self.records if record.question.expects_an_answer)

    @property
    def must_refuse(self) -> tuple[E2ERecord, ...]:
        return tuple(record for record in self.records if record.question.must_retrieve_nothing)

    @property
    def gates(self) -> dict[str, bool]:
        """Each gate, and whether no question broke it."""
        return {name: not self.with_failure(*failures) for name, failures in GATES.items()}

    @property
    def passed(self) -> bool:
        return self.aborted is None and all(self.gates.values())


# --- export -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorpusIdentity:
    """Which corpus the answers were resolved against."""

    documents: int
    chunks: int
    sha256: str
    """Over every document id and its ingestion fingerprint, in id order."""


def corpus_identity(chunks: Sequence[KnowledgeChunk]) -> CorpusIdentity:
    """Name the corpus by the document fingerprints ingestion already computes."""
    documents = {
        chunk.document_id: chunk.provenance.document.document_fingerprint for chunk in chunks
    }
    canonical = "\n".join(
        f"{name}:{fingerprint}" for name, fingerprint in sorted(documents.items())
    )
    return CorpusIdentity(
        documents=len(documents),
        chunks=len(chunks),
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class E2ERunMetadata:
    """What an end-to-end run measured, on top of what a retrieval run records."""

    retrieval: RunMetadata
    generation_provider: str
    generation_model: str
    prompt_version: str
    context_policy: ContextPolicy
    generation_delay_seconds: float
    corpus: CorpusIdentity
    notes: tuple[str, ...] = ()
    """Free text from whoever started the run: known conditions a reader needs
    in order to interpret the numbers."""

    selected_question_ids: tuple[str, ...] = ()
    """The ids the run was narrowed to, when it was. Empty for a whole suite."""

    tier: str | None = None
    """The provider-run tier (``smoke``, ``acceptance``), or ``None`` for a run
    that is not tiered. Only an acceptance-tier run can be a release acceptance."""

    transport_attempts: int | None = None
    """Transport attempts the generation adapter had per generation, when the
    run overrode the adapter's default; ``None`` when it did not."""

    rerun_of: str | None = None
    """The run this one repeats under the provider-outlier rule, if it does."""


def export_e2e(
    report: E2EReport,
    metadata: E2ERunMetadata,
    operations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe *report* as plain JSON-ready data.

    Built on the retrieval export, so the run header, the retrieval metrics and
    every question's hits have exactly the shape a retrieval-only file has.
    """
    # Every question gets a row, observed or not; the metrics are computed
    # over the observed ones only.
    base = export_retrieval(
        build_report([record.retrieval for record in report.records]), metadata.retrieval
    )
    retrieval_metrics = export_retrieval(report.retrieval, metadata.retrieval)["metrics"]
    return {
        "format": E2E_FORMAT_VERSION,
        "run": {
            **base["run"],
            "corpus": {
                "path": base["run"]["dataset"]["corpus"],
                "documents": metadata.corpus.documents,
                "chunks": metadata.corpus.chunks,
                "sha256": metadata.corpus.sha256,
            },
            "generation": {
                "provider": metadata.generation_provider,
                "model": metadata.generation_model,
                "prompt_version": metadata.prompt_version,
                "grounding_check_version": GROUNDING_CHECK_VERSION,
                "response_format": GROUNDED_RESPONSE_FORMAT.value,
                "grounding_check_response_format": GROUNDING_CHECK_RESPONSE_FORMAT.value,
                "max_prompt_tokens": metadata.context_policy.max_prompt_tokens,
                "output_reserve_tokens": metadata.context_policy.output_reserve_tokens,
                # A first request asks for at most the output reserve.
                "max_output_tokens": metadata.context_policy.output_reserve_tokens,
                # A recovery after a reply stopped at the limit asks for this.
                "recovery_output_tokens": metadata.context_policy.recovery_output_tokens,
                # Generations per question: the first and its one regeneration.
                "generation_attempt_limit": MAX_GENERATION_ATTEMPTS,
                # Grounding checks per answer: the first and its one recovery.
                "grounding_check_attempt_limit": MAX_GROUNDING_CHECK_ATTEMPTS,
                "transport_attempts": metadata.transport_attempts,
                "delay_seconds": metadata.generation_delay_seconds,
            },
            "tier": metadata.tier,
            "rerun_of": metadata.rerun_of,
            "notes": list(metadata.notes),
            "selected_question_ids": list(metadata.selected_question_ids),
            "complete": report.aborted is None,
            "aborted": (
                {
                    "reason": report.aborted.reason.value,
                    "question_id": report.aborted.question_id,
                    "error_type": report.aborted.error_type,
                }
                if report.aborted is not None
                else None
            ),
        },
        "operations": dict(operations) if operations is not None else None,
        "passed": report.passed,
        "gates": report.gates,
        "failure_classes": {
            failure.value: FAILURE_CLASSES[failure].value for failure in E2EFailure
        },
        "metrics": {
            "retrieval": {
                **retrieval_metrics,
                "observed": {
                    "correct": len(report.records) - len(report.retrieval_not_observed),
                    "total": len(report.records),
                },
                "not_observed": _ids(report.retrieval_not_observed),
            },
            "availability": _availability_metrics(report),
            "answers": _answer_metrics(report),
            "citations": _citation_metrics(report),
            "refusals": _refusal_metrics(report),
            "robustness": _robustness_metrics(report),
            "latency": _latency_metrics(report),
            "provider_calls": _provider_call_metrics(report),
        },
        "failures": {
            failure.value: _ids(report.with_failure(failure))
            for failure in E2EFailure
            if report.with_failure(failure)
        },
        "questions": [
            {**question, **_answer(record)}
            for question, record in zip(base["questions"], report.records, strict=True)
        ],
    }


def _ids(records: Sequence[E2ERecord]) -> list[str]:
    return [record.question.id for record in records]


def _availability_metrics(report: E2EReport) -> dict[str, Any]:
    """Whether the pipeline produced an outcome at all — apart from what it found."""
    errored = [record for record in report.records if record.errored]
    by_category: dict[str, list[str]] = {}
    for record in errored:
        category = record.error.category.value if record.error else "not_a_generation_failure"
        by_category.setdefault(category, []).append(record.question.id)
    return {
        "completed": {
            "correct": len(report.records) - len(errored),
            "total": len(report.records),
        },
        "errors_by_category": by_category,
    }


def _answer_metrics(report: E2EReport) -> dict[str, Any]:
    answerable = report.answerable
    answered = [record for record in answerable if record.outcome is AnswerOutcome.ANSWERED]
    return {
        "outcomes": {
            AnswerOutcome.ANSWERED.value: report.count(AnswerOutcome.ANSWERED),
            AnswerOutcome.NO_KNOWLEDGE.value: report.count(AnswerOutcome.NO_KNOWLEDGE),
            AnswerOutcome.NOT_GROUNDED.value: report.count(AnswerOutcome.NOT_GROUNDED),
            "error": report.count(None),
        },
        "answerable_answered": {"correct": len(answered), "total": len(answerable)},
        "answered_citing_expected_source": {
            "correct": sum(1 for record in answered if record.cites_expected_source),
            "total": len(answered),
        },
        "answered_without_expected_source": _ids(
            [record for record in answered if not record.cites_expected_source]
        ),
    }


def _citation_metrics(report: E2EReport) -> dict[str, Any]:
    published = sum(len(record.citations) for record in report.records)
    return {
        "published": published,
        "validity": {
            "correct": sum(record.verified_citations for record in report.records),
            "total": published,
        },
        "invented_source_labels": sum(len(record.unknown_labels) for record in report.records),
        "answerable_without_citation": _ids(
            [record for record in report.answerable if not record.citations]
        ),
    }


def _refusal_metrics(report: E2EReport) -> dict[str, Any]:
    expected = report.must_refuse
    return {
        "controlled_refusals": {
            "correct": sum(1 for record in expected if record.is_controlled_refusal),
            "total": len(expected),
        },
        "not_refused": _ids([record for record in expected if not record.is_controlled_refusal]),
    }


def _robustness_metrics(report: E2EReport) -> dict[str, Any]:
    outcomes = [record.retrieval.outcome for record in report.records]
    return {
        "pipeline_errors": report.count(None),
        "empty_answers": len(report.with_failure(E2EFailure.EMPTY_ANSWER)),
        "internal_label_leaks": len(report.with_failure(E2EFailure.LABEL_LEAK)),
        "internal_passages_retrieved": len(report.with_failure(E2EFailure.INTERNAL_PASSAGE)),
        "unresolved_matches": sum(outcome.unresolved for outcome in outcomes if outcome),
        "withheld_matches": sum(outcome.withheld for outcome in outcomes if outcome),
    }


def _latency_metrics(report: E2EReport) -> dict[str, Any]:
    """Wall-clock per question, pacing waits included. Not a benchmark."""
    durations = [record.duration_seconds for record in report.records if not record.errored]
    if not durations:
        return {"median_seconds": None, "max_seconds": None}
    return {
        "median_seconds": round(statistics.median(durations), 3),
        "max_seconds": round(max(durations), 3),
    }


def _provider_call_metrics(report: E2EReport) -> dict[str, Any]:
    """Every provider call of the run, counted by the step that made it.

    Token sums cover only the calls whose provider reported usage, and
    ``usage_reported`` says how many that was: a missing count is never
    filled in. Neurons are not reported — the provider does not return them,
    and an estimate would need a rate this run does not know.
    """
    calls = [call for record in report.records for call in record.provider_calls]
    return {
        "total": summarize_provider_calls(calls),
        **{
            call_type.value: summarize_provider_calls(
                [call for call in calls if call.call_type is call_type]
            )
            for call_type in ProviderCallType
        },
    }


def summarize_provider_calls(calls: Sequence[ProviderCallRecord]) -> dict[str, Any]:
    """Counts, results, reported usage, elapsed time and finish reasons of *calls*.

    Shared by the end-to-end export and the provider-contract experiment, so
    that both describe a set of calls the same way.
    """
    reported = [call for call in calls if call.usage_reported]
    latencies = [call.elapsed_seconds for call in calls]
    finish_reasons: dict[str, int] = {}
    for call in calls:
        key = call.finish_reason if call.finish_reason is not None else "none"
        finish_reasons[key] = finish_reasons.get(key, 0) + 1
    return {
        "calls": len(calls),
        "regenerations": sum(1 for call in calls if call.is_regeneration),
        "results": {
            result.value: sum(1 for call in calls if call.result is result) for result in CallResult
        },
        "usage_reported": len(reported),
        # No reported usage is "unknown", not zero.
        "input_tokens": sum(call.input_tokens or 0 for call in reported) if reported else None,
        "output_tokens": sum(call.output_tokens or 0 for call in reported) if reported else None,
        "elapsed_seconds": {
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "finish_reasons": dict(sorted(finish_reasons.items())),
    }


def _answer(record: E2ERecord) -> dict[str, Any]:
    return {
        "retrieval_observed": record.retrieval_observed,
        "outcome": record.outcome.value if record.outcome is not None else "error",
        "error_code": record.error_code,
        "error": record.error.fields() if record.error is not None else None,
        "answer": record.answer,
        "controlled_refusal": record.is_controlled_refusal,
        "support": record.support.value if record.support is not None else None,
        "grounding_check": (
            record.grounding_check.value if record.grounding_check is not None else None
        ),
        "context_sources": record.context_sources,
        "citations": {
            "claimed_source_labels": list(record.claimed_labels),
            "published": [
                {"document_id": citation.document_id, "section": citation.section}
                for citation in record.citations
            ],
            "verified": record.verified_citations,
            "cites_expected_source": record.cites_expected_source,
            "invented_source_labels": list(record.unknown_labels),
        },
        "generation_seconds": record.generation_seconds,
        "generation_attempts": record.generation_attempts,
        "grounding_check_attempts": record.grounding_check_attempts,
        "provider_calls": [call.fields() for call in record.provider_calls],
        "grounding_check_seconds": record.grounding_check_seconds,
        "duration_seconds": record.duration_seconds,
        "failures": [failure.value for failure in record.failures],
    }


# --- summary ------------------------------------------------------------------


def render_summary(payload: dict[str, Any]) -> str:
    """Render an export as Markdown.

    Reads the export and nothing else, so the summary cannot say something the
    data file does not.
    """
    run, metrics = payload["run"], payload["metrics"]
    retrieval, answers = metrics["retrieval"], metrics["answers"]
    availability = metrics["availability"]
    citations, refusals = metrics["citations"], metrics["refusals"]
    robustness, latency = metrics["robustness"], metrics["latency"]

    lines = [
        "# End-to-end evaluation",
        "",
        f"Format `{payload['format']}`. Every question was asked once and went through the "
        "whole pipeline: query embedding, vector search, generation, grounding and citation "
        "validation. Retrieval is scored from the retrieval each answer actually used.",
        "",
        f"**Gates: {'PASS' if payload['passed'] else 'FAIL'}**",
        "",
        *_abort_notice(run),
        *_release_summary(payload),
        "## Run",
        "",
        "| | |",
        "| --- | --- |",
        f"| Generated | {run['generated_at']} |",
        f"| Run | `{run.get('run_id') or '—'}` |",
        f"| Git revision | `{run['git_revision']}`{' (dirty)' if run['git_dirty'] else ''} |",
        f"| Dataset | `{run['dataset']['path']}` v{run['dataset']['version']}, "
        f"sha256 `{run['dataset']['sha256'][:12]}` |",
        f"| Suite | {run['suite']}, {run['question_count']} questions"
        f"{' (selected by id)' if run['selected_question_ids'] else ''} |",
        f"| Corpus | {run['corpus']['documents']} documents, {run['corpus']['chunks']} chunks, "
        f"sha256 `{run['corpus']['sha256'][:12]}` |",
        f"| Embedding | `{run['embedding']['identity']}` |",
        f"| Vector store | {run['vector_store']} |",
        f"| Generation | {run['generation']['provider']}, `{run['generation']['model']}`, "
        f"prompt `{run['generation']['prompt_version']}`, "
        f"check `{run['generation']['grounding_check_version']}` |",
        f"| Retrieval policy | top_k {run['retrieval_policy']['top_k']}, "
        f"min_similarity {run['retrieval_policy']['min_similarity']} |",
        "",
        "## Gates",
        "",
        "Properties that must hold whatever the quality numbers are.",
        "",
        "| Gate | Result |",
        "| --- | --- |",
        *(f"| {name} | {'pass' if ok else 'FAIL'} |" for name, ok in payload["gates"].items()),
        "",
        "## Measurements",
        "",
        "Counts with their denominators. No pass mark is attached to any of them.",
        "",
        "| Metric | Result |",
        "| --- | --- |",
        f"| Questions the pipeline completed | {_ratio(availability['completed'])} |",
        f"| Questions with an observed retrieval | {_ratio(retrieval['observed'])} |",
        *(
            f"| Retrieval {label} | {_ratio(score)} |"
            for label, score in retrieval["hit_rates"].items()
        ),
        f"| Retrieval MRR | {retrieval['mean_reciprocal_rank']:.3f} |",
        f"| Answerable questions answered | {_ratio(answers['answerable_answered'])} |",
        "| Answers citing an expected source | "
        f"{_ratio(answers['answered_citing_expected_source'])} |",
        f"| Unanswerable questions refused | {_ratio(refusals['controlled_refusals'])} |",
        f"| Published citations verified | {_ratio(citations['validity'])} |",
        f"| Source labels invented by the model | {citations['invented_source_labels']} |",
        f"| Pipeline errors | {robustness['pipeline_errors']} |",
        f"| Empty answers | {robustness['empty_answers']} |",
        f"| Internal label leaks | {robustness['internal_label_leaks']} |",
        f"| Internal passages retrieved | {robustness['internal_passages_retrieved']} |",
        f"| Stale index matches dropped | {robustness['unresolved_matches']} |",
        f"| Seconds per question, median / max | {latency['median_seconds']} / "
        f"{latency['max_seconds']} (pacing included) |",
        "",
        "Outcomes: "
        + ", ".join(f"{count} {name}" for name, count in answers["outcomes"].items())
        + ".",
        "",
        *_question_table(payload),
        "## Failures",
        "",
        "`retrieval_miss` and `not_answered` are quality findings. Every other reason "
        "breaks a gate.",
        "",
    ]

    failing = [question for question in payload["questions"] if question["failures"]]
    classes = payload.get("failure_classes", {})
    if failing:
        lines += [
            "| Question | Category | Outcome | Reasons | Class |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{question['id']}` | {question['category']} | {question['outcome']} | "
            f"{', '.join(question['failures'])} | "
            f"{', '.join(sorted({classes.get(f, '—') for f in question['failures']}))} |"
            for question in failing
        ]
    else:
        lines.append("None.")

    errored = [question for question in payload["questions"] if question["outcome"] == "error"]
    if errored:
        lines += [
            "",
            "## Pipeline errors",
            "",
            "Retrieval is scored for these questions wherever it was observed; the error "
            "counts against availability, not against retrieval.",
            "",
            "| Question | Code | Step | Category | Detail | Status | Attempts | Finish reason "
            "| Characters / visible / tokens |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for question in errored:
            error = question["error"] or {}
            lines.append(
                f"| `{question['id']}` | {question['error_code']} | "
                + " | ".join(
                    _cell(error, name)
                    for name in (
                        "failure_step",
                        "failure_category",
                        "failure_detail",
                        "status_code",
                        "attempts",
                        "finish_reason",
                    )
                )
                + f" | {_reply_size(error)} |"
            )

    lines += _provider_call_summary(payload)
    lines += _operations_summary(payload.get("operations"))

    lines += [
        "",
        "## What this does not measure",
        "",
        "- Whether an answer's wording is correct or complete. No model judges another "
        "model here; the answers are in the data file to be read.",
        "- A semantically fitting passage that the ground truth does not list still counts "
        "as a retrieval miss.",
        "- Similarities are similarities in the index's metric, not confidences.",
        "- One run of a non-deterministic generation model. A second run can differ.",
    ]
    if run["notes"]:
        lines += ["", "## Notes for this run", "", *(f"- {note}" for note in run["notes"])]
    return "\n".join(lines) + "\n"


def _question_table(payload: dict[str, Any]) -> list[str]:
    """Every question of the run, one row each: what it is and what happened.

    Compact on purpose — no answer text; the export holds that. A reader sees
    at a glance which kinds of question the suite asks and how each ended.
    """
    lines = [
        "## Questions",
        "",
        f"Suite `{payload['run']['suite']}`"
        + (
            f" v{payload['run']['suite_version']}"
            if payload["run"].get("suite_version") is not None
            else ""
        )
        + f", {len(payload['questions'])} questions.",
        "",
        "| Question | Asks | Category | Expected | Outcome | Citations | Findings |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for question in payload["questions"]:
        citations = question.get("citations") or {}
        published = len(citations.get("published") or [])
        lines.append(
            f"| `{question['id']}` | {_table_text(question['question'])} | "
            f"{question['category']} | "
            f"{_expected(question)} | "
            f"{question['outcome']} | "
            f"{citations.get('verified', 0)}/{published} verified | "
            f"{', '.join(question['failures']) or '—'} |"
        )
    lines.append("")
    return lines


def _expected(question: dict[str, Any]) -> str:
    """What the dataset expects of a question, in one word."""
    if question["expects_an_answer"]:
        return "answer"
    if question["category"] in {"unknown", "internal"}:
        return "refuse"
    return "no leak"


def _table_text(text: str) -> str:
    """One table cell: no pipes, no line breaks."""
    return " ".join(text.replace("|", "/").split())


def _release_summary(payload: dict[str, Any]) -> list[str]:
    """The release-acceptance verdict and what it is about. Exports written
    before the verdict was recorded have none, and get nothing."""
    release = payload.get("release_acceptance")
    if not release:
        return []
    run = payload["run"]
    generation = run["generation"]
    operations = payload.get("operations") or {}
    observed = operations.get("observed") or {}
    gates = release["gates"]
    lines = [
        "## Release acceptance",
        "",
        f"**Release acceptance: {release['verdict']}** (protocol `{release['protocol']}`)",
        "",
        "| | |",
        "| --- | --- |",
        f"| Project version | {_cell(run, 'project_version')} |",
        f"| Release source identity | `{_cell(run, 'release_source_identity')}` |",
        f"| Commit | `{release['commit'] or '—'}` |",
        f"| Working tree | {_tree_state(release['git_dirty'])} |",
        f"| Release tag | {_cell(run, 'git_tag')} |",
        f"| Suite | {release['suite']} v{release['suite_version']}, "
        f"sha256 `{str(run.get('suite_sha256') or '—')[:12]}` |",
        f"| Questions | {release['question_count']} |",
        f"| Tier | {run.get('tier') or '—'} |",
        f"| Run status | {'complete' if run['complete'] else 'aborted'} |",
        f"| Rerun of | {release.get('rerun_of') or '—'} |",
        f"| Provider / model | {generation['provider']} / `{generation['model']}` |",
        f"| Embedding | `{run['embedding']['identity']}` |",
        f"| Prompt / grounding check | `{generation['prompt_version']}` / "
        f"`{generation['grounding_check_version']}` |",
        f"| Retrieval | top_k {run['retrieval_policy']['top_k']}, min_similarity "
        f"{run['retrieval_policy']['min_similarity']}, visibility "
        f"{_cell(run['retrieval_policy'], 'visibility')} |",
        f"| Budget estimate | {_neurons(observed.get('estimated_neurons'))} neurons, "
        f"usage coverage {_share(observed.get('usage_coverage'))} |",
        f"| Tested | {release['scope']['tested']}; deployed image tested: "
        f"{'yes' if release['scope']['deployed_image_tested'] else 'no'} |",
        "",
        "| Acceptance gate | Result |",
        "| --- | --- |",
        *(f"| {name} | {value} |" for name, value in gates.items()),
        "",
    ]
    if release["reasons"]:
        lines += ["Not a release acceptance because:", ""]
        lines += [f"- {reason}" for reason in release["reasons"]]
        lines.append("")
    return lines


def _tree_state(dirty: Any) -> str:
    if dirty is None:
        return "unknown"
    return "dirty" if dirty else "clean"


def _operations_summary(operations: dict[str, Any] | None) -> list[str]:
    """Tier, budget and what the run cost — estimates named as estimates."""
    if not operations:
        return []
    forecast = operations.get("forecast") or {}
    observed = operations.get("observed") or {}
    lines = [
        "",
        "## Budget / Operations",
        "",
        "Neurons are estimates: reported tokens at published rates, calls without "
        "reported usage at the structural upper bound. The provider's own account is "
        "not visible here; consumption outside this tool is not included.",
        "",
        "| | |",
        "| --- | --- |",
        f"| Tier | {operations.get('tier', '—')} |",
        f"| Budget zone at start | {_cell(operations, 'zone').upper()} |",
        f"| Forecast | {_neurons(forecast.get('neurons'))} "
        f"({_share(operations.get('forecast_share'))} of daily, "
        f"{str(operations.get('cost_band') or '—').upper()}, "
        f"{forecast.get('confidence', '—')} confidence) |",
        f"| History | {forecast.get('profile_sources', '—')} run(s), "
        f"age {_cell(forecast, 'profile_age_days')} days, "
        f"{len(forecast.get('compatibility') or [])} compatibility note(s) |",
        f"| Observed estimate | {_neurons(observed.get('estimated_neurons'))} |",
        f"| Usage coverage | {_share(observed.get('usage_coverage'))} |",
        f"| Known local daily consumption | {_neurons(operations.get('known_daily_after'))} |",
        f"| Nominal daily budget | {_neurons(operations.get('daily_budget'))} |",
        f"| Estimated nominal reserve | {_neurons(operations.get('nominal_reserve_after'))} |",
        f"| Run status | {operations.get('status', '—')} |",
    ]
    guard = operations.get("guard")
    if guard:
        lines.append(
            f"| Budget guard | {guard.get('status', '—')}; projected final "
            f"{_neurons(guard.get('projected_final_neurons'))} "
            f"({_share(guard.get('projected_final_share'))}) |"
        )
    smoke = operations.get("smoke")
    if smoke is not None:
        verdict = "pass" if smoke.get("passed") else "FAIL"
        reasons = "; ".join(smoke.get("reasons") or []) or "—"
        lines.append(f"| Smoke | {verdict} ({reasons}) |")
    return lines


def _neurons(value: Any) -> str:
    return "—" if value is None else f"{float(value):,.0f}"


def _share(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.0f} %"


def _abort_notice(run: dict[str, Any]) -> list[str]:
    """A partial run says so before anything else. Older exports have no field."""
    aborted = run.get("aborted")
    if not aborted:
        return []
    reason = aborted.get("reason", "internal_defect")
    cause = (
        f"by an internal defect (`{aborted['error_type']}`)"
        if reason == "internal_defect"
        else f"by `{reason}`"
    )
    return [
        f"**Run aborted** at `{aborted['question_id']}` {cause}. This is a partial result "
        "of the questions before it, not a complete run.",
        "",
    ]


def _provider_call_summary(payload: dict[str, Any]) -> list[str]:
    """The provider-call section: counts by step, then every call not parsed.

    Exports written before calls were recorded have neither, and get nothing.
    """
    metrics = payload["metrics"].get("provider_calls")
    if metrics is None:
        return []
    columns = (
        ProviderCallType.GENERATION.value,
        ProviderCallType.GROUNDING_CHECK.value,
        "total",
    )

    def row(label: str, value: Any) -> str:
        return f"| {label} | " + " | ".join(str(value(metrics[name])) for name in columns) + " |"

    lines = [
        "",
        "## Provider calls",
        "",
        "Every request the answering service made to the generation provider. Tokens "
        "are summed over the calls whose provider reported usage, and only those. "
        "Elapsed time is one call as the answering service saw it: transport retries and, "
        "in a paced run, the pacing wait are included, so it is not provider latency.",
        "",
        "| | Generation | Grounding check | Total |",
        "| --- | --- | --- | --- |",
        row("Calls", lambda m: m["calls"]),
        row("Regenerations", lambda m: m["regenerations"]),
        row(
            "Parsed / unusable / provider error",
            lambda m: " / ".join(str(m["results"][result.value]) for result in CallResult),
        ),
        row("Usage reported", lambda m: f"{m['usage_reported']}/{m['calls']}"),
        row("Input tokens", lambda m: _dash(m["input_tokens"])),
        row("Output tokens", lambda m: _dash(m["output_tokens"])),
        row(
            "Elapsed seconds per call, median / max",
            lambda m: (
                f"{_dash(m['elapsed_seconds']['median'])} / {_dash(m['elapsed_seconds']['max'])}"
            ),
        ),
        row(
            "Finish reasons",
            lambda m: ", ".join(f"{k} {v}" for k, v in m["finish_reasons"].items()) or "—",
        ),
    ]

    unusable = [
        (question["id"], call)
        for question in payload["questions"]
        for call in question.get("provider_calls", ())
        if call["result"] != CallResult.PARSED.value
    ]
    if unusable:
        lines += [
            "",
            "### Calls that were not usable",
            "",
            "Sizes only; no reply text is kept. A visible count far below the character "
            "count is a reply made mostly of whitespace.",
            "",
            "| Question | Step | Attempt | Result | Detail | Finish reason "
            "| Characters / visible / tokens |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{question_id}` | {call['call_type']} | {call['attempt']} | {call['result']} | "
            f"{_cell(call, 'failure_detail')} | {_cell(call, 'finish_reason')} | "
            f"{_reply_size(call)} |"
            for question_id, call in unusable
        ]
    return lines


def _dash(value: Any) -> str:
    return "—" if value is None else str(value)


def _cell(facts: dict[str, Any], name: str) -> str:
    value = facts.get(name)
    return "—" if value is None else str(value)


def _reply_size(facts: dict[str, Any]) -> str:
    """Characters, non-whitespace characters and output tokens of one reply."""
    return " / ".join(
        _cell(facts, name)
        for name in ("reply_characters", "reply_visible_characters", "output_tokens")
    )


def _ratio(score: dict[str, int]) -> str:
    correct, total = score["correct"], score["total"]
    if total == 0:
        return "0/0"
    return f"{correct}/{total} ({100.0 * correct / total:.1f}%)"
