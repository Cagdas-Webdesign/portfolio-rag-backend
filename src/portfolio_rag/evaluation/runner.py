"""Running the dataset against a wired retrieval and answering stack.

Two runs, kept apart on purpose.

**Retrieval** asks whether the right passages come back, and needs no language
model at all. **Grounding** asks what the pipeline does with them: does it
answer when it can, refuse when it cannot, and publish only citations the
backend verified. Averaging those two into one "RAG score" is how a system with
excellent retrieval and broken citation validation gets a good grade.

The runner is handed its services rather than building them. That is what lets
the same dataset run against the offline stack in CI and against real providers
from the CLI, with nothing swapped but the wiring.

**Pacing between questions is the runner's, and only the runner's.** Every
question costs one query embedding, and a dataset run asks them back to back —
a burst no visitor produces. ``delay_seconds`` waits that long *between* two
questions: never before the first, never after the last, so N questions make
at most N - 1 pauses. It lives here rather than in the retrieval service or an
adapter because it is a property of a batch run; the chat path never reaches
this module and cannot acquire it.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from portfolio_rag.core.errors import AppError
from portfolio_rag.core.logging import get_logger
from portfolio_rag.evaluation.dataset import EvaluationDataset, EvaluationQuestion
from portfolio_rag.evaluation.e2e import (
    AbortReason,
    E2ERecord,
    E2EReport,
    RunAbort,
    errored_record,
    score_answer,
)
from portfolio_rag.evaluation.metrics import (
    RetrievalReport,
    build_report,
    score_retrieval,
)
from portfolio_rag.evaluation.pacing import Sleeper
from portfolio_rag.rag.errors import (
    GenerationUnavailableError,
    QueryEmbeddingError,
    QueryValidationError,
    RetrievalUnavailableError,
)
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswerService

_logger = get_logger(__name__)


def validate_question_delay(delay_seconds: float) -> float:
    """A usable pause between questions: finite and not negative."""
    if not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("the delay between questions must be a finite number, 0 or more")
    return delay_seconds


async def _pause_between(position: int, delay_seconds: float, sleeper: Sleeper | None) -> None:
    """Wait before every question but the first, and only when asked to."""
    if position > 0 and delay_seconds > 0:
        await (sleeper or asyncio.sleep)(delay_seconds)


async def run_retrieval_evaluation(
    dataset: EvaluationDataset,
    retrieval: PublicRetrievalService,
    *,
    delay_seconds: float = 0.0,
    sleeper: Sleeper | None = None,
) -> RetrievalReport:
    """Score every question's retrieval. No provider generates anything.

    *delay_seconds* pauses between questions; see the module docstring.
    """
    validate_question_delay(delay_seconds)
    records = []
    for position, question in enumerate(dataset.questions):
        await _pause_between(position, delay_seconds, sleeper)
        outcome = await retrieval.retrieve(normalize_query(question.question))
        records.append(score_retrieval(question, outcome.chunks, outcome=outcome))
    return build_report(records)


@dataclass(frozen=True, slots=True)
class GroundingRecord:
    """What the whole pipeline did with one question."""

    question: EvaluationQuestion
    outcome: AnswerOutcome
    citation_count: int
    unknown_labels: tuple[str, ...]
    answer: str
    duration_seconds: float

    @property
    def behaved_correctly(self) -> bool:
        """Whether the outcome matches what the dataset says should happen.

        An answerable question should be answered with at least one verified
        citation. A question the corpus cannot answer publicly should end in a
        refusal with none. An adversarial question is judged on neither: it may
        legitimately retrieve and be answered from public passages, and what
        must hold for it is structural — no internal content, no forged
        citation — which the evaluation asserts directly.

        What is *not* checked anywhere here: whether the answer text is any
        good. That needs a human or a judge model, and this harness claims
        neither.
        """
        if self.question.expects_an_answer:
            return self.outcome is AnswerOutcome.ANSWERED and self.citation_count > 0
        if self.question.must_retrieve_nothing:
            return self.outcome is not AnswerOutcome.ANSWERED and self.citation_count == 0
        return True


@dataclass(frozen=True, slots=True)
class GroundingReport:
    """Aggregate behaviour of the answering path."""

    records: tuple[GroundingRecord, ...]

    @property
    def answered(self) -> int:
        return sum(1 for record in self.records if record.outcome is AnswerOutcome.ANSWERED)

    @property
    def no_knowledge(self) -> int:
        return sum(1 for record in self.records if record.outcome is AnswerOutcome.NO_KNOWLEDGE)

    @property
    def not_grounded(self) -> int:
        return sum(1 for record in self.records if record.outcome is AnswerOutcome.NOT_GROUNDED)

    @property
    def failures(self) -> tuple[GroundingRecord, ...]:
        return tuple(record for record in self.records if not record.behaved_correctly)

    def correct_for(self, expects_an_answer: bool) -> tuple[int, int]:
        """(correct, total) for the answerable or the must-refuse half.

        Adversarial questions are in neither: they are scored structurally.
        """
        group = [
            record
            for record in self.records
            if (
                record.question.expects_an_answer
                if expects_an_answer
                else record.question.must_retrieve_nothing
            )
        ]
        return sum(1 for record in group if record.behaved_correctly), len(group)

    @property
    def slowest(self) -> GroundingRecord | None:
        return max(self.records, key=lambda record: record.duration_seconds, default=None)


async def run_grounding_evaluation(
    dataset: EvaluationDataset,
    answers: GroundedAnswerService,
    *,
    delay_seconds: float = 0.0,
    sleeper: Sleeper | None = None,
) -> GroundingReport:
    """Run the full pipeline for every question and record what came out.

    Each answer embeds its question again, so the same pause applies here.
    It is not counted in a question's duration.
    """
    validate_question_delay(delay_seconds)
    records: list[GroundingRecord] = []
    for position, question in enumerate(dataset.questions):
        await _pause_between(position, delay_seconds, sleeper)
        started = time.perf_counter()
        try:
            answer = await answers.answer(question.question)
        except QueryValidationError:  # pragma: no cover - dataset questions are valid
            continue
        records.append(
            GroundingRecord(
                question=question,
                outcome=answer.outcome,
                citation_count=len(answer.citations),
                unknown_labels=answer.unknown_labels,
                answer=answer.answer,
                duration_seconds=round(time.perf_counter() - started, 4),
            )
        )
    return GroundingReport(records=tuple(records))


#: Guard stops that cut the question they were observed on short: it ended
#: *because* the provider stopped serving, not on its own terms.
_INTERRUPTING: Final = frozenset(
    {
        AbortReason.RATE_LIMITED,
        AbortReason.AUTH_FAILURE,
        AbortReason.SYSTEMIC_PROVIDER_FAILURE,
        AbortReason.RETRIEVAL_UNAVAILABLE,
    }
)


async def run_e2e_evaluation(
    dataset: EvaluationDataset,
    answers: GroundedAnswerService,
    *,
    delay_seconds: float = 0.0,
    sleeper: Sleeper | None = None,
    guard: QuestionGuard | None = None,
    completed: Sequence[E2ERecord] = (),
    on_record: Callable[[tuple[E2ERecord, ...]], None] | None = None,
    pause_on_last_question: bool = False,
) -> E2EReport:
    """Ask every question once and record the whole pass.

    One pipeline call per question, so one query embedding, one vector search
    and at most one generation each. A question the pipeline raises on is
    recorded as an error and the run goes on: a provider failing on question 30
    must not cost the 29 answers already paid for. Its retrieval is still
    scored when there was one.

    **Only failures the pipeline itself classified are recorded and passed.**
    Those are :class:`AppError` — every provider failure, including one the
    port did not define, arrives as one. Anything else is a defect in this
    code: it is not dressed up as a provider problem and the run does not
    carry on past it. The run ends there, the report says so, and the records
    measured before it are kept.

    *guard*, when given, is asked after every question whether the run goes
    on (`evaluation.operations`): a provider that said no more, one that keeps
    failing, a run heading past the budget. When it says stop, no further
    question is asked and the report is the partial one, marked aborted.

    **Resuming.** *completed* are the records of questions an earlier segment
    of the same logical run already finished — the first questions of the
    dataset, in order. They are not asked again: no embedding, no search, no
    generation. The report holds them followed by the questions asked now.
    *on_record* is called with every record so far after each question the
    guard let the run continue past, so a checkpoint can be written before the
    next one starts. With *pause_on_last_question*, a provider stop on the last
    question ends the run as aborted like on any other, so that question can be
    asked again rather than standing as the provider's failure.
    """
    validate_question_delay(delay_seconds)
    ids = [question.id for question in dataset.questions]
    done = [record.question.id for record in completed]
    if done != ids[: len(done)]:
        raise ValueError("completed records must be the dataset's first questions, in order")
    records: list[E2ERecord] = list(completed)
    for position, question in enumerate(dataset.questions[len(done) :]):
        await _pause_between(position, delay_seconds, sleeper)
        started = time.perf_counter()
        try:
            answer = await answers.answer(question.question)
        except AppError as exc:
            duration = round(time.perf_counter() - started, 4)
            # A failed generation says why, and what had been retrieved before
            # it failed. Any other failure carries neither.
            generation = exc if isinstance(exc, GenerationUnavailableError) else None
            retrieval_failure = (
                exc.failure
                if isinstance(exc, (QueryEmbeddingError, RetrievalUnavailableError))
                else None
            )
            records.append(
                errored_record(
                    question,
                    exc.code.value,
                    duration_seconds=duration,
                    retrieval=generation.retrieval if generation else None,
                    failure=generation.failure if generation else None,
                    provider_calls=generation.provider_calls if generation else (),
                    retrieval_failure=retrieval_failure,
                )
            )
        except Exception as exc:  # an internal defect: stop, keep what was measured
            _logger.exception(
                "evaluation aborted by an internal defect",
                extra={"question_id": question.id, "error_type": type(exc).__name__},
            )
            return E2EReport(
                records=tuple(records),
                aborted=RunAbort(question_id=question.id, error_type=type(exc).__name__),
            )
        else:
            duration = round(time.perf_counter() - started, 4)
            records.append(score_answer(question, answer, duration_seconds=duration))

        is_last = len(records) == len(dataset.questions)
        reason = guard.observe(records[-1]) if guard is not None else None
        if reason is not None and (
            not is_last or (pause_on_last_question and reason in _INTERRUPTING)
        ):
            _logger.warning(
                "evaluation stopped by its guard",
                extra={"question_id": question.id, "reason": reason.value},
            )
            return E2EReport(
                records=tuple(records), aborted=RunAbort(question_id=question.id, reason=reason)
            )
        if on_record is not None:
            on_record(tuple(records))
    return E2EReport(records=tuple(records))


class QuestionGuard(Protocol):
    """Asked after each question whether a run goes on. See `evaluation.operations`."""

    def observe(self, record: E2ERecord) -> AbortReason | None: ...
