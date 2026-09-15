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
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from portfolio_rag.evaluation.dataset import EvaluationDataset, EvaluationQuestion
from portfolio_rag.evaluation.metrics import (
    RetrievalReport,
    build_report,
    score_retrieval,
)
from portfolio_rag.rag.errors import QueryValidationError
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswerService


async def run_retrieval_evaluation(
    dataset: EvaluationDataset, retrieval: PublicRetrievalService
) -> RetrievalReport:
    """Score every question's retrieval. No provider generates anything."""
    records = []
    for question in dataset.questions:
        outcome = await retrieval.retrieve(normalize_query(question.question))
        records.append(score_retrieval(question, outcome.chunks))
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
    dataset: EvaluationDataset, answers: GroundedAnswerService
) -> GroundingReport:
    """Run the full pipeline for every question and record what came out."""
    records: list[GroundingRecord] = []
    for question in dataset.questions:
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
