"""Turning retrieval results into numbers that answer a specific question.

Four metrics, and each one exists because it answers something a developer
would otherwise have to guess at:

* **hit@k** — "was a passage that actually answers this among the first *k*?"
  Reported at 1, 3 and 5 because those are the decisions: what a model sees
  first, what it sees at all, and what the default ``top_k`` allows.
* **MRR** — "how *early* does the first correct passage appear?" Hit rates
  cannot tell rank 1 from rank 5; a corpus that always answers at rank 5 is
  working and is also one threshold change away from not working.
* **threshold rejection** — "for a question the corpus cannot answer, did the
  similarity threshold reject everything?" Reported as *calibration*, not as
  correctness, and the difference matters. Retrieval returns nearest
  neighbours; whether the nearest one is "close enough" is a threshold
  decision, and a threshold cannot tell a topic from a topic. Measured on this
  corpus, the similarity ranges of answerable and unanswerable questions
  overlap completely, so no value separates them — which is precisely why
  refusing is the *answer path's* job (`NOT_GROUNDED`) and not retrieval's.
  Expecting empty retrieval for an out-of-scope question would be a category
  error, and scoring it as failure would punish the system for behaving the
  way retrieval works.
* **internal leakage** — "did any question surface a non-public passage?" The
  one number here that is a hard gate rather than a measurement. It is
  structural, must be zero at every threshold, and is asserted as such.

Deliberately **not** a single aggregate score. On a set this small an average
is a way of not looking at the four questions that failed, and those four are
the entire value of running it.

Every count is reported alongside its denominator, and every failure keeps its
question id. A metric you cannot trace back to a question is a metric you
cannot act on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.evaluation.dataset import EvaluationQuestion

#: The cut-offs hit rate is reported at.
HIT_RATE_CUTOFFS: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True, slots=True)
class RetrievalOutcomeRecord:
    """What one question retrieved, and how well.

    ``first_relevant_rank`` is 1-based and ``None`` when nothing relevant was
    found at any depth — which is the distinction between "ranked badly" and
    "not found", and they need different fixes.
    """

    question: EvaluationQuestion
    retrieved: tuple[RetrievedChunk, ...]
    first_relevant_rank: int | None
    leaked_internal: bool

    @property
    def found_anything(self) -> bool:
        return bool(self.retrieved)

    def hit_at(self, cutoff: int) -> bool:
        return self.first_relevant_rank is not None and self.first_relevant_rank <= cutoff

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.first_relevant_rank is None else 1.0 / self.first_relevant_rank

    @property
    def rejected_by_threshold(self) -> bool:
        """For a question with no public answer: did the threshold reject everything?

        A calibration signal, not a verdict. ``False`` means near-matches
        scored above the threshold, which is normal and is why the answer path
        has its own refusal.
        """
        return not self.retrieved

    @property
    def structurally_safe(self) -> bool:
        """The property every question must satisfy, whatever its category.

        Adversarial questions are scored on this alone: they may retrieve
        public passages, and retrieving a document that contains an
        instruction is not the same as following it.
        """
        return not self.leaked_internal

    def describe_failure(self) -> str:
        """One line naming what went wrong, for the failure list."""
        if self.question.expects_an_answer:
            if not self.retrieved:
                return "retrieved nothing"
            if self.first_relevant_rank is None:
                found = ", ".join(
                    f"{item.chunk.document_id}/{item.chunk.section or '—'}"
                    for item in self.retrieved[:3]
                )
                return f"no expected source retrieved; got {found}"
            return f"expected source only at rank {self.first_relevant_rank}"
        if self.leaked_internal:
            return "retrieved an internal passage"
        return f"retrieved {len(self.retrieved)} passage(s) where none should be used"

    @property
    def is_failure(self) -> bool:
        """Whether this question did something the system must not do.

        For an answerable question that means missing its source entirely. For
        every other question it means one thing only: surfacing non-public
        content. Retrieving public near-matches for an unanswerable question is
        measured, reported and not a failure.
        """
        if self.question.expects_an_answer:
            return not self.hit_at(max(HIT_RATE_CUTOFFS))
        return not self.structurally_safe


def score_retrieval(
    question: EvaluationQuestion, retrieved: Sequence[RetrievedChunk]
) -> RetrievalOutcomeRecord:
    """Compare one question's retrieval against its ground truth."""
    rank: int | None = None
    for position, item in enumerate(retrieved, start=1):
        if _is_relevant(question, item):
            rank = position
            break

    return RetrievalOutcomeRecord(
        question=question,
        retrieved=tuple(retrieved),
        first_relevant_rank=rank,
        leaked_internal=any(
            item.chunk.document_metadata.visibility.value != "public" for item in retrieved
        ),
    )


def _is_relevant(question: EvaluationQuestion, item: RetrievedChunk) -> bool:
    return any(
        expected.matches(item.chunk.document_id, item.chunk.heading_path)
        for expected in question.expected_sources
    )


@dataclass(frozen=True, slots=True)
class CategoryScore:
    """Counts for one group of questions. Percentages are derived, never stored."""

    label: str
    total: int
    correct: int

    @property
    def percentage(self) -> float:
        return 0.0 if self.total == 0 else 100.0 * self.correct / self.total

    def describe(self) -> str:
        return f"{self.correct}/{self.total} ({self.percentage:.0f}%)"


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """The whole run: aggregate counts, and every individual failure."""

    records: tuple[RetrievalOutcomeRecord, ...]
    hit_rates: dict[int, CategoryScore] = field(default_factory=dict)
    mean_reciprocal_rank: float = 0.0
    threshold_rejections: dict[str, CategoryScore] = field(default_factory=dict)
    """Calibration per category, not correctness. See the module docstring."""

    @property
    def answerable(self) -> tuple[RetrievalOutcomeRecord, ...]:
        return tuple(record for record in self.records if record.question.expects_an_answer)

    @property
    def failures(self) -> tuple[RetrievalOutcomeRecord, ...]:
        """Every question that did not do what the dataset says it should."""
        return tuple(record for record in self.records if record.is_failure)

    @property
    def internal_leaks(self) -> tuple[RetrievalOutcomeRecord, ...]:
        """Any question at all that surfaced non-public content. Must be empty."""
        return tuple(record for record in self.records if record.leaked_internal)


def build_report(records: Sequence[RetrievalOutcomeRecord]) -> RetrievalReport:
    """Aggregate per-question records into the reported metrics."""
    ordered = tuple(records)
    answerable = [record for record in ordered if record.question.expects_an_answer]

    hit_rates = {
        cutoff: CategoryScore(
            label=f"hit@{cutoff}",
            total=len(answerable),
            correct=sum(1 for record in answerable if record.hit_at(cutoff)),
        )
        for cutoff in HIT_RATE_CUTOFFS
    }

    mrr = (
        sum(record.reciprocal_rank for record in answerable) / len(answerable)
        if answerable
        else 0.0
    )

    rejections = {
        category: CategoryScore(
            label=category,
            total=len(group),
            correct=sum(1 for record in group if record.rejected_by_threshold),
        )
        for category, group in _unanswerable_groups(ordered).items()
    }

    return RetrievalReport(
        records=ordered,
        hit_rates=hit_rates,
        mean_reciprocal_rank=mrr,
        threshold_rejections=rejections,
    )


def _unanswerable_groups(
    records: Sequence[RetrievalOutcomeRecord],
) -> dict[str, list[RetrievalOutcomeRecord]]:
    groups: dict[str, list[RetrievalOutcomeRecord]] = {}
    for record in records:
        if not record.question.must_retrieve_nothing:
            continue
        groups.setdefault(record.question.category.value, []).append(record)
    return groups
