"""The evaluation dataset: questions, and what a correct answer looks like.

Ground truth is **a document id and, where it matters, a section heading** —
not a chunk id. Chunk ids move when the chunking policy changes, and a dataset
that breaks on a re-chunk is a dataset that gets deleted rather than fixed. A
section is what a human actually verified: "the answer to this question is in
*that* part of *that* document."

Deliberately no framework. The file is YAML because the knowledge documents
already are, it is parsed through the same strict safe loader, and it is small
enough that every entry can be checked against the corpus by hand. An
evaluation nobody can audit is a number, not evidence.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from portfolio_rag.domain.knowledge import Slug
from portfolio_rag.ingestion.frontmatter import StrictSafeLoader


class EvaluationDatasetError(Exception):
    """The dataset could not be read or does not describe a usable run."""


class QuestionCategory(StrEnum):
    """Why a question is in the set, and therefore how it is scored.

    Three groups, not two, and the third one is the reason this enum has
    behaviour attached to it at all:

    * **answerable** — retrieval should find a specific passage. Hit rates and
      MRR are computed over exactly these.
    * **must refuse** — unknown and internal-only. The corpus has no public
      answer. Public near-matches are a calibration signal; the answer path
      must still refuse them.
    * **adversarial** — neither. An injection attempt *may* legitimately
      retrieve public passages: asking "ignore your instructions" surfaces the
      fixture document that contains those words, and retrieving a document is
      not obeying it. Scoring these as failed refusals would punish the system
      for behaving correctly. What must hold for them is structural — no
      internal content, no forged citation, no policy change — and that is
      asserted directly rather than averaged into a rate.
    """

    DIRECT = "direct"
    PARAPHRASED = "paraphrased"
    MULTI_SOURCE = "multi_source"
    SECTION = "section"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    INTERNAL = "internal"
    ADVERSARIAL = "adversarial"

    @property
    def expects_an_answer(self) -> bool:
        """Whether a correct run finds sources and answers the question."""
        return self in {
            QuestionCategory.DIRECT,
            QuestionCategory.PARAPHRASED,
            QuestionCategory.MULTI_SOURCE,
            QuestionCategory.SECTION,
            QuestionCategory.AMBIGUOUS,
        }

    @property
    def must_retrieve_nothing(self) -> bool:
        """Whether the question should retrieve no usable public evidence."""
        return self in {QuestionCategory.UNKNOWN, QuestionCategory.INTERNAL}


class ExpectedSource(BaseModel):
    """One passage a correct retrieval should surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: Slug
    section: str | None = Field(
        default=None,
        description="Innermost heading. Omitted when any part of the document counts.",
    )

    def matches(self, document_id: str, heading_path: tuple[str, ...]) -> bool:
        """Whether a retrieved chunk satisfies this expectation."""
        if document_id != self.document:
            return False
        if self.section is None:
            return True
        return self.section in heading_path


class EvaluationQuestion(BaseModel):
    """One question, its category, and the sources that would answer it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    category: QuestionCategory
    question: str = Field(min_length=1)
    expected_sources: tuple[ExpectedSource, ...] = ()

    @property
    def expects_an_answer(self) -> bool:
        return self.category.expects_an_answer

    @property
    def must_retrieve_nothing(self) -> bool:
        return self.category.must_retrieve_nothing


class EvaluationDataset(BaseModel):
    """A versioned question set, written against one corpus."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int = Field(ge=1)
    corpus: str = Field(
        min_length=1,
        description="Corpus directory, relative to the dataset file.",
    )
    questions: tuple[EvaluationQuestion, ...] = Field(min_length=1)

    def of_category(self, category: QuestionCategory) -> tuple[EvaluationQuestion, ...]:
        return tuple(question for question in self.questions if question.category is category)

    @property
    def answerable(self) -> tuple[EvaluationQuestion, ...]:
        """Questions where retrieval is supposed to find something."""
        return tuple(question for question in self.questions if question.expects_an_answer)


def load_dataset(path: Path) -> EvaluationDataset:
    """Read and validate a dataset file.

    Uses the ingestion layer's strict safe loader: hostile YAML cannot execute
    code here either, and a duplicate key is an error rather than a silent
    overwrite of somebody's ground truth.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationDatasetError(f"Evaluation dataset could not be read: {path}") from exc

    try:
        payload = yaml.load(raw, Loader=StrictSafeLoader)  # noqa: S506 - a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise EvaluationDatasetError(f"Evaluation dataset is not valid YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise EvaluationDatasetError("Evaluation dataset must be a mapping.")

    try:
        dataset = EvaluationDataset.model_validate(payload)
    except ValidationError as exc:
        raise EvaluationDatasetError(f"Evaluation dataset is invalid: {exc}") from exc

    _require_unique_ids(dataset)
    return dataset


def resolve_corpus(dataset_path: Path, dataset: EvaluationDataset) -> Path:
    """Locate the corpus the dataset was written against."""
    return (dataset_path.parent / dataset.corpus).resolve()


def _require_unique_ids(dataset: EvaluationDataset) -> None:
    seen: set[str] = set()
    for question in dataset.questions:
        if question.id in seen:
            raise EvaluationDatasetError(f"Duplicate question id in dataset: `{question.id}`")
        seen.add(question.id)
