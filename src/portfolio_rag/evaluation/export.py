"""Writing one retrieval run down as data, so it can be read again later.

The stdout report answers "how did this run go?". This answers the questions
that come after it, without paying for a second run: which passages came back
for each question, at which similarity, and which of them the ground truth
accepts.

**Raw enough to re-threshold offline.** The similarity threshold is applied
after the vector store has returned its ``top_k`` matches, and only removes
some of them. A run at ``--min-similarity -1`` therefore records every match
the store returned, with its score, and the result of any higher threshold is
exactly those matches filtered by score. The export does not do that
filtering, sweep candidates or recommend a value: it keeps the data that makes
the question answerable.

**What is in the file is what may be published.** Question ids and text from
the dataset, document ids, headings and scores from the corpus, the embedding
space and the retrieval policy. Never a setting that is not listed here by
name: no credential, no account id, no index name, no endpoint. Passage text
and generated answers are not written either — neither is needed to judge
retrieval, and both would make the file a copy of things that live elsewhere.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from portfolio_rag import __version__
from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.evaluation.dataset import EvaluationDataset, EvaluationSuite, SuiteIdentity
from portfolio_rag.evaluation.metrics import (
    RetrievalOutcomeRecord,
    RetrievalReport,
    is_relevant,
)
from portfolio_rag.rag.policy import RetrievalPolicy
from portfolio_rag.rag.query import QUERY_REPRESENTATION_VERSION
from portfolio_rag.rag.retrieval import PUBLIC_VISIBILITY_FILTER

#: Paths whose changes do not count towards ``git_dirty``: where runs write
#: their own outputs (exports, summaries, the ledger). They are results of a
#: run, never inputs to one — without this, a run's own artifact would make the
#: next run on the same commit dirty. Recorded in every export, so the rule is
#: never implicit.
GIT_DIRTY_EXCLUDES: Final = ("evaluation/results",)

#: Identifies the shape of the file. Bumped when a field changes meaning or
#: disappears, so a reader can tell which shape it is holding.
EXPORT_FORMAT_VERSION: Final = "retrieval-eval-v1"


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Everything needed to say what a run measured, and nothing secret."""

    generated_at: datetime
    git_revision: str | None
    """``None`` when it could not be determined — never a guess."""

    git_dirty: bool | None
    """Whether the working tree had uncommitted changes. ``None`` when unknown."""

    dataset_path: str
    dataset_sha256: str
    dataset: EvaluationDataset
    """The dataset *as run* — for a suite, its selected questions."""

    dataset_question_count: int
    """Questions in the whole dataset file, whatever the suite selected."""

    suite: EvaluationSuite
    embedding: EmbeddingSpec
    vector_store: str
    policy: RetrievalPolicy
    retrieval_delay_seconds: float = 0.0
    """The pause the run kept between two questions."""

    run_id: str | None = None
    """The run's identifier — the correlation id its log lines carry. Random,
    never derived from anything secret. ``None`` for a report built outside
    the CLI, which has no run to name."""

    suite_identity: SuiteIdentity | None = None
    """The version and file hash of the suite that selected the questions.
    ``None`` for a report built outside the CLI."""

    git_tag: str | None = None
    """The release tag pointing at :attr:`git_revision`, when one does."""

    source_identity: str | None = None
    """The release-relevant content the run started from
    (:func:`~portfolio_rag.evaluation.acceptance.source_identity`). ``None``
    when it could not be determined."""


def dataset_fingerprint(path: Path) -> str:
    """SHA-256 of the dataset file, so a result names the exact ground truth."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_retrieval(report: RetrievalReport, metadata: RunMetadata) -> dict[str, Any]:
    """Describe *report* as plain JSON-ready data."""
    return {
        "format": EXPORT_FORMAT_VERSION,
        "run": _run(metadata),
        "metrics": _metrics(report),
        "questions": [_question(record) for record in report.records],
    }


def write_export(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* as UTF-8 JSON.

    ``allow_nan=False`` because a similarity that is not a number is a bug to
    surface, not a value to write down as something no JSON parser accepts.
    """
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")


def _run(metadata: RunMetadata) -> dict[str, Any]:
    spec = metadata.embedding
    suite = metadata.suite_identity
    return {
        "run_id": metadata.run_id,
        "generated_at": metadata.generated_at.isoformat(timespec="seconds"),
        # The package's one version (portfolio_rag.__version__), not a setting.
        "project_version": __version__,
        "git_revision": metadata.git_revision,
        "git_dirty": metadata.git_dirty,
        "git_dirty_excludes": list(GIT_DIRTY_EXCLUDES),
        "git_tag": metadata.git_tag,
        "release_source_identity": metadata.source_identity,
        "dataset": {
            "path": metadata.dataset_path,
            "sha256": metadata.dataset_sha256,
            "version": metadata.dataset.version,
            "corpus": metadata.dataset.corpus,
            "question_count": metadata.dataset_question_count,
        },
        "suite": metadata.suite.value,
        "suite_version": suite.version if suite else None,
        "suite_sha256": suite.sha256 if suite else None,
        "suite_path": suite.path if suite else None,
        "question_count": len(metadata.dataset.questions),
        "embedding": {
            "provider": spec.provider,
            "model": spec.model,
            "dimensions": spec.dimensions,
            "representation_version": spec.representation_version,
            "identity": spec.identity,
        },
        "query_representation_version": QUERY_REPRESENTATION_VERSION,
        "vector_store": metadata.vector_store,
        "retrieval_policy": {
            "top_k": metadata.policy.top_k,
            "min_similarity": metadata.policy.min_similarity,
            # Not a setting: the constant the retrieval service builds into
            # every query. Recorded so the artifact states it, not implies it.
            "visibility": PUBLIC_VISIBILITY_FILTER["visibility"],
        },
        "retrieval_delay_seconds": metadata.retrieval_delay_seconds,
    }


def _metrics(report: RetrievalReport) -> dict[str, Any]:
    """The stdout metrics, at the threshold this run used."""
    return {
        "answerable": len(report.answerable),
        "hit_rates": {
            score.label: {"correct": score.correct, "total": score.total}
            for score in report.hit_rates.values()
        },
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "threshold_rejections": {
            label: {"rejected": score.correct, "total": score.total}
            for label, score in sorted(report.threshold_rejections.items())
        },
        "internal_leaks": len(report.internal_leaks),
        "failures": [record.question.id for record in report.failures],
    }


def _question(record: RetrievalOutcomeRecord) -> dict[str, Any]:
    question = record.question
    outcome = record.outcome
    return {
        "id": question.id,
        "category": question.category.value,
        "question": question.question,
        "expects_an_answer": question.expects_an_answer,
        "expected_sources": [
            {"document": expected.document, "section": expected.section}
            for expected in question.expected_sources
        ],
        "first_relevant_rank": record.first_relevant_rank,
        "retrieval": {
            "matches_returned": None if outcome is None else outcome.matches_returned,
            "below_threshold": None if outcome is None else outcome.below_threshold,
            "unresolved": None if outcome is None else outcome.unresolved,
            "withheld": None if outcome is None else outcome.withheld,
            "retrieved": len(record.retrieved),
        },
        "hits": [
            _hit(rank, item, relevant=is_relevant(question, item))
            for rank, item in enumerate(record.retrieved, start=1)
        ],
    }


def _hit(rank: int, item: RetrievedChunk, *, relevant: bool) -> dict[str, Any]:
    chunk = item.chunk
    return {
        "rank": rank,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "section": chunk.section,
        "heading_path": list(chunk.heading_path),
        "similarity": item.similarity,
        "relevant": relevant,
    }
