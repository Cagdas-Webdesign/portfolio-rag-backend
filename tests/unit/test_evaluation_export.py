"""The retrieval export: raw enough to re-threshold offline, and nothing secret.

Records are built by hand from retrieval outcomes with known scores, so every
number in the file can be checked against the one that went in.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.evaluation import (
    EXPORT_FORMAT_VERSION,
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationSuite,
    ExpectedSource,
    QuestionCategory,
    RetrievalOutcomeRecord,
    RunMetadata,
    build_report,
    export_retrieval,
    run_retrieval_evaluation,
    score_retrieval,
    write_export,
)
from portfolio_rag.rag.policy import RetrievalPolicy
from portfolio_rag.rag.query import UserQuery
from portfolio_rag.rag.retrieval import PublicRetrievalService, RetrievalOutcome
from tests.doubles import make_chunk
from tests.support import run

ANSWERABLE = EvaluationQuestion(
    id="q-answerable",
    category=QuestionCategory.DIRECT,
    question="Which framework serves the API?",
    expected_sources=(ExpectedSource(document="backend", section="API"),),
)
UNKNOWN = EvaluationQuestion(
    id="q-unknown", category=QuestionCategory.UNKNOWN, question="What is his salary?"
)
DATASET = EvaluationDataset(version=3, corpus="../knowledge", questions=(ANSWERABLE, UNKNOWN))

SPEC = EmbeddingSpec(
    provider="mistral",
    model="mistral-embed",
    dimensions=1024,
    representation_version="embedding-text-v1",
)


def hit(chunk_id: str, document_id: str, section: str, similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=make_chunk(chunk_id, "text", document_id=document_id, heading_path=("Doc", section)),
        similarity=similarity,
    )


#: Ranked as a store returns them: the relevant passage is second, and the
#: scores carry every digit a float can hold.
ANSWERABLE_HITS = (
    hit("other--0001", "other", "Intro", 0.8123456789012345),
    hit("backend--0002", "backend", "API", 0.7012345678901234),
    hit("backend--0003", "backend", "Storage", -0.0312345678901234),
)
ANSWERABLE_OUTCOME = RetrievalOutcome(
    chunks=ANSWERABLE_HITS,
    matches_returned=6,
    below_threshold=1,
    unresolved=1,
    withheld=1,
    duration_seconds=0.01,
)
UNKNOWN_OUTCOME = RetrievalOutcome(
    chunks=(hit("other--0001", "other", "Intro", 0.31),),
    matches_returned=5,
    below_threshold=4,
    unresolved=0,
    withheld=0,
    duration_seconds=0.01,
)


def metadata(**overrides: Any) -> RunMetadata:
    fields: dict[str, Any] = {
        "generated_at": datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
        "git_revision": "abc123",
        "git_dirty": False,
        "dataset_path": "evaluation/portfolio-questions.yaml",
        "dataset_sha256": "f" * 64,
        "dataset": DATASET,
        "dataset_question_count": 49,
        "suite": EvaluationSuite.SMOKE,
        "embedding": SPEC,
        "vector_store": "vectorize",
        "policy": RetrievalPolicy(top_k=5, min_similarity=-1.0),
    }
    fields.update(overrides)
    return RunMetadata(**fields)


def records() -> list[RetrievalOutcomeRecord]:
    return [
        score_retrieval(ANSWERABLE, ANSWERABLE_OUTCOME.chunks, outcome=ANSWERABLE_OUTCOME),
        score_retrieval(UNKNOWN, UNKNOWN_OUTCOME.chunks, outcome=UNKNOWN_OUTCOME),
    ]


def exported(tmp_path: Path) -> dict[str, Any]:
    """Through the file, as a later reader would see it."""
    path = tmp_path / "run.json"
    write_export(path, export_retrieval(build_report(records()), metadata()))
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_the_export_is_valid_json(tmp_path: Path):
    """Case F."""
    data = exported(tmp_path)

    assert data["format"] == EXPORT_FORMAT_VERSION


def test_the_run_metadata_is_recorded(tmp_path: Path):
    """Case G."""
    run_ = exported(tmp_path)["run"]

    assert run_["generated_at"] == "2026-09-28T12:00:00+00:00"
    assert (run_["git_revision"], run_["git_dirty"]) == ("abc123", False)
    assert run_["dataset"] == {
        "path": "evaluation/portfolio-questions.yaml",
        "sha256": "f" * 64,
        "version": 3,
        "corpus": "../knowledge",
        "question_count": 49,
    }
    assert (run_["suite"], run_["question_count"]) == ("smoke", 2)
    assert run_["embedding"] == {
        "provider": "mistral",
        "model": "mistral-embed",
        "dimensions": 1024,
        "representation_version": "embedding-text-v1",
        "identity": SPEC.identity,
    }
    assert run_["query_representation_version"] == "query-text-v1"
    assert run_["vector_store"] == "vectorize"
    assert run_["retrieval_policy"] == {
        "top_k": 5,
        "min_similarity": -1.0,
        "visibility": "public",
    }
    assert run_["git_dirty_excludes"] == ["evaluation/results"]


def test_an_unknown_git_revision_is_recorded_as_unknown(tmp_path: Path):
    path = tmp_path / "run.json"
    payload = export_retrieval(build_report(records()), metadata(git_revision=None, git_dirty=None))
    write_export(path, payload)

    run_ = json.loads(path.read_text(encoding="utf-8"))["run"]
    assert (run_["git_revision"], run_["git_dirty"]) == (None, None)


def test_each_question_carries_its_identity_and_ground_truth(tmp_path: Path):
    """Case H."""
    first, second = exported(tmp_path)["questions"]

    assert (first["id"], first["category"], first["question"]) == (
        "q-answerable",
        "direct",
        "Which framework serves the API?",
    )
    assert first["expected_sources"] == [{"document": "backend", "section": "API"}]
    assert first["expects_an_answer"] is True
    assert (second["id"], second["category"], second["expected_sources"]) == (
        "q-unknown",
        "unknown",
        [],
    )


def test_every_hit_is_exported_with_its_passage_identity(tmp_path: Path):
    """Case H."""
    hits = exported(tmp_path)["questions"][0]["hits"]

    assert [(h["chunk_id"], h["document_id"], h["section"]) for h in hits] == [
        ("other--0001", "other", "Intro"),
        ("backend--0002", "backend", "API"),
        ("backend--0003", "backend", "Storage"),
    ]
    assert hits[1]["heading_path"] == ["Doc", "API"]


def test_similarity_scores_survive_the_file_exactly(tmp_path: Path):
    """Case I: numbers, not strings, and not rounded — negative ones included."""
    hits = exported(tmp_path)["questions"][0]["hits"]

    assert [h["similarity"] for h in hits] == [item.similarity for item in ANSWERABLE_HITS]
    assert all(isinstance(h["similarity"], float) for h in hits)


def test_the_ranking_is_kept(tmp_path: Path):
    """Case J."""
    hits = exported(tmp_path)["questions"][0]["hits"]

    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert exported(tmp_path)["questions"][0]["first_relevant_rank"] == 2


def test_relevance_is_what_the_expected_sources_say(tmp_path: Path):
    """Case K: the same rule `ExpectedSource.matches` applies."""
    hits = exported(tmp_path)["questions"][0]["hits"]

    for exported_hit, item in zip(hits, ANSWERABLE_HITS, strict=True):
        expected = any(
            source.matches(item.chunk.document_id, item.chunk.heading_path)
            for source in ANSWERABLE.expected_sources
        )
        assert exported_hit["relevant"] is expected
    assert [h["relevant"] for h in hits] == [False, True, False]


def test_the_retrieval_counts_are_carried_over(tmp_path: Path):
    """Case L."""
    first, second = exported(tmp_path)["questions"]

    assert first["retrieval"] == {
        "matches_returned": 6,
        "below_threshold": 1,
        "unresolved": 1,
        "withheld": 1,
        "retrieved": 3,
    }
    assert second["retrieval"]["below_threshold"] == 4


def test_a_record_without_an_outcome_exports_unknown_counts(tmp_path: Path):
    path = tmp_path / "run.json"
    bare = [score_retrieval(ANSWERABLE, ANSWERABLE_HITS)]
    write_export(path, export_retrieval(build_report(bare), metadata()))

    counts = json.loads(path.read_text(encoding="utf-8"))["questions"][0]["retrieval"]
    assert counts["below_threshold"] is None
    assert counts["retrieved"] == 3


def test_the_exported_metrics_are_the_reported_ones(tmp_path: Path):
    """Case M."""
    report = build_report(records())
    metrics = exported(tmp_path)["metrics"]

    assert metrics["hit_rates"] == {
        score.label: {"correct": score.correct, "total": score.total}
        for score in report.hit_rates.values()
    }
    assert metrics["hit_rates"]["hit@1"] == {"correct": 0, "total": 1}
    assert metrics["hit_rates"]["hit@3"] == {"correct": 1, "total": 1}
    assert metrics["mean_reciprocal_rank"] == report.mean_reciprocal_rank == 0.5
    assert metrics["threshold_rejections"] == {"unknown": {"rejected": 0, "total": 1}}
    assert metrics["internal_leaks"] == 0
    assert metrics["failures"] == []


def test_keeping_the_outcome_changes_no_metric():
    """Case M: the record gained an observation, not a different score."""
    with_outcome = build_report(records())
    without = build_report(
        [
            score_retrieval(ANSWERABLE, ANSWERABLE_OUTCOME.chunks),
            score_retrieval(UNKNOWN, UNKNOWN_OUTCOME.chunks),
        ]
    )

    assert with_outcome.hit_rates == without.hit_rates
    assert with_outcome.mean_reciprocal_rank == without.mean_reciprocal_rank
    assert with_outcome.threshold_rejections == without.threshold_rejections


def test_no_passage_text_is_written(tmp_path: Path):
    text = json.dumps(exported(tmp_path))

    assert '"content"' not in text
    assert '"text"' not in text


# --- the runner keeps what retrieval reported ----------------------------------


class _ScriptedRetrieval:
    """Returns a prepared outcome per question — the counts are the point."""

    def __init__(self, outcomes: dict[str, RetrievalOutcome]) -> None:
        self._outcomes = outcomes

    async def retrieve(self, query: UserQuery) -> RetrievalOutcome:
        return self._outcomes[query.text]


def test_the_runner_keeps_the_whole_outcome_on_each_record():
    """Case L: before this, only `outcome.chunks` survived the runner."""
    retrieval = _ScriptedRetrieval(
        {ANSWERABLE.question: ANSWERABLE_OUTCOME, UNKNOWN.question: UNKNOWN_OUTCOME}
    )

    report = run(run_retrieval_evaluation(DATASET, cast(PublicRetrievalService, retrieval)))

    assert [record.outcome for record in report.records] == [ANSWERABLE_OUTCOME, UNKNOWN_OUTCOME]
    assert report.records[0].retrieved == ANSWERABLE_HITS


def test_the_retrieval_delay_is_recorded(tmp_path: Path):
    """Case J: which pacing a baseline was taken under stays on record."""
    path = tmp_path / "run.json"
    write_export(
        path, export_retrieval(build_report(records()), metadata(retrieval_delay_seconds=3.0))
    )

    assert json.loads(path.read_text(encoding="utf-8"))["run"]["retrieval_delay_seconds"] == 3.0


def test_an_unpaced_run_records_a_delay_of_zero(tmp_path: Path):
    assert exported(tmp_path)["run"]["retrieval_delay_seconds"] == 0.0
