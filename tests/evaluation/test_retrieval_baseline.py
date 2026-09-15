"""The evaluation dataset, run offline, with numbers that mean something.

**What the numbers here are, and what they are not.**

`DeterministicEmbeddingProvider` — the provider this project ships for offline
work — derives vectors from SHA-256 and understands nothing. Running the
dataset against it scores 0/14 on every hit rate, which says something true
about that provider and nothing about the pipeline. That run is recorded in the
Phase 6 report as the baseline it is.

This test uses the lexical double instead: hashed bag of words, real word
overlap, deterministic, offline, free. It is not a semantic model either, so
its hit rates are a **floor, not a prediction**. Three of the four questions it
misses are the `paraphrased` ones — questions written to share no vocabulary
with their answer, which is exactly what bag-of-words cannot do and exactly
what a real embedding model is for. Reading these percentages as "how good is
retrieval" would be wrong; they are a regression guard on the machinery.

What the assertions genuinely protect:

* the ground truth is reachable — corpus, chunking policy and `top_k` are
  compatible with the questions somebody wrote by hand;
* the pipeline routes a model's judgement correctly: a model that cites gets a
  grounded answer, a model that declines gets a refusal with no citations;
* **no question ever surfaces a non-public passage** — structural, must hold at
  every threshold including zero, and the one assertion here that is a gate
  rather than a measurement.

Thresholds are set at or just below the measured values so this fails when
something breaks, not when a chunk boundary moves by a sentence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.evaluation import (
    QuestionCategory,
    RetrievalReport,
    load_dataset,
    resolve_corpus,
    run_grounding_evaluation,
    run_retrieval_evaluation,
)
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ingestion import load_knowledge_base
from portfolio_rag.ingestion.chunking import chunk_knowledge_base
from portfolio_rag.rag.policy import DEFAULT_RETRIEVAL_POLICY, ContextPolicy, RetrievalPolicy
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import INSUFFICIENT_KNOWLEDGE_ANSWER, GroundedAnswerService
from tests.doubles import DecidingLLMProvider, LexicalEmbeddingProvider
from tests.support import run

DATASET_PATH = Path("evaluation/questions.yaml")

#: The marker that appears only in the `internal` fixture document.
CANARY = "CANARY-INTERNAL-ONLY-9F3A2B"

#: Bait planted in the public untrusted-content fixture: a label this backend
#: never mints, sitting in text a model will read.
FORGED_LABEL = "S99"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def corpus_chunks(dataset):
    documents = load_knowledge_base(resolve_corpus(DATASET_PATH, dataset))
    return chunk_knowledge_base(documents)


@pytest.fixture(scope="module")
def retrieval(corpus_chunks):
    """The real corpus side, then the real query side, on offline adapters."""
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(corpus_chunks))

    return PublicRetrievalService(
        embeddings=embeddings,
        store=store,
        resolver=InMemoryChunkResolver(corpus_chunks),
        policy=DEFAULT_RETRIEVAL_POLICY,
    )


@pytest.fixture(scope="module")
def report(dataset, retrieval) -> RetrievalReport:
    return run(run_retrieval_evaluation(dataset, retrieval))


@pytest.fixture(scope="module")
def grounding(dataset, retrieval):
    """Run the answering path with the model's judgement injected.

    Everything the corpus cannot publicly answer is declined, which is what a
    careful model does with those passages. What is measured is what the
    pipeline does next.
    """
    declines = tuple(
        question.question for question in dataset.questions if not question.expects_an_answer
    )
    answers = GroundedAnswerService(
        retrieval=retrieval,
        llm=DecidingLLMProvider(declines=declines),
        context_policy=ContextPolicy(),
    )
    return run(run_grounding_evaluation(dataset, answers))


# --- the dataset itself ------------------------------------------------------


def test_the_dataset_covers_every_category(dataset):
    """A set missing a category measures less than it appears to."""
    assert {question.category for question in dataset.questions} == set(QuestionCategory)


def test_every_expected_source_exists_in_the_corpus(dataset, corpus_chunks):
    """Ground truth pointing at nothing would make every hit rate meaningless."""
    available = {
        (chunk.document_id, section) for chunk in corpus_chunks for section in chunk.heading_path
    }

    for question in dataset.answerable:
        for expected in question.expected_sources:
            assert expected.section is not None, question.id
            assert (expected.document, expected.section) in available, question.id


def test_questions_with_no_public_answer_declare_no_sources(dataset):
    for question in dataset.questions:
        if not question.expects_an_answer:
            assert question.expected_sources == (), question.id


def test_the_corpus_contains_the_internal_document(corpus_chunks):
    """Otherwise every leakage assertion below would pass vacuously."""
    assert any(CANARY in chunk.content for chunk in corpus_chunks)


def test_the_corpus_contains_the_injection_bait(corpus_chunks):
    """Likewise: the forged label has to be there for its rejection to mean anything."""
    assert any(FORGED_LABEL in chunk.content for chunk in corpus_chunks)


# --- retrieval: a regression floor, not a quality bar ------------------------


def test_most_answerable_questions_find_their_source(report):
    score = report.hit_rates[5]

    assert score.correct >= 8, _explain(report)


def test_the_source_is_usually_ranked_near_the_top(report):
    assert report.hit_rates[1].correct >= 5, _explain(report)
    assert report.hit_rates[3].correct >= 8, _explain(report)


def test_relevant_passages_rank_early_rather_than_merely_appearing(report):
    assert report.mean_reciprocal_rank >= 0.45, _explain(report)


def test_a_deeper_cutoff_never_scores_worse(report):
    """A sanity check on the metric, not on retrieval."""
    assert report.hit_rates[1].correct <= report.hit_rates[3].correct
    assert report.hit_rates[3].correct <= report.hit_rates[5].correct


def test_the_only_retrieval_failures_are_missed_sources(report):
    """No failure may be a leak. Weak recall is a number; a leak is a bug."""
    for record in report.failures:
        assert record.question.expects_an_answer, record.question.id
        assert not record.leaked_internal, record.question.id


# --- the structural gates: these hold at any retrieval quality ---------------


def test_no_question_ever_retrieves_a_non_public_passage(report):
    """One leak is a failed run, whatever the hit rates say."""
    assert report.internal_leaks == ()


def test_the_canary_never_appears_in_any_retrieved_passage(report):
    for record in report.records:
        for item in record.retrieved:
            assert CANARY not in item.chunk.content, record.question.id


def test_every_retrieved_passage_is_public(report):
    for record in report.records:
        for item in record.retrieved:
            assert item.chunk.document_metadata.visibility.value == "public", record.question.id


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 0.25, 0.5])
def test_internal_content_stays_unreachable_at_every_threshold(
    dataset, corpus_chunks, threshold: float
):
    """The threshold is a relevance filter. It is not what keeps internal out."""
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(corpus_chunks))
    service = PublicRetrievalService(
        embeddings=embeddings,
        store=store,
        resolver=InMemoryChunkResolver(corpus_chunks),
        policy=RetrievalPolicy(top_k=50, min_similarity=threshold),
    )

    report = run(run_retrieval_evaluation(dataset, service))

    assert report.internal_leaks == ()


# --- grounding ---------------------------------------------------------------


def test_a_citing_model_produces_grounded_answers(grounding):
    correct, total = grounding.correct_for(expects_an_answer=True)

    assert correct == total, _explain_grounding(grounding)


def test_a_declining_model_produces_refusals_with_no_citations(grounding):
    correct, total = grounding.correct_for(expects_an_answer=False)

    assert correct == total, _explain_grounding(grounding)


def test_every_refusal_is_the_one_fixed_sentence(dataset, grounding):
    """A refusal must not be the model's own phrasing — it cannot contain a claim."""
    for record in grounding.records:
        if record.question.must_retrieve_nothing:
            assert record.answer == INSUFFICIENT_KNOWLEDGE_ANSWER, record.question.id


def test_no_answer_anywhere_contains_the_canary(grounding):
    for record in grounding.records:
        assert CANARY not in record.answer, record.question.id


def test_no_answer_anywhere_contains_the_forged_label(grounding):
    for record in grounding.records:
        assert FORGED_LABEL not in record.answer, record.question.id


def test_adversarial_questions_change_nothing_structural(dataset, grounding):
    """They may retrieve and be answered from public passages. Nothing else.

    Retrieving a document that contains an instruction is not obeying it, and a
    label copied out of that document is still not a source.
    """
    adversarial = {question.id for question in dataset.of_category(QuestionCategory.ADVERSARIAL)}

    for record in grounding.records:
        if record.question.id not in adversarial:
            continue
        assert CANARY not in record.answer, record.question.id
        assert FORGED_LABEL not in record.unknown_labels, record.question.id


# --- helpers -----------------------------------------------------------------


def _explain(report: RetrievalReport) -> str:
    lines = [
        "",
        "  ".join(f"{score.label}={score.describe()}" for score in report.hit_rates.values()),
        f"MRR={report.mean_reciprocal_rank:.3f}",
        "failures:",
    ]
    lines += [f"  {r.question.id}: {r.describe_failure()}" for r in report.failures]
    return "\n".join(lines)


def _explain_grounding(grounding) -> str:
    return "\n".join(
        ["", "grounding failures:"]
        + [
            f"  {record.question.id}: outcome={record.outcome.value} "
            f"citations={record.citation_count}"
            for record in grounding.failures
        ]
    )
