"""A paced run against the dataset: same measurements, fewer provider calls.

The pacing exists for a live run against a rate-limited account, which this
suite will never make. What can be checked offline is the part that would
quietly corrupt a report if it were wrong: that pacing and retrying change
*when* the provider is called and never *what the run counted*.

The clock and the sleeper are injected, so a run that would pause for eight
seconds between generations takes none.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.evaluation import (
    GenerationPacing,
    PacedLLMProvider,
    load_dataset,
    resolve_corpus,
    run_grounding_evaluation,
)
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ingestion import load_knowledge_base
from portfolio_rag.ingestion.chunking import chunk_knowledge_base
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse
from portfolio_rag.rag.policy import DEFAULT_RETRIEVAL_POLICY, ContextPolicy, RetrievalPolicy
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswerService
from tests.doubles import DecidingLLMProvider, LexicalEmbeddingProvider
from tests.support import run

DATASET_PATH = Path("evaluation/questions.yaml")

INTERVAL_SECONDS = 8.0


class _Timeline:
    """Time that only moves when something waits for it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class _RateLimitedOnce:
    """A provider that refuses the first call it ever gets, then behaves.

    It delegates to the deciding double, so the answers a retried run produces
    are the same answers an unimpeded run produces — which is what makes the
    two reports comparable.
    """

    def __init__(self, inner: DecidingLLMProvider) -> None:
        self._inner = inner
        self.call_count = 0

    @property
    def model(self) -> str:
        return self._inner.model

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.call_count += 1
        if self.call_count == 1:
            raise LLMProviderError(
                "provider rate limit reached", retryable=True, retry_after_seconds=12.0
            )
        return await self._inner.generate(request)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def indexed(dataset):
    """The corpus side, indexed once, ready to be queried at any threshold."""
    documents = load_knowledge_base(resolve_corpus(DATASET_PATH, dataset))
    chunks = chunk_knowledge_base(documents)
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(chunks))
    return embeddings, store, InMemoryChunkResolver(chunks)


def _retrieval_at(indexed, policy: RetrievalPolicy) -> PublicRetrievalService:
    embeddings, store, resolver = indexed
    return PublicRetrievalService(
        embeddings=embeddings, store=store, resolver=resolver, policy=policy
    )


@pytest.fixture(scope="module")
def retrieval(indexed):
    return _retrieval_at(indexed, DEFAULT_RETRIEVAL_POLICY)


@pytest.fixture(scope="module")
def split_retrieval(dataset, indexed):
    """Retrieval tuned so roughly half the questions find nothing.

    The lexical double clears the default threshold on every question in the
    dataset, which would make "no generation happened" untestable. Rather than
    hardcoding a threshold that a chunk boundary could invalidate, this
    measures the questions and cuts at their median: deterministic for a given
    corpus, and still a split when the corpus moves.
    """
    permissive = _retrieval_at(indexed, RetrievalPolicy(min_similarity=0.0))
    best = sorted(
        max(
            (
                chunk.similarity
                for chunk in run(permissive.retrieve(normalize_query(question.question))).chunks
            ),
            default=0.0,
        )
        for question in dataset.questions
    )
    return _retrieval_at(indexed, RetrievalPolicy(min_similarity=best[len(best) // 2]))


def _deciding(dataset) -> DecidingLLMProvider:
    declines = tuple(
        question.question for question in dataset.questions if not question.expects_an_answer
    )
    return DecidingLLMProvider(declines=declines)


def _service(retrieval, llm) -> GroundedAnswerService:
    return GroundedAnswerService(retrieval=retrieval, llm=llm, context_policy=ContextPolicy())


def _paced(retrieval, llm, timeline: _Timeline) -> GroundedAnswerService:
    return _service(
        retrieval,
        PacedLLMProvider(
            llm,
            GenerationPacing(min_interval_seconds=INTERVAL_SECONDS),
            sleeper=timeline.sleep,
            clock=timeline.clock,
        ),
    )


# --- the run is unchanged by pacing -------------------------------------------


def test_a_paced_run_reports_exactly_what_an_unpaced_one_does(dataset, retrieval):
    plain = run(run_grounding_evaluation(dataset, _service(retrieval, _deciding(dataset))))
    paced = run(
        run_grounding_evaluation(dataset, _paced(retrieval, _deciding(dataset), _Timeline()))
    )

    assert len(paced.records) == len(plain.records) == len(dataset.questions)
    assert [record.outcome for record in paced.records] == [
        record.outcome for record in plain.records
    ]
    assert len(paced.failures) == len(plain.failures)


def test_a_retried_generation_still_counts_its_question_once(dataset, retrieval):
    """The one thing a retry must never do to a report.

    The retry happens behind the provider port, inside a single ``answer()``.
    Retrieval is not repeated, so hit rates, grounding, refusals and the
    failure list all see one question exactly once — while the provider itself
    was called one extra time.
    """
    timeline = _Timeline()
    llm = _RateLimitedOnce(_deciding(dataset))
    clean = run(run_grounding_evaluation(dataset, _service(retrieval, _deciding(dataset))))

    report = run(run_grounding_evaluation(dataset, _paced(retrieval, llm, timeline)))

    assert len(report.records) == len(dataset.questions)
    assert [record.question.id for record in report.records] == [
        record.question.id for record in clean.records
    ]
    assert report.answered == clean.answered
    assert report.no_knowledge == clean.no_knowledge
    assert report.not_grounded == clean.not_grounded
    assert 12.0 in timeline.slept, "the provider's own Retry-After was taken"


def test_a_question_retried_twice_is_still_answered_and_still_cited(dataset, retrieval):
    """The budget change must not reach the answer.

    Two refusals then a success is the worst case that still produces an
    answer. What comes out has to be indistinguishable from a run that was
    never rate limited — same outcome, same citations, one record.
    """

    class _RefusesTwice:
        def __init__(self, inner: DecidingLLMProvider) -> None:
            self._inner = inner
            self.calls = 0

        @property
        def model(self) -> str:
            return self._inner.model

        async def generate(self, request: GenerationRequest) -> GenerationResponse:
            self.calls += 1
            if self.calls <= 2:
                raise LLMProviderError(
                    "provider rate limit reached", retryable=True, retry_after_seconds=9.0
                )
            return await self._inner.generate(request)

    llm = _RefusesTwice(_deciding(dataset))
    clean = run(run_grounding_evaluation(dataset, _service(retrieval, _deciding(dataset))))

    report = run(run_grounding_evaluation(dataset, _paced(retrieval, llm, _Timeline())))

    assert len(report.records) == len(clean.records)
    assert [record.outcome for record in report.records] == [
        record.outcome for record in clean.records
    ]
    assert [record.citation_count for record in report.records] == [
        record.citation_count for record in clean.records
    ]
    assert report.answered == clean.answered


def test_questions_answered_without_a_model_never_wait(dataset, split_retrieval):
    """Pacing is spent on generations, not on questions.

    A question the corpus cannot answer short-circuits before the provider, so
    it consumes no slot — and a run whose waits matched the question count
    would be spending minutes on questions that cost nothing.
    """
    timeline = _Timeline()
    llm = _deciding(dataset)

    report = run(run_grounding_evaluation(dataset, _paced(split_retrieval, llm, timeline)))

    generated = sum(
        1 for record in report.records if record.outcome is not AnswerOutcome.NO_KNOWLEDGE
    )
    assert 0 < generated < len(report.records), "the dataset must exercise both paths"
    assert len(timeline.slept) == generated - 1, "one wait between generations, none around them"
    assert timeline.slept == [INTERVAL_SECONDS] * (generated - 1)
