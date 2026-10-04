"""A small real stack for end-to-end evaluation tests.

Two chunks, the in-memory index, a lexical embedding and the real answering
service — with the model's decision left to the caller. Shared by the tests of
the evaluation itself and by the tests of what it exports about provider calls,
so that neither has to reach into the other's module.
"""

from __future__ import annotations

from datetime import UTC, datetime

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.evaluation import (
    E2ERunMetadata,
    EvaluationDataset,
    EvaluationQuestion,
    EvaluationSuite,
    ExpectedSource,
    QuestionCategory,
    RunMetadata,
    corpus_identity,
)
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ports.llm import LLMProvider
from portfolio_rag.rag.policy import ContextPolicy, RetrievalPolicy
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import GroundedAnswerService
from tests.doubles import LexicalEmbeddingProvider, ScriptedLLMProvider, make_chunk
from tests.support import run

CHUNKS = (
    make_chunk(
        "stack--0000",
        "FastAPI serves the HTTP API of the backend.",
        document_id="stack",
        heading_path=("Stack", "API"),
    ),
    make_chunk(
        "deploy--0000",
        "Docker ships the backend as a container image.",
        document_id="deploy",
        heading_path=("Deploy",),
    ),
)

ANSWERABLE = EvaluationQuestion(
    id="q-api",
    category=QuestionCategory.DIRECT,
    question="Which framework serves the HTTP API?",
    expected_sources=(ExpectedSource(document="stack", section="API"),),
)


def e2e_retrieval() -> PublicRetrievalService:
    """Retrieval over :data:`CHUNKS`, with every match admitted."""
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(CHUNKS))
    return PublicRetrievalService(
        embeddings=embeddings,
        store=store,
        resolver=InMemoryChunkResolver(CHUNKS),
        policy=RetrievalPolicy(min_similarity=-1.0),
    )


def label_of(document_id: str, question: EvaluationQuestion) -> str:
    """The context label *document_id* gets for *question*: S1 is the top match."""
    chunks = run(e2e_retrieval().retrieve(normalize_query(question.question))).chunks
    return f"S{[chunk.document_id for chunk in chunks].index(document_id) + 1}"


def e2e_service(llm: LLMProvider) -> GroundedAnswerService:
    """The real answering service over :func:`e2e_retrieval`, generating with *llm*."""
    return GroundedAnswerService(retrieval=e2e_retrieval(), llm=llm, context_policy=ContextPolicy())


def dataset_of(*questions: EvaluationQuestion) -> EvaluationDataset:
    return EvaluationDataset(version=1, corpus="corpus", questions=questions)


def run_metadata(dataset: EvaluationDataset, *, notes: tuple[str, ...] = ()) -> E2ERunMetadata:
    """Fixed run metadata for exporting a report of *dataset*."""
    retrieval = e2e_retrieval()
    return E2ERunMetadata(
        retrieval=RunMetadata(
            generated_at=datetime(2026, 10, 1, 12, 0, tzinfo=UTC),
            git_revision="a" * 40,
            git_dirty=True,
            dataset_path="evaluation/portfolio-questions.yaml",
            dataset_sha256="f" * 64,
            dataset=dataset,
            dataset_question_count=len(dataset.questions),
            suite=EvaluationSuite.FULL,
            embedding=retrieval.spec,
            vector_store="memory",
            policy=retrieval.policy,
        ),
        generation_provider="scripted",
        generation_model=ScriptedLLMProvider.MODEL,
        prompt_version="grounded-answer-v3",
        context_policy=ContextPolicy(),
        generation_delay_seconds=8.0,
        corpus=corpus_identity(CHUNKS),
        notes=notes,
    )
