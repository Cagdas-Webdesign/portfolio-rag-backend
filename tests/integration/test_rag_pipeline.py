"""The whole system, corpus side and query side, with nothing mocked out.

Markdown files on disk → ingestion → chunking → embedding → a vector index →
a question → retrieval → context → generation → citation validation → an
answer. No network, no credentials, no cost: the embedding provider and the
vector store are the real offline ones, and only the language model is
scripted, because a real one cannot be asked to take a specific branch on
demand.

This catches integration regressions that unit tests can miss: two correct
components disagreeing about the contract between them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.llm import DeterministicLLMProvider
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ingestion import load_knowledge_base
from portfolio_rag.ingestion.chunking import chunk_knowledge_base
from portfolio_rag.rag.policy import ContextPolicy, RetrievalPolicy
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import (
    INSUFFICIENT_KNOWLEDGE_ANSWER,
    AnswerOutcome,
    GroundedAnswerService,
)
from tests.doubles import LexicalEmbeddingProvider, ScriptedLLMProvider, grounded
from tests.support import run

FIXTURE_ROOT = Path("tests/fixtures/knowledge/rag")
SECRET_MARKER = "INTERNAL-ONLY-SECRET-VALUE"  # noqa: S105 - a fixture marker, not a secret


def build_pipeline(
    *,
    llm: object | None = None,
    retrieval_policy: RetrievalPolicy | None = None,
    root: Path = FIXTURE_ROOT,
) -> tuple[GroundedAnswerService, LexicalEmbeddingProvider, ScriptedLLMProvider]:
    """Run the corpus side for real, then wire the query side on top of it."""
    documents = load_knowledge_base(root)
    chunks = chunk_knowledge_base(documents)

    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(chunks))

    provider = llm or ScriptedLLMProvider(grounded("The service uses FastAPI.", "S1"))
    retrieval = PublicRetrievalService(
        embeddings=embeddings,
        store=store,
        resolver=InMemoryChunkResolver(chunks),
        policy=retrieval_policy or RetrievalPolicy(min_similarity=0.1),
    )
    service = GroundedAnswerService(
        retrieval=retrieval,
        llm=provider,  # type: ignore[arg-type]
        context_policy=ContextPolicy(),
    )
    return service, embeddings, provider  # type: ignore[return-value]


# --- the single-source golden flow -------------------------------------------


def test_a_question_the_corpus_answers_produces_a_grounded_answer_with_a_real_citation():
    service, _, _ = build_pipeline()

    answer = run(service.answer("Which HTTP framework does the service use?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert "FastAPI" in answer.answer
    (citation,) = answer.citations
    assert citation.document_id == "http-stack"
    assert citation.title == "HTTP Stack"
    assert citation.section == "Web framework"
    assert citation.source == "tests/fixtures/knowledge/rag/http-stack.md"


def test_the_cited_passage_is_the_one_that_actually_contains_the_fact():
    service, _, _ = build_pipeline()

    answer = run(service.answer("Which HTTP framework does the service use?"))

    assert answer.context is not None
    cited = answer.context.source("S1")
    assert cited is not None
    assert "FastAPI" in cited.retrieved.chunk.content


def test_the_whole_flow_needs_no_network_and_no_credential():
    """Asserted by construction: every adapter in this file is a local one."""
    service, embeddings, llm = build_pipeline()

    run(service.answer("Which HTTP framework does the service use?"))

    assert embeddings.spec.provider == "lexical-test"
    assert llm.call_count == 1


def test_the_same_question_produces_the_same_answer_twice():
    service, _, _ = build_pipeline()

    first = run(service.answer("Which HTTP framework does the service use?"))
    second = run(service.answer("Which HTTP framework does the service use?"))

    assert first.answer == second.answer
    assert first.citations == second.citations


# --- the multi-source golden flow --------------------------------------------


def test_a_question_needing_two_documents_can_cite_both():
    service, _, _ = build_pipeline(
        llm=ScriptedLLMProvider(
            grounded("FastAPI serves the API and documents are Markdown files.", "S1", "S2")
        )
    )

    answer = run(service.answer("Which framework serves the API and how are documents stored?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert len(answer.citations) == 2
    assert {citation.document_id for citation in answer.citations} == {
        "http-stack",
        "storage-layer",
    }


def test_both_documents_reach_the_context_under_distinct_labels():
    service, _, _ = build_pipeline(llm=ScriptedLLMProvider(grounded("Both.", "S1", "S2")))

    answer = run(service.answer("Which framework serves the API and how are documents stored?"))

    assert answer.context is not None
    assert len(answer.context.labels) >= 2
    assert len(set(answer.context.labels)) == len(answer.context.labels)


# --- the unknown question ----------------------------------------------------


def test_a_question_the_corpus_knows_nothing_about_is_answered_honestly():
    service, _, llm = build_pipeline(retrieval_policy=RetrievalPolicy(min_similarity=0.25))

    answer = run(service.answer("What is the maintainer's favourite pizza?"))

    assert answer.outcome is AnswerOutcome.NO_KNOWLEDGE
    assert answer.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert answer.citations == ()
    assert llm.call_count == 0, "no provider call for a question nothing supports"


def test_nothing_about_the_question_is_invented_in_the_refusal():
    service, _, _ = build_pipeline(retrieval_policy=RetrievalPolicy(min_similarity=0.25))

    answer = run(service.answer("What is the maintainer's favourite pizza?"))

    assert "pizza" not in answer.answer.lower()


def test_a_model_that_cannot_ground_an_answer_gets_the_same_honest_response():
    """The other insufficiency path: retrieval found something, the model could not use it."""
    service, _, llm = build_pipeline(
        llm=ScriptedLLMProvider(grounded("The passages do not cover this."))
    )

    answer = run(service.answer("Which HTTP framework does the service use?"))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.answer == INSUFFICIENT_KNOWLEDGE_ANSWER
    assert answer.citations == ()
    assert llm.call_count == 1, "this path does call the provider — that is the difference"


# --- internal knowledge ------------------------------------------------------


def test_the_internal_document_is_in_the_corpus_and_in_the_index():
    """Otherwise the leakage test below would prove nothing."""
    documents = load_knowledge_base(FIXTURE_ROOT)
    chunks = chunk_knowledge_base(documents)

    assert any(SECRET_MARKER in chunk.content for chunk in chunks)


def test_a_question_aimed_straight_at_internal_knowledge_retrieves_none_of_it():
    service, _, _ = build_pipeline()

    answer = run(service.answer("What is the confidential codename?"))

    assert all(SECRET_MARKER not in item.chunk.content for item in answer.retrieval.chunks)
    assert all(citation.document_id != "internal-notes" for citation in answer.citations)


def test_internal_content_never_reaches_the_context_or_the_provider():
    service, _, llm = build_pipeline()

    answer = run(service.answer("Tell me the confidential internal codename."))

    context_text = answer.context.text if answer.context is not None else ""
    sent = "\n".join(message.content for request in llm.requests for message in request.messages)

    assert SECRET_MARKER not in context_text
    assert SECRET_MARKER not in sent


def test_internal_content_never_reaches_the_answer():
    service, _, _ = build_pipeline(
        llm=ScriptedLLMProvider(grounded("Public information only.", "S1"))
    )

    answer = run(service.answer("What is the confidential codename?"))

    assert SECRET_MARKER not in answer.answer


def test_the_public_corpus_alone_answers_identically(tmp_path: Path):
    """Removing the internal document changes nothing a public client can see."""
    public_only = tmp_path / "public"
    public_only.mkdir()
    for name in ("http-stack.md", "storage-layer.md"):
        (public_only / name).write_text((FIXTURE_ROOT / name).read_text(), encoding="utf-8")

    with_internal, _, _ = build_pipeline()
    without_internal, _, _ = build_pipeline(root=public_only)

    question = "Which HTTP framework does the service use?"
    assert (
        run(with_internal.answer(question)).citations
        == run(without_internal.answer(question)).citations
    )


# --- the offline development stack -------------------------------------------


def test_the_shipped_offline_stack_runs_the_pipeline_end_to_end():
    """The deterministic LLM is a stub, but the plumbing it exercises is real."""
    service, _, _ = build_pipeline(llm=DeterministicLLMProvider())

    answer = run(service.answer("Which HTTP framework does the service use?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.citations, "the stub cites real labels, so the mapping is exercised"
    assert "development stub" in answer.answer.lower()


@pytest.mark.parametrize(
    "question",
    [
        "Welche Technologien werden verwendet?",
        "サービスは何を使っていますか?",
        "¿Qué framework se usa?",
    ],
)
def test_a_question_in_any_language_completes_the_pipeline(question: str):
    service, _, _ = build_pipeline(retrieval_policy=RetrievalPolicy(min_similarity=0.0))

    answer = run(service.answer(question))

    assert answer.answer
