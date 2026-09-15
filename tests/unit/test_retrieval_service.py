"""Public retrieval: what it finds, what it refuses, and what it cannot be asked.

Run against the real in-memory store and a real embedding provider rather than
mocks, so what is asserted is behaviour rather than call sequences.
"""

from __future__ import annotations

import pytest

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    VectorIndexSpec,
    VectorMetadata,
    VectorRecord,
)
from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.infrastructure.embedding import DeterministicEmbeddingProvider
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ingestion.embedding import build_embedding_text
from portfolio_rag.ports.embeddings import EmbeddingResult
from portfolio_rag.ports.vector_store import VectorQuery
from portfolio_rag.rag.errors import (
    EmbeddingSpaceMismatchError,
    QueryEmbeddingError,
    RetrievalUnavailableError,
)
from portfolio_rag.rag.policy import RetrievalPolicy
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PUBLIC_VISIBILITY_FILTER, PublicRetrievalService
from tests.doubles import (
    FailingChunkResolver,
    FailingEmbeddingProvider,
    FailingVectorStore,
    LexicalEmbeddingProvider,
    MiscountingEmbeddingProvider,
    make_chunk,
)
from tests.support import run

PUBLIC_BACKEND = make_chunk(
    "backend--0000",
    "The service exposes its HTTP API with FastAPI and runs under Uvicorn.",
    document_id="backend",
    title="Backend Stack",
    heading_path=("HTTP layer",),
)
PUBLIC_STORAGE = make_chunk(
    "storage--0000",
    "Documents are stored as Markdown files in the repository.",
    document_id="storage",
    title="Storage",
    heading_path=("Corpus",),
)
INTERNAL_SECRET = make_chunk(
    "secrets--0000",
    "The confidential codename is INTERNAL-ONLY-SECRET-VALUE.",
    document_id="secrets",
    title="Internal Notes",
    heading_path=("Codename",),
    visibility="internal",
)


def metadata_for(chunk: KnowledgeChunk) -> VectorMetadata:
    return VectorMetadata(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        document_title=chunk.document_metadata.title,
        heading_path=chunk.heading_path,
        source_path=chunk.provenance.document.source_path,
        document_type=chunk.document_metadata.document_type,
        language=chunk.document_metadata.language,
        visibility=chunk.document_metadata.visibility,
        trust_level=chunk.document_metadata.trust_level,
    )


def build(
    chunks: list[KnowledgeChunk],
    *,
    policy: RetrievalPolicy | None = None,
    indexed: list[KnowledgeChunk] | None = None,
    resolvable: list[KnowledgeChunk] | None = None,
) -> tuple[PublicRetrievalService, LexicalEmbeddingProvider, InMemoryVectorStore]:
    """Index *chunks* and wire a retrieval service over them."""
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(embeddings.index_spec)
    fingerprint = "0" * 64

    for chunk in indexed if indexed is not None else chunks:
        run(
            store.upsert(
                [
                    VectorRecord(
                        id=chunk.id,
                        spec=embeddings.spec,
                        embedding=embeddings.vector_for(build_embedding_text(chunk)),
                        embedding_fingerprint=fingerprint,
                        chunk_fingerprint=chunk.provenance.fingerprint,
                        document_fingerprint=chunk.provenance.document.document_fingerprint,
                        metadata=metadata_for(chunk),
                    )
                ]
            )
        )

    resolver = InMemoryChunkResolver(resolvable if resolvable is not None else chunks)
    service = PublicRetrievalService(
        embeddings=embeddings,
        store=store,
        resolver=resolver,
        policy=policy or RetrievalPolicy(min_similarity=0.0),
    )
    return service, embeddings, store


def retrieve(service: PublicRetrievalService, question: str, **kwargs: object):
    return run(service.retrieve(normalize_query(question), **kwargs))  # type: ignore[arg-type]


# --- the happy path ----------------------------------------------------------


def test_a_question_finds_the_passage_that_answers_it():
    service, _, _ = build([PUBLIC_BACKEND, PUBLIC_STORAGE])

    outcome = retrieve(service, "Which HTTP framework does the service use?")

    assert outcome.is_sufficient
    assert outcome.chunks[0].chunk.id == "backend--0000"


def test_a_retrieved_passage_carries_its_text_provenance_and_heading():
    service, _, _ = build([PUBLIC_BACKEND])

    (retrieved_chunk,) = retrieve(service, "FastAPI Uvicorn HTTP").chunks

    assert "FastAPI" in retrieved_chunk.chunk.content
    assert retrieved_chunk.chunk.heading_path == ("HTTP layer",)
    assert retrieved_chunk.chunk.document_metadata.title == "Backend Stack"
    assert retrieved_chunk.chunk.provenance.document.source_path == "backend.md"


def test_results_are_ordered_by_similarity_descending():
    service, _, _ = build([PUBLIC_BACKEND, PUBLIC_STORAGE])

    outcome = retrieve(service, "FastAPI Uvicorn HTTP API")
    similarities = [item.similarity for item in outcome.chunks]

    assert similarities == sorted(similarities, reverse=True)


def test_ties_are_broken_by_chunk_id_so_two_runs_agree():
    twins = [
        make_chunk("twin--0000", "Identical text.", document_id="twin", ordinal=0),
        make_chunk("twin--0001", "Identical text.", document_id="twin", ordinal=1),
    ]
    service, _, _ = build(twins)

    outcome = retrieve(service, "Identical text.")

    assert [item.chunk.id for item in outcome.chunks] == ["twin--0000", "twin--0001"]


def test_the_question_is_embedded_exactly_once():
    service, embeddings, _ = build([PUBLIC_BACKEND])
    embeddings.embedded.clear()

    retrieve(service, "Which framework?")

    assert embeddings.embedded == ["Which framework?"]


def test_no_fake_chunk_is_manufactured_to_embed_a_question():
    """The question is embedded as itself, not dressed up as corpus text."""
    service, embeddings, _ = build([PUBLIC_BACKEND])
    embeddings.embedded.clear()

    retrieve(service, "Which framework?")

    (embedded,) = embeddings.embedded
    assert embedded == "Which framework?"
    assert "Document:" not in embedded
    assert PUBLIC_BACKEND.document_metadata.title not in embedded


# --- visibility --------------------------------------------------------------


def test_the_public_filter_is_part_of_the_query_that_is_sent(monkeypatch: pytest.MonkeyPatch):
    service, _, store = build([PUBLIC_BACKEND])
    sent: list[VectorQuery] = []
    original = store.query

    async def record(query: VectorQuery):
        sent.append(query)
        return await original(query)

    monkeypatch.setattr(store, "query", record)
    retrieve(service, "anything")

    assert sent[0].filters == {"visibility": "public"}


def test_the_filter_is_a_constant_and_not_an_argument():
    assert PUBLIC_VISIBILITY_FILTER == {"visibility": "public"}


def test_visibility_cannot_be_passed_to_retrieve():
    """Not an override that is ignored — an argument that does not exist."""
    service, _, _ = build([PUBLIC_BACKEND])

    with pytest.raises(TypeError):
        run(service.retrieve(normalize_query("x"), visibility="internal"))  # type: ignore[call-arg]


def test_visibility_cannot_be_smuggled_in_through_the_policy():
    with pytest.raises(Exception):  # noqa: B017 - any refusal is the assertion
        RetrievalPolicy(visibility="internal")  # type: ignore[call-arg]


def test_an_internal_passage_is_not_returned_even_when_it_matches_best():
    service, _, _ = build([PUBLIC_BACKEND, INTERNAL_SECRET])

    outcome = retrieve(service, "What is the confidential codename?")

    ids = [item.chunk.id for item in outcome.chunks]
    assert "secrets--0000" not in ids
    assert "INTERNAL-ONLY-SECRET-VALUE" not in str(outcome)


def test_an_index_that_still_calls_a_document_public_does_not_override_the_corpus():
    """The corpus is the source of truth, and it is the newer of the two."""
    public_at_index_time = make_chunk(
        "flipped--0000", "Text that used to be public.", document_id="flipped"
    )
    internal_now = public_at_index_time.model_copy(
        update={
            "document_metadata": public_at_index_time.document_metadata.model_copy(
                update={"visibility": "internal"}
            )
        }
    )
    service, _, _ = build(
        [public_at_index_time], indexed=[public_at_index_time], resolvable=[internal_now]
    )

    outcome = retrieve(service, "Text that used to be public.")

    assert outcome.chunks == ()
    assert outcome.withheld == 1


# --- policy ------------------------------------------------------------------


def test_top_k_of_one_returns_one_passage():
    service, _, _ = build(
        [PUBLIC_BACKEND, PUBLIC_STORAGE], policy=RetrievalPolicy(top_k=1, min_similarity=0.0)
    )

    assert len(retrieve(service, "storage FastAPI").chunks) == 1


def test_asking_for_more_than_the_index_holds_returns_what_there_is():
    service, _, _ = build(
        [PUBLIC_BACKEND, PUBLIC_STORAGE], policy=RetrievalPolicy(top_k=50, min_similarity=0.0)
    )

    assert len(retrieve(service, "FastAPI storage Markdown").chunks) == 2


def test_a_per_call_policy_overrides_the_configured_one():
    service, _, _ = build([PUBLIC_BACKEND, PUBLIC_STORAGE])

    outcome = retrieve(service, "FastAPI storage", policy=RetrievalPolicy(top_k=1))

    assert len(outcome.chunks) <= 1


def test_a_match_just_above_the_threshold_counts_as_evidence():
    service, _, _ = build([PUBLIC_BACKEND])
    exact = retrieve(service, "FastAPI").chunks[0].similarity

    outcome = retrieve(service, "FastAPI", policy=RetrievalPolicy(min_similarity=exact - 0.001))

    assert outcome.is_sufficient


def test_a_match_just_below_the_threshold_does_not():
    service, _, _ = build([PUBLIC_BACKEND])
    exact = retrieve(service, "FastAPI").chunks[0].similarity

    outcome = retrieve(service, "FastAPI", policy=RetrievalPolicy(min_similarity=exact + 0.001))

    assert not outcome.is_sufficient
    assert outcome.below_threshold == 1


def test_an_unrelated_question_scores_below_a_related_one():
    service, _, _ = build([PUBLIC_BACKEND, PUBLIC_STORAGE])

    related = retrieve(service, "Which HTTP framework does the service use?")
    unrelated = retrieve(service, "What is the maintainer's favourite pizza?")

    assert unrelated.chunks[0].similarity < related.chunks[0].similarity


def test_a_threshold_between_the_two_turns_the_unrelated_one_into_nothing():
    """What the threshold is for: weak matches are not evidence."""
    service, _, _ = build([PUBLIC_BACKEND, PUBLIC_STORAGE])
    unrelated = retrieve(service, "What is the maintainer's favourite pizza?")
    cutoff = unrelated.chunks[0].similarity + 0.01

    outcome = retrieve(
        service,
        "What is the maintainer's favourite pizza?",
        policy=RetrievalPolicy(min_similarity=cutoff),
    )

    assert not outcome.is_sufficient
    assert outcome.below_threshold == outcome.matches_returned


def test_an_empty_index_is_an_insufficient_result_and_not_a_failure():
    service, _, _ = build([])

    outcome = retrieve(service, "anything at all")

    assert not outcome.is_sufficient
    assert outcome.matches_returned == 0


# --- consistency and failure -------------------------------------------------


def test_a_record_the_corpus_no_longer_contains_is_dropped_and_counted():
    service, _, _ = build([PUBLIC_BACKEND], indexed=[PUBLIC_BACKEND], resolvable=[])

    outcome = retrieve(service, "FastAPI")

    assert outcome.chunks == ()
    assert outcome.unresolved == 1


def test_a_query_space_that_differs_from_the_index_is_refused_before_searching():
    embeddings = LexicalEmbeddingProvider()
    other_space = embeddings.spec.model_copy(update={"model": "some-other-model"})
    store = InMemoryVectorStore(VectorIndexSpec(embedding=other_space))
    service = PublicRetrievalService(
        embeddings=embeddings, store=store, resolver=InMemoryChunkResolver([])
    )

    with pytest.raises(EmbeddingSpaceMismatchError):
        retrieve(service, "anything")


def test_equal_dimensionality_is_not_compatibility():
    """Two models emitting the same number of floats do not share a space."""
    embeddings = DeterministicEmbeddingProvider()
    same_size_other_model = EmbeddingSpec(
        provider="deterministic",
        model="a-different-model",
        dimensions=embeddings.spec.dimensions,
        representation_version=embeddings.spec.representation_version,
    )
    service = PublicRetrievalService(
        embeddings=embeddings,
        store=InMemoryVectorStore(VectorIndexSpec(embedding=same_size_other_model)),
        resolver=InMemoryChunkResolver([]),
    )

    with pytest.raises(EmbeddingSpaceMismatchError):
        retrieve(service, "anything")


def test_an_unreachable_embedding_provider_is_a_controlled_failure():
    embeddings = LexicalEmbeddingProvider()
    service = PublicRetrievalService(
        embeddings=FailingEmbeddingProvider(embeddings.spec),
        store=InMemoryVectorStore(embeddings.index_spec),
        resolver=InMemoryChunkResolver([]),
    )

    with pytest.raises(QueryEmbeddingError):
        retrieve(service, "anything")


def test_an_embedding_batch_that_does_not_match_the_request_is_refused():
    embeddings = LexicalEmbeddingProvider()
    service = PublicRetrievalService(
        embeddings=MiscountingEmbeddingProvider(embeddings.spec),
        store=InMemoryVectorStore(embeddings.index_spec),
        resolver=InMemoryChunkResolver([]),
    )

    with pytest.raises(QueryEmbeddingError):
        retrieve(service, "anything")


def test_an_embedding_of_the_wrong_size_is_refused():
    embeddings = LexicalEmbeddingProvider()
    service = PublicRetrievalService(
        embeddings=MiscountingEmbeddingProvider(
            embeddings.spec, results=[EmbeddingResult(id="query", vector=(1.0, 0.0))]
        ),
        store=InMemoryVectorStore(embeddings.index_spec),
        resolver=InMemoryChunkResolver([]),
    )

    with pytest.raises(QueryEmbeddingError):
        retrieve(service, "anything")


def test_an_unreachable_vector_store_is_a_controlled_failure():
    embeddings = LexicalEmbeddingProvider()
    service = PublicRetrievalService(
        embeddings=embeddings,
        store=FailingVectorStore(embeddings.index_spec),
        resolver=InMemoryChunkResolver([]),
    )

    with pytest.raises(RetrievalUnavailableError):
        retrieve(service, "anything")


def test_an_unreachable_corpus_is_a_controlled_failure():
    service, _, _ = build([PUBLIC_BACKEND])
    broken = PublicRetrievalService(
        embeddings=LexicalEmbeddingProvider(),
        store=service._store,  # the index populated above
        resolver=FailingChunkResolver(),
        policy=RetrievalPolicy(min_similarity=0.0),
    )

    with pytest.raises(RetrievalUnavailableError):
        retrieve(broken, "FastAPI")


def test_a_failure_never_carries_the_provider_message_to_the_caller():
    embeddings = LexicalEmbeddingProvider()
    service = PublicRetrievalService(
        embeddings=embeddings,
        store=FailingVectorStore(embeddings.index_spec),
        resolver=InMemoryChunkResolver([]),
    )

    with pytest.raises(RetrievalUnavailableError) as caught:
        retrieve(service, "anything")

    assert "store unreachable" not in caught.value.message
