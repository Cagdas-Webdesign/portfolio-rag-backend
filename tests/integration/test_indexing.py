"""The indexing pipeline end to end, with real components.

A deterministic embedding provider and the in-memory store — both real
implementations of their ports, neither a mock. That matters: mocks would
assert that the service called what the test expected it to call, which is a
weaker claim than "the index ends up in the right state". Fakes with deliberate
failure behaviour appear further down, where the point *is* the failure.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from portfolio_rag.application.indexing import (
    EmbeddingValidationError,
    IncompatibleEmbeddingSpaceError,
    IndexingService,
    IndexSynchronizationError,
)
from portfolio_rag.domain.embedding import VectorIndexSpec, VectorMetadata, VectorRecord
from portfolio_rag.infrastructure.embedding import DeterministicEmbeddingProvider
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ingestion.chunking import chunk_document, chunk_knowledge_base
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingResult
from portfolio_rag.ports.errors import EmbeddingProviderError, VectorStoreError
from tests.conftest import DocumentFactory
from tests.support import run

DIMENSIONS = 32
BODY = "# Guide\n\n## First\n\nFirst body.\n\n## Second\n\nSecond body.\n"


class CountingProvider(DeterministicEmbeddingProvider):
    """The real provider, plus a tally of how much work it was asked to do."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        super().__init__(dimensions)
        self.calls = 0
        self.embedded = 0

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        self.calls += 1
        self.embedded += len(inputs)
        return await super().embed(inputs)


def build(provider: DeterministicEmbeddingProvider | None = None):
    provider = provider or CountingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=provider.spec))
    return provider, store, IndexingService(provider, store)


def chunks_of(document_factory: DocumentFactory, body: str = BODY, **kwargs: object):
    return chunk_document(document_factory(body, **kwargs))


# --- first run, second run --------------------------------------------------


def test_a_first_run_creates_a_record_for_every_chunk(make_document: DocumentFactory):
    provider, store, service = build()
    chunks = chunks_of(make_document)

    result = run(service.synchronize(chunks))

    assert result.created == len(chunks)
    assert result.embeddings_generated == len(chunks)
    assert result.deleted == 0
    assert [state.id for state in run(store.list_state())] == [chunk.id for chunk in chunks]
    assert provider.embedded == len(chunks)


def test_a_second_identical_run_is_a_genuine_no_op(make_document: DocumentFactory):
    """Zero embeddings, zero writes, zero deletes — the whole point of the
    fingerprints."""
    provider, _store, service = build()
    chunks = chunks_of(make_document)
    run(service.synchronize(chunks))
    provider.calls = 0
    provider.embedded = 0

    result = run(service.synchronize(chunks))

    assert result.is_noop
    assert result.unchanged == len(chunks)
    assert result.embeddings_generated == 0
    assert result.records_written == 0
    assert result.deleted == 0
    assert provider.calls == 0


def test_the_index_ends_up_the_same_however_many_times_it_is_run(
    make_document: DocumentFactory,
):
    _, store, service = build()
    chunks = chunks_of(make_document)

    run(service.synchronize(chunks))
    first = run(store.list_state())
    run(service.synchronize(chunks))
    run(service.synchronize(chunks))

    assert run(store.list_state()) == first


def test_records_carry_the_embedding_space_and_all_three_fingerprints(
    make_document: DocumentFactory,
):
    provider, store, service = build()
    chunks = chunks_of(make_document)
    run(service.synchronize(chunks))

    records = run(store.fetch([chunk.id for chunk in chunks]))

    for record, chunk in zip(records, chunks, strict=True):
        assert record.spec == provider.spec
        assert record.chunk_fingerprint == chunk.provenance.fingerprint
        assert record.document_fingerprint == chunk.provenance.document.document_fingerprint
        assert len(record.embedding_fingerprint) == 64
        assert len(record.embedding) == DIMENSIONS


# --- what changes, and what that costs --------------------------------------


def test_changing_one_chunk_re_embeds_only_that_chunk(make_document: DocumentFactory):
    provider, _store, service = build()
    run(service.synchronize(chunks_of(make_document)))
    provider.embedded = 0

    edited = BODY.replace("Second body.", "Second body, rewritten.")
    result = run(service.synchronize(chunks_of(make_document, edited)))

    assert result.reembedded == 1
    assert result.unchanged == 1
    assert result.embeddings_generated == 1
    assert provider.embedded == 1


def test_a_metadata_only_change_costs_no_embedding_at_all(
    make_document: DocumentFactory,
):
    """`visibility: internal` -> `public` rewrites the record around the same
    vector."""
    provider, store, service = build()
    run(service.synchronize(chunks_of(make_document, visibility="internal")))
    provider.embedded = 0
    provider.calls = 0

    result = run(service.synchronize(chunks_of(make_document, visibility="public")))

    assert result.metadata_updated == 2
    assert result.reembedded == 0
    assert result.embeddings_generated == 0
    assert provider.calls == 0
    assert all(state.metadata.visibility.value == "public" for state in run(store.list_state()))


def test_a_metadata_only_update_keeps_the_vector_it_already_had(
    make_document: DocumentFactory,
):
    _, store, service = build()
    run(service.synchronize(chunks_of(make_document, visibility="internal")))
    before = {record.id: record.embedding for record in run(store.fetch(["sample--0000"]))}

    run(service.synchronize(chunks_of(make_document, visibility="public")))
    after = {record.id: record.embedding for record in run(store.fetch(["sample--0000"]))}

    assert before == after


def test_a_changed_title_re_embeds_because_it_is_part_of_the_representation(
    make_document: DocumentFactory,
):
    provider, _, service = build()
    run(service.synchronize(chunks_of(make_document, title="Original")))
    provider.embedded = 0

    result = run(service.synchronize(chunks_of(make_document, title="Renamed")))

    assert result.reembedded == 2
    assert provider.embedded == 2


def test_a_removed_chunk_leaves_no_stale_vector_behind(make_document: DocumentFactory):
    _, store, service = build()
    run(service.synchronize(chunks_of(make_document)))

    shortened = "# Guide\n\n## First\n\nFirst body.\n"
    result = run(service.synchronize(chunks_of(make_document, shortened)))

    assert result.deleted == 1
    assert [state.id for state in run(store.list_state())] == ["sample--0000"]


def test_a_new_chunk_costs_exactly_one_embedding(make_document: DocumentFactory):
    provider, _, service = build()
    run(service.synchronize(chunks_of(make_document)))
    provider.embedded = 0

    extended = f"{BODY}\n## Third\n\nThird body.\n"
    result = run(service.synchronize(chunks_of(make_document, extended)))

    assert result.created == 1
    assert result.unchanged == 2
    assert provider.embedded == 1


def test_two_chunks_with_identical_text_share_one_embedding_call(
    make_document: DocumentFactory,
):
    provider, _, service = build()
    body = "## A\n\nExactly the same words.\n\n## A\n\nExactly the same words.\n"

    result = run(service.synchronize(chunks_of(make_document, body)))

    assert result.created == 2
    assert result.embeddings_generated == 1
    assert provider.embedded == 1


# --- corpora ----------------------------------------------------------------


def test_a_corpus_of_several_documents_indexes_completely(
    make_document: DocumentFactory,
):
    _, store, service = build()
    documents = [
        make_document(BODY, document_id="first"),
        make_document("Single paragraph.\n", document_id="second"),
    ]
    chunks = chunk_knowledge_base(documents)

    result = run(service.synchronize(chunks))

    assert result.created == 3
    assert len(run(store.list_state())) == 3


def test_deleting_a_whole_document_removes_all_of_its_records(
    make_document: DocumentFactory,
):
    _, store, service = build()
    documents = [
        make_document(BODY, document_id="first"),
        make_document("Single paragraph.\n", document_id="second"),
    ]
    run(service.synchronize(chunk_knowledge_base(documents)))

    result = run(service.synchronize(chunk_knowledge_base(documents[:1])))

    assert result.deleted == 1
    assert all(state.document_id != "second" for state in _metadata(store))


def _metadata(store: InMemoryVectorStore):
    return [state.metadata for state in run(store.list_state())]


# --- embedding spaces -------------------------------------------------------


def test_an_index_for_another_embedding_space_is_refused_before_any_write(
    make_document: DocumentFactory,
):
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)
    other = provider.spec.model_copy(update={"model": "some-other-model"})
    store = InMemoryVectorStore(VectorIndexSpec(embedding=other))
    service = IndexingService(provider, store)

    with pytest.raises(IncompatibleEmbeddingSpaceError, match=r"embedding space|index holds"):
        run(service.synchronize(chunks_of(make_document)))

    assert run(store.list_state()) == []


def test_a_dimension_mismatch_fails_before_the_third_upsert_not_after(
    make_document: DocumentFactory,
):
    provider = DeterministicEmbeddingProvider(dimensions=16)
    store = InMemoryVectorStore(
        VectorIndexSpec(embedding=provider.spec.model_copy(update={"dimensions": 8}))
    )
    service = IndexingService(provider, store)

    with pytest.raises(IncompatibleEmbeddingSpaceError):
        run(service.plan(chunks_of(make_document)))

    assert run(store.list_state()) == []


def test_an_index_holding_foreign_vectors_is_not_silently_extended(
    make_document: DocumentFactory,
):
    """Same index spec, but a record from a different model already inside."""
    provider, store, service = build()
    foreign_spec = provider.spec.model_copy(update={"model": "legacy-model"})
    store._records["legacy--0000"] = VectorRecord(
        id="legacy--0000",
        spec=foreign_spec,
        embedding=tuple(0.5 for _ in range(DIMENSIONS)),
        embedding_fingerprint="a" * 64,
        chunk_fingerprint="b" * 64,
        document_fingerprint="c" * 64,
        metadata=_first_metadata(service, make_document),
    )

    with pytest.raises(IncompatibleEmbeddingSpaceError, match=r"legacy-model|cannot"):
        run(service.plan(chunks_of(make_document)))


def _first_metadata(service: IndexingService, make_document: DocumentFactory) -> VectorMetadata:
    from portfolio_rag.application.indexing import build_desired_state

    (desired, *_) = build_desired_state(chunks_of(make_document), service.spec)
    return desired.metadata


def test_rebuild_re_embeds_everything_and_clears_what_is_no_longer_wanted(
    make_document: DocumentFactory,
):
    provider, _store, service = build()
    run(service.synchronize(chunks_of(make_document)))
    provider.embedded = 0

    result = run(service.synchronize(chunks_of(make_document), rebuild=True))

    assert result.created == 2
    assert result.unchanged == 0
    assert provider.embedded == 2


# --- failures ---------------------------------------------------------------


class BrokenProvider(DeterministicEmbeddingProvider):
    """A provider that misbehaves in one specific, chosen way."""

    def __init__(self, mode: str) -> None:
        super().__init__(DIMENSIONS)
        self.mode = mode

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        results = await super().embed(inputs)
        match self.mode:
            case "short":
                return results[:-1]
            case "wrong-dimensions":
                return [
                    EmbeddingResult(id=result.id, vector=result.vector[:-1]) for result in results
                ]
            case "unknown-id":
                return [
                    EmbeddingResult(id=f"not-an-input-{index}", vector=result.vector)
                    for index, result in enumerate(results)
                ]
            case "raises":
                raise EmbeddingProviderError("upstream is down", retryable=True)
        return results  # pragma: no cover


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("short", "1 embeddings for 2 inputs"),
        ("wrong-dimensions", "dimensions"),
        ("unknown-id", "unknown input"),
    ],
)
def test_a_provider_that_breaks_its_contract_stops_the_run(
    make_document: DocumentFactory, mode: str, match: str
):
    provider = BrokenProvider(mode)
    store = InMemoryVectorStore(VectorIndexSpec(embedding=provider.spec))
    service = IndexingService(provider, store)

    with pytest.raises(EmbeddingValidationError, match=match):
        run(service.synchronize(chunks_of(make_document)))

    assert run(store.list_state()) == []


def test_a_provider_failure_is_never_reported_as_success(make_document: DocumentFactory):
    provider = BrokenProvider("raises")
    store = InMemoryVectorStore(VectorIndexSpec(embedding=provider.spec))
    service = IndexingService(provider, store)

    with pytest.raises(EmbeddingProviderError):
        run(service.synchronize(chunks_of(make_document)))


def test_stale_records_are_not_deleted_when_embedding_fails(
    make_document: DocumentFactory,
):
    """Upserts before deletes: a failed run must not strip a working index."""
    _provider, store, service = build()
    run(service.synchronize(chunks_of(make_document)))
    before = run(store.list_state())

    broken = BrokenProvider("raises")
    failing = IndexingService(broken, store)
    shortened = "# Guide\n\n## First\n\nRewritten first body.\n"

    with pytest.raises(EmbeddingProviderError):
        run(failing.synchronize(chunks_of(make_document, shortened)))

    assert run(store.list_state()) == before


class FailingStore(InMemoryVectorStore):
    """Accepts the first upsert batch, then refuses."""

    def __init__(self, index_spec: VectorIndexSpec) -> None:
        super().__init__(index_spec)
        self.upserts = 0

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        self.upserts += 1
        if self.upserts > 1:
            raise VectorStoreError("store is unavailable")
        await super().upsert(records)


def test_a_store_failure_is_reported_rather_than_swallowed(
    make_document: DocumentFactory,
):
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)
    store = FailingStore(VectorIndexSpec(embedding=provider.spec))
    service = IndexingService(provider, store)
    run(service.synchronize(chunks_of(make_document)))

    edited = BODY.replace("Second body.", "Second body, rewritten.")

    with pytest.raises(IndexSynchronizationError, match="synchronization failed"):
        run(service.synchronize(chunks_of(make_document, edited)))


def test_the_next_run_converges_after_a_partial_failure(make_document: DocumentFactory):
    """The recovery story: no compensation, no journal — just plan again."""
    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)
    store = FailingStore(VectorIndexSpec(embedding=provider.spec))
    service = IndexingService(provider, store)
    run(service.synchronize(chunks_of(make_document)))

    edited = BODY.replace("Second body.", "Second body, rewritten.")
    with pytest.raises(IndexSynchronizationError):
        run(service.synchronize(chunks_of(make_document, edited)))

    store.upserts = 0  # the store recovers
    result = run(service.synchronize(chunks_of(make_document, edited)))

    assert result.reembedded == 1
    assert run(service.synchronize(chunks_of(make_document, edited))).is_noop


# --- planning does nothing --------------------------------------------------


def test_planning_makes_no_provider_call_and_writes_nothing(
    make_document: DocumentFactory,
):
    provider, store, service = build()
    chunks = chunks_of(make_document)

    plan = run(service.plan(chunks))

    assert len(plan.create) == len(chunks)
    assert len(plan.embeddings_required) == len(chunks)
    assert provider.calls == 0
    assert run(store.list_state()) == []


def test_planning_an_unchanged_corpus_requires_no_embeddings(
    make_document: DocumentFactory,
):
    provider, _, service = build()
    chunks = chunks_of(make_document)
    run(service.synchronize(chunks))
    provider.calls = 0

    plan = run(service.plan(chunks))

    assert plan.is_noop
    assert plan.embeddings_required == ()
    assert provider.calls == 0


def test_a_store_that_cannot_enumerate_still_plans_creates_and_updates(
    make_document: DocumentFactory,
):
    """Vectorize's real limitation, modelled honestly rather than assumed away."""

    class NonEnumerableStore(InMemoryVectorStore):
        @property
        def supports_enumeration(self) -> bool:
            return False

    provider = DeterministicEmbeddingProvider(dimensions=DIMENSIONS)
    store = NonEnumerableStore(VectorIndexSpec(embedding=provider.spec))
    service = IndexingService(provider, store)
    chunks = chunks_of(make_document)

    first = run(service.synchronize(chunks))
    second = run(service.plan(chunks))

    assert first.created == 2
    assert first.stale_detection is False
    assert second.is_noop
    assert second.stale_detection is False


def test_the_embedding_space_is_reported_by_the_service():
    provider, _, service = build()

    assert service.spec == provider.spec
    assert service.spec.dimensions == DIMENSIONS


def test_only_the_representation_ever_leaves_the_process(make_document: DocumentFactory):
    """The privacy boundary, asserted end to end rather than per adapter.

    Whatever an external provider is handed, it is the composed representation
    and nothing else — no ids, no visibility, no trust level, no licence, no
    source path, no fingerprints, no dates.
    """
    sent: list[str] = []

    class RecordingProvider(DeterministicEmbeddingProvider):
        async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
            sent.extend(item.text for item in inputs)
            return await super().embed(inputs)

    provider = RecordingProvider(DIMENSIONS)
    store = InMemoryVectorStore(VectorIndexSpec(embedding=provider.spec))
    document = make_document(
        "# Backend\n\n## Cloudflare\n\nThe actual statement.\n",
        document_id="secret-doc",
        title="Public Title",
        source="internal/vault",
        source_path="internal/secret.md",
        document_fingerprint="f" * 64,
        visibility="internal",
        trust_level="authoritative",
        license="Proprietary",
        topics=["confidential"],
        technologies=["Vault"],
    )

    run(IndexingService(provider, store).synchronize(chunk_document(document)))

    assert sent == ["Public Title\n\nBackend > Cloudflare\n\nThe actual statement."]
    everything = " ".join(sent)
    for private in (
        "secret-doc",
        "internal",
        "authoritative",
        "Proprietary",
        "confidential",
        "Vault",
        "f" * 64,
        "secret.md",
        "2026-08-07",
    ):
        assert private not in everything, f"{private!r} must not reach an embedding provider"


def test_chunk_content_is_never_stored_in_the_index(make_document: DocumentFactory):
    """The index holds vectors and metadata; the corpus stays the source of truth."""
    _, store, service = build()
    body = "## S\n\nA very distinctive sentence that should not be stored.\n"
    run(service.synchronize(chunks_of(make_document, body)))

    serialized = str([state.model_dump() for state in run(store.list_state())])

    assert "distinctive sentence" not in serialized
