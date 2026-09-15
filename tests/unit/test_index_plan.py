"""Index planning: deciding what to do, before doing any of it."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from portfolio_rag.application.indexing import DesiredRecord, build_desired_state, plan_index
from portfolio_rag.domain.embedding import VectorRecordState
from portfolio_rag.ingestion.chunking import chunk_document
from tests.conftest import DocumentFactory
from tests.factories import DEFAULT_SPEC, make_metadata, make_vector_record


def _desired(record_id: str, *, embedding: str = "e", metadata_marker: str = "en") -> DesiredRecord:
    return DesiredRecord(
        id=record_id,
        embedding_text=f"text for {record_id}",
        embedding_fingerprint=_digest(embedding),
        chunk_fingerprint=_digest("chunk"),
        document_fingerprint=_digest("document"),
        metadata=make_metadata(record_id, language=metadata_marker),
    )


def _actual(
    record_id: str, *, embedding: str = "e", metadata_marker: str = "en", **overrides: Any
) -> VectorRecordState:
    return make_vector_record(
        record_id,
        embedding_fingerprint=_digest(embedding),
        chunk_fingerprint=_digest("chunk"),
        document_fingerprint=_digest("document"),
        metadata=make_metadata(record_id, language=metadata_marker),
        **overrides,
    ).state()


def _digest(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode()).hexdigest()


# --- the four outcomes ------------------------------------------------------


def test_nothing_desired_and_nothing_indexed_is_nothing_to_do():
    plan = plan_index([], [])

    assert plan.is_noop
    assert plan.describe() == (
        ("create", "0"),
        ("re-embed", "0"),
        ("metadata only", "0"),
        ("unchanged", "0"),
        ("delete", "0"),
    )


def test_a_record_the_index_does_not_have_is_created():
    plan = plan_index([_desired("doc--0000")], [])

    assert [record.id for record in plan.create] == ["doc--0000"]
    assert len(plan.embeddings_required) == 1
    assert not plan.is_noop


def test_a_matching_record_is_unchanged():
    plan = plan_index([_desired("doc--0000")], [_actual("doc--0000")])

    assert [record.id for record in plan.unchanged] == ["doc--0000"]
    assert plan.embeddings_required == ()
    assert plan.is_noop


def test_a_different_embedding_fingerprint_means_re_embedding():
    plan = plan_index([_desired("doc--0000", embedding="new")], [_actual("doc--0000")])

    assert [record.id for record in plan.reembed] == ["doc--0000"]
    assert len(plan.embeddings_required) == 1


def test_an_indexed_record_nobody_wants_is_deleted():
    plan = plan_index([_desired("doc--0000")], [_actual("doc--0000"), _actual("doc--0001")])

    assert plan.delete == ("doc--0001",)
    assert plan.embeddings_required == ()


def test_all_four_outcomes_can_happen_in_one_plan():
    desired = [
        _desired("doc--0000"),  # unchanged
        _desired("doc--0001", embedding="new"),  # re-embed
        _desired("doc--0002", metadata_marker="de"),  # metadata only
        _desired("doc--0003"),  # create
    ]
    actual = [
        _actual("doc--0000"),
        _actual("doc--0001"),
        _actual("doc--0002"),
        _actual("doc--0009"),  # delete
    ]

    plan = plan_index(desired, actual)

    assert [record.id for record in plan.unchanged] == ["doc--0000"]
    assert [record.id for record in plan.reembed] == ["doc--0001"]
    assert [record.id for record in plan.metadata_updates] == ["doc--0002"]
    assert [record.id for record in plan.create] == ["doc--0003"]
    assert plan.delete == ("doc--0009",)
    assert plan.desired_count == 4


# --- metadata-only updates --------------------------------------------------


def test_changed_metadata_alone_does_not_require_a_new_embedding():
    """Flipping `visibility` costs a metadata write, not a provider call."""
    desired = DesiredRecord(
        id="doc--0000",
        embedding_text="text",
        embedding_fingerprint=_digest("e"),
        chunk_fingerprint=_digest("chunk"),
        document_fingerprint=_digest("document"),
        metadata=make_metadata("doc--0000", visibility="public"),
    )
    actual = make_vector_record(
        "doc--0000",
        embedding_fingerprint=_digest("e"),
        chunk_fingerprint=_digest("chunk"),
        document_fingerprint=_digest("document"),
        metadata=make_metadata("doc--0000", visibility="internal"),
    ).state()

    plan = plan_index([desired], [actual])

    assert [record.id for record in plan.metadata_updates] == ["doc--0000"]
    assert plan.embeddings_required == ()
    assert not plan.is_noop


def test_a_changed_document_fingerprint_alone_is_a_metadata_update():
    """Lineage moved; the text this unit contains did not. No re-embedding."""
    desired = _desired("doc--0000")
    desired = replace(desired, document_fingerprint=_digest("edited"))

    plan = plan_index([desired], [_actual("doc--0000")])

    assert [record.id for record in plan.metadata_updates] == ["doc--0000"]
    assert plan.embeddings_required == ()


def test_a_changed_chunk_fingerprint_alone_is_also_a_metadata_update():
    desired = _desired("doc--0000")
    desired = replace(desired, chunk_fingerprint=_digest("recut"))

    plan = plan_index([desired], [_actual("doc--0000")])

    assert [record.id for record in plan.metadata_updates] == ["doc--0000"]


# --- determinism and efficiency --------------------------------------------


def test_planning_the_same_inputs_twice_gives_the_same_plan():
    desired = [_desired(f"doc--{index:04d}") for index in range(4)]
    actual = [_actual("doc--0000"), _actual("doc--0009")]

    assert plan_index(desired, actual) == plan_index(desired, actual)


def test_deletions_are_sorted_so_the_plan_reads_the_same_every_time():
    actual = [_actual("doc--0009"), _actual("doc--0001"), _actual("doc--0005")]

    plan = plan_index([], actual)

    assert plan.delete == ("doc--0001", "doc--0005", "doc--0009")


def test_identical_text_is_only_embedded_once():
    """Two chunks with the same representation share one vector."""
    desired = [_desired("doc--0000", embedding="same"), _desired("doc--0001", embedding="same")]

    plan = plan_index(desired, [])

    assert len(plan.create) == 2
    assert len(plan.embeddings_required) == 1


def test_rebuild_treats_everything_as_new():
    desired = [_desired("doc--0000", embedding="a"), _desired("doc--0001", embedding="b")]
    actual = [_actual("doc--0000", embedding="a"), _actual("doc--0001", embedding="b")]

    plan = plan_index(desired, actual, rebuild=True)

    assert len(plan.create) == 2
    assert plan.unchanged == ()
    assert len(plan.embeddings_required) == 2


def test_a_plan_records_whether_stale_records_could_be_looked_for():
    assert plan_index([], [], stale_detection=False).stale_detection is False
    assert plan_index([], []).stale_detection is True


# --- desired state from real chunks ----------------------------------------


def test_desired_state_carries_the_metadata_retrieval_will_need(
    make_document: DocumentFactory,
):
    document = make_document(
        "# Backend\n\n## APIs\n\nBody text.\n",
        document_id="apis",
        title="API Reference",
        visibility="public",
        trust_level="verified",
        topics=["backend"],
        technologies=["REST"],
        source_path="skills/apis.md",
    )
    chunks = chunk_document(document)

    (record,) = build_desired_state(chunks, DEFAULT_SPEC)

    assert record.id == "apis--0000"
    assert record.metadata.document_title == "API Reference"
    assert record.metadata.heading_path == ("Backend", "APIs")
    assert record.metadata.source_path == "skills/apis.md"
    assert record.metadata.visibility.value == "public"
    assert record.metadata.trust_level.value == "verified"
    assert record.metadata.topics == ("backend",)
    assert record.chunk_fingerprint == chunks[0].provenance.fingerprint
    assert record.document_fingerprint == document.provenance.document_fingerprint


def test_desired_state_keeps_the_corpus_order(make_document: DocumentFactory):
    chunks = chunk_document(make_document("# A\n\nOne.\n\n## B\n\nTwo.\n\n## C\n\nThree.\n"))

    desired = build_desired_state(chunks, DEFAULT_SPEC)

    assert [record.id for record in desired] == [chunk.id for chunk in chunks]
