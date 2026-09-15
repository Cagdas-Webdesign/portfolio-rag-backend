"""Chunk identity: readable ids, content fingerprints, and corpus invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from portfolio_rag.ingestion.chunking import (
    DEFAULT_CHUNKING_POLICY,
    MARKDOWN_CHUNKING_STRATEGY_VERSION,
    ChunkingInvariantError,
    ChunkingPolicy,
    UnsplittableBlockError,
    chunk_document,
    chunk_knowledge_base,
)
from portfolio_rag.ingestion.chunking.fingerprint import build_chunk_id, compute_fingerprint
from tests.conftest import DocumentFactory

MULTI_SECTION = """# One

Body of section one.

## Two

Body of section two.

## Three

Body of section three.
"""


def _fingerprint(
    *,
    strategy_version: str = MARKDOWN_CHUNKING_STRATEGY_VERSION,
    policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
    document_id: str = "sample",
    heading_path: tuple[str, ...] = ("A", "B"),
    content: str = "Some chunk content.",
) -> str:
    """One fingerprint input, with a single field varied per test."""
    return compute_fingerprint(
        strategy_version=strategy_version,
        policy=policy,
        document_id=document_id,
        heading_path=heading_path,
        content=content,
    )


# --- ids --------------------------------------------------------------------


def test_ids_are_the_document_id_plus_a_padded_ordinal():
    assert build_chunk_id("api-integrations", 0) == "api-integrations--0000"
    assert build_chunk_id("api-integrations", 42) == "api-integrations--0042"


def test_ids_follow_the_ordinals_of_a_real_document(make_document: DocumentFactory):
    chunks = chunk_document(make_document(MULTI_SECTION, document_id="apis"))

    assert [chunk.id for chunk in chunks] == ["apis--0000", "apis--0001", "apis--0002"]
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]


def test_ordinals_are_gapless_and_start_at_zero(make_document: DocumentFactory):
    chunks = chunk_document(make_document(MULTI_SECTION))

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))


def test_ids_are_stable_across_runs(make_document: DocumentFactory):
    first = chunk_document(make_document(MULTI_SECTION))
    second = chunk_document(make_document(MULTI_SECTION))

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]


# --- fingerprints -----------------------------------------------------------


def test_a_fingerprint_is_a_sha256_hex_digest(make_document: DocumentFactory):
    for chunk in chunk_document(make_document(MULTI_SECTION)):
        assert len(chunk.provenance.fingerprint) == 64
        assert set(chunk.provenance.fingerprint) <= set("0123456789abcdef")


def test_the_same_chunk_fingerprints_the_same_every_time():
    assert _fingerprint() == _fingerprint()


def test_changed_content_changes_the_fingerprint():
    assert _fingerprint(content="Different chunk content.") != _fingerprint()


def test_a_changed_heading_path_changes_the_fingerprint():
    """The same words under a different heading are a different retrieval unit."""
    assert _fingerprint(heading_path=("A", "C")) != _fingerprint()


def test_a_changed_strategy_version_changes_the_fingerprint():
    assert _fingerprint(strategy_version="markdown-structure-v2") != _fingerprint()


@pytest.mark.parametrize(
    "policy",
    [
        ChunkingPolicy(target_chars=900),
        ChunkingPolicy(max_chars=1500),
        ChunkingPolicy(overlap_chars=0),
    ],
)
def test_a_changed_policy_changes_the_fingerprint(policy: ChunkingPolicy):
    """A chunk is a unit produced by a recipe; changing the recipe changes it."""
    assert _fingerprint(policy=policy) != _fingerprint()


def test_a_changed_document_id_changes_the_fingerprint():
    assert _fingerprint(document_id="other-document") != _fingerprint()


def test_moving_a_document_does_not_change_its_chunk_fingerprints(
    make_document: DocumentFactory,
):
    """A path is provenance, not part of what a document says."""
    here = chunk_document(make_document(MULTI_SECTION, source_path="here/doc.md"))
    there = chunk_document(make_document(MULTI_SECTION, source_path="elsewhere/doc.md"))

    assert [c.provenance.fingerprint for c in here] == [c.provenance.fingerprint for c in there]
    assert here[0].provenance.document.source_path != there[0].provenance.document.source_path


def test_an_edit_elsewhere_in_a_document_leaves_unchanged_chunks_alone(
    make_document: DocumentFactory,
):
    """The document hash is carried for lineage but deliberately not fingerprinted:
    fixing a typo in section one must not invalidate section three's embedding."""
    before = chunk_document(make_document(MULTI_SECTION, document_fingerprint="a" * 64))
    edited = MULTI_SECTION.replace("Body of section one.", "Body of section one, corrected.")
    after = chunk_document(make_document(edited, document_fingerprint="b" * 64))

    assert before[0].provenance.fingerprint != after[0].provenance.fingerprint
    assert before[2].provenance.fingerprint == after[2].provenance.fingerprint
    assert (
        before[2].provenance.document.document_fingerprint
        != after[2].provenance.document.document_fingerprint
    )


def test_identical_text_under_the_same_heading_fingerprints_identically(
    make_document: DocumentFactory,
):
    """Documented consequence of leaving the ordinal out: same unit, same hash."""
    content = "## Repeat\n\nExactly the same words.\n\n## Repeat\n\nExactly the same words.\n"
    chunks = chunk_document(make_document(content))

    assert chunks[0].provenance.fingerprint == chunks[1].provenance.fingerprint
    assert chunks[0].id != chunks[1].id


# --- what every chunk carries ----------------------------------------------


def test_every_chunk_carries_its_document_metadata_and_provenance(
    make_document: DocumentFactory,
):
    document = make_document(
        MULTI_SECTION,
        document_id="apis",
        title="API Reference",
        visibility="public",
        trust_level="verified",
        topics=["backend"],
        technologies=["REST"],
        license="CC0-1.0",
        source_path="skills/apis.md",
        document_fingerprint="c" * 64,
    )

    for chunk in chunk_document(document):
        assert chunk.document_id == "apis"
        assert chunk.document_metadata.title == "API Reference"
        assert chunk.document_metadata.visibility.value == "public"
        assert chunk.document_metadata.trust_level.value == "verified"
        assert chunk.document_metadata.topics == ("backend",)
        assert chunk.document_metadata.technologies == ("REST",)
        assert chunk.document_metadata.license == "CC0-1.0"
        assert chunk.provenance.document.source_path == "skills/apis.md"
        assert chunk.provenance.document.document_fingerprint == "c" * 64
        assert chunk.provenance.strategy_version == MARKDOWN_CHUNKING_STRATEGY_VERSION


def test_the_innermost_heading_is_available_for_citations(make_document: DocumentFactory):
    chunks = chunk_document(make_document(MULTI_SECTION))

    assert chunks[1].heading_path == ("One", "Two")
    assert chunks[1].section == "Two"


def test_a_chunk_above_the_first_heading_has_no_section(make_document: DocumentFactory):
    (chunk,) = chunk_document(make_document("Body without any heading.\n"))

    assert chunk.heading_path == ()
    assert chunk.section is None


def test_chunks_are_immutable(make_document: DocumentFactory):
    (chunk,) = chunk_document(make_document("Body.\n"))

    with pytest.raises(ValidationError):
        chunk.content = "rewritten"  # type: ignore[misc]


def test_the_content_carries_no_synthetic_context(make_document: DocumentFactory):
    """Titles and headings live beside the content, never inside it."""
    document = make_document("# Heading\n\nJust the body.\n", title="Some Title")
    (chunk,) = chunk_document(document)

    assert chunk.content == "Just the body."
    assert "Some Title" not in chunk.content
    assert "Heading" not in chunk.content


# --- corpus level -----------------------------------------------------------


def test_a_corpus_keeps_the_order_its_documents_arrive_in(make_document: DocumentFactory):
    documents = [
        make_document("Only one chunk.\n", document_id="zeta"),
        make_document(MULTI_SECTION, document_id="alpha"),
        make_document("Also one chunk.\n", document_id="mid"),
    ]

    chunks = chunk_knowledge_base(documents)

    assert [chunk.document_id for chunk in chunks] == ["zeta"] + ["alpha"] * 3 + ["mid"]


def test_chunk_ids_are_unique_across_the_whole_corpus(make_document: DocumentFactory):
    documents = [make_document(MULTI_SECTION, document_id=f"doc-{index}") for index in range(5)]

    chunks = chunk_knowledge_base(documents)
    ids = [chunk.id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_ordinals_restart_at_zero_for_each_document(make_document: DocumentFactory):
    documents = [
        make_document(MULTI_SECTION, document_id="first"),
        make_document(MULTI_SECTION, document_id="second"),
    ]

    chunks = chunk_knowledge_base(documents)

    for document_id in ("first", "second"):
        ordinals = [c.ordinal for c in chunks if c.document_id == document_id]
        assert ordinals == list(range(len(ordinals)))


def test_a_corpus_is_reproducible(make_document: DocumentFactory):
    documents = [make_document(MULTI_SECTION, document_id=f"doc-{i}") for i in range(3)]

    first = chunk_knowledge_base(documents)
    second = chunk_knowledge_base(documents)

    assert first == second


def test_one_unchunkable_document_fails_the_whole_corpus(make_document: DocumentFactory):
    """A partial corpus that looks complete is worse than a loud failure."""
    policy = ChunkingPolicy(target_chars=100, max_chars=120, overlap_chars=0)
    oversized = "```\n" + "\n".join(f"line {index}" for index in range(40)) + "\n```"
    documents = [
        make_document("Fine.\n", document_id="good-one"),
        make_document(oversized, document_id="bad-one"),
        make_document("Also fine.\n", document_id="good-two"),
    ]

    with pytest.raises(UnsplittableBlockError) as caught:
        chunk_knowledge_base(documents, policy)

    assert caught.value.document_id == "bad-one"


def test_duplicate_document_ids_are_caught_as_an_invariant_violation(
    make_document: DocumentFactory,
):
    """The loader prevents this upstream; the check is a safety net, not a duplicate."""
    documents = [
        make_document("One.\n", document_id="same"),
        make_document("Two.\n", document_id="same"),
    ]

    with pytest.raises(ChunkingInvariantError, match="not unique"):
        chunk_knowledge_base(documents)


def test_an_empty_corpus_produces_no_chunks():
    assert chunk_knowledge_base([]) == ()
