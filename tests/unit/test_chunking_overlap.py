"""Controlled overlap: repeated context inside a split section, and nowhere else."""

from __future__ import annotations

from itertools import pairwise

from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.ingestion.chunking import ChunkingPolicy, chunk_document
from tests.conftest import DocumentFactory

WITH_OVERLAP = ChunkingPolicy(target_chars=120, max_chars=260, overlap_chars=60)
WITHOUT_OVERLAP = ChunkingPolicy(target_chars=120, max_chars=260, overlap_chars=0)

LONG_SECTION = "\n\n".join(
    f"Paragraph {index} of a section that has to be split." for index in range(8)
)


def _split_section_document(document_factory: DocumentFactory) -> list[KnowledgeChunk]:
    return list(chunk_document(document_factory(f"## S\n\n{LONG_SECTION}\n"), WITH_OVERLAP))


def test_a_split_section_repeats_context_after_its_first_chunk(
    make_document: DocumentFactory,
):
    chunks = _split_section_document(make_document)

    assert len(chunks) > 1
    assert chunks[0].provenance.overlap_prefix_chars == 0
    assert any(chunk.provenance.overlap_prefix_chars > 0 for chunk in chunks[1:])


def test_the_repeated_text_really_comes_from_the_previous_chunk(
    make_document: DocumentFactory,
):
    """`content[:n]` is exactly the repeated text — no separator included."""
    chunks = _split_section_document(make_document)

    for previous, current in pairwise(chunks):
        size = current.provenance.overlap_prefix_chars
        if size:
            repeated = current.content[:size]
            assert repeated in previous.content
            assert not repeated.startswith("\n")
            assert not repeated.endswith("\n")


def test_the_separator_after_the_repeated_text_is_not_counted_as_repetition(
    make_document: DocumentFactory,
):
    """The blank line joining the prefix to the body is structure, not content."""
    chunks = _split_section_document(make_document)
    with_overlap = [c for c in chunks if c.provenance.overlap_prefix_chars]

    assert with_overlap
    for chunk in with_overlap:
        size = chunk.provenance.overlap_prefix_chars
        assert chunk.content[size : size + 2] == "\n\n"


def test_zero_overlap_means_no_repetition_at_all(make_document: DocumentFactory):
    chunks = chunk_document(make_document(f"## S\n\n{LONG_SECTION}\n"), WITHOUT_OVERLAP)

    assert len(chunks) > 1
    assert all(chunk.provenance.overlap_prefix_chars == 0 for chunk in chunks)


def test_overlap_never_crosses_a_section_boundary(make_document: DocumentFactory):
    """A new heading is a new subject; carrying the old one in would be noise."""
    content = f"## First\n\n{LONG_SECTION}\n\n## Second\n\n{LONG_SECTION}\n"
    chunks = chunk_document(make_document(content), WITH_OVERLAP)

    seen_paths: set[tuple[str, ...]] = set()
    for chunk in chunks:
        if chunk.heading_path not in seen_paths:
            assert chunk.provenance.overlap_prefix_chars == 0, (
                f"first chunk of {chunk.heading_path} must not repeat another section"
            )
            seen_paths.add(chunk.heading_path)

    assert seen_paths == {("First",), ("Second",)}


def test_a_short_standalone_section_gets_no_artificial_overlap(
    make_document: DocumentFactory,
):
    content = "## A\n\nShort body.\n\n## B\n\nAnother short body.\n"
    chunks = chunk_document(make_document(content), WITH_OVERLAP)

    assert [chunk.provenance.overlap_prefix_chars for chunk in chunks] == [0, 0]


def test_overlap_never_pushes_a_chunk_over_the_maximum(make_document: DocumentFactory):
    chunks = chunk_document(make_document(f"## S\n\n{LONG_SECTION}\n"), WITH_OVERLAP)

    for chunk in chunks:
        assert len(chunk.content) <= WITH_OVERLAP.max_chars


def test_the_repeated_text_stays_within_its_budget(make_document: DocumentFactory):
    chunks = _split_section_document(make_document)

    for chunk in chunks:
        assert chunk.provenance.overlap_prefix_chars <= WITH_OVERLAP.overlap_chars


def test_a_clean_boundary_beats_an_exact_character_count(make_document: DocumentFactory):
    """The actual overlap is usually shorter than the budget, and that is correct."""
    chunks = _split_section_document(make_document)
    sizes = [c.provenance.overlap_prefix_chars for c in chunks if c.provenance.overlap_prefix_chars]

    assert sizes
    assert any(size < WITH_OVERLAP.overlap_chars for size in sizes)


def test_no_chunk_consists_only_of_repeated_text(make_document: DocumentFactory):
    chunks = _split_section_document(make_document)

    for chunk in chunks:
        own_content = chunk.content[chunk.provenance.overlap_prefix_chars :]
        assert own_content.strip()


def test_overlap_does_not_carry_half_a_code_fence_into_the_next_chunk(
    make_document: DocumentFactory,
):
    fence = "```python\n" + "\n".join(f"line_{index} = {index}" for index in range(6)) + "\n```"
    content = (
        f"## S\n\nIntro paragraph here.\n\n{fence}\n\n"
        "Trailing paragraph one.\n\nTrailing paragraph two.\n"
    )

    chunks = chunk_document(make_document(content), WITH_OVERLAP)

    for chunk in chunks:
        assert chunk.content.count("```") % 2 == 0


def test_overlap_is_deterministic(make_document: DocumentFactory):
    first = _split_section_document(make_document)
    second = _split_section_document(make_document)

    assert [c.content for c in first] == [c.content for c in second]
    assert [c.provenance.overlap_prefix_chars for c in first] == [
        c.provenance.overlap_prefix_chars for c in second
    ]


def test_overlap_changes_the_content_and_therefore_the_fingerprint(
    make_document: DocumentFactory,
):
    document = make_document(f"## S\n\n{LONG_SECTION}\n")
    with_overlap = chunk_document(document, WITH_OVERLAP)
    without_overlap = chunk_document(document, WITHOUT_OVERLAP)

    assert with_overlap[1].content != without_overlap[1].content
    assert with_overlap[1].provenance.fingerprint != without_overlap[1].provenance.fingerprint
