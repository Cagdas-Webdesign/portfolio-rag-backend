"""Chunk statistics — in particular, what counts as one section."""

from __future__ import annotations

from portfolio_rag.ingestion.chunking import ChunkingPolicy, ChunkStatistics, chunk_document
from tests.conftest import DocumentFactory

DEFAULT = ChunkingPolicy()
TIGHT = ChunkingPolicy(target_chars=120, max_chars=260, overlap_chars=60)


def _statistics(
    document_factory: DocumentFactory, body: str, policy: ChunkingPolicy = DEFAULT
) -> ChunkStatistics:
    return ChunkStatistics.of(chunk_document(document_factory(body), policy), policy)


def test_two_sections_sharing_a_heading_are_two_sections(make_document: DocumentFactory):
    """The heading path is a label, not an identity.

    `# Same` twice is two source sections that each produced one chunk — not
    one section that had to be split in two.
    """
    statistics = _statistics(make_document, "# Same\n\nFirst.\n\n# Same\n\nSecond.\n")

    assert statistics.chunks == 2
    assert statistics.sections == 2
    assert statistics.split_sections == 0


def test_a_genuinely_split_section_is_counted_as_split(make_document: DocumentFactory):
    body = "## S\n\n" + "\n\n".join(f"Paragraph {index} of one long section." for index in range(8))
    statistics = _statistics(make_document, body, TIGHT)

    assert statistics.chunks > 1
    assert statistics.sections == 1
    assert statistics.split_sections == 1


def test_distinct_headings_are_distinct_sections(make_document: DocumentFactory):
    body = "".join(f"## S{index}\n\nBody {index}.\n\n" for index in range(5))
    statistics = _statistics(make_document, body)

    assert statistics.sections == 5
    assert statistics.split_sections == 0


def test_repeated_headings_across_documents_do_not_collide(make_document: DocumentFactory):
    first = chunk_document(make_document("## Setup\n\nOne.\n", document_id="first"))
    second = chunk_document(make_document("## Setup\n\nTwo.\n", document_id="second"))

    statistics = ChunkStatistics.of([*first, *second], DEFAULT)

    assert statistics.documents == 2
    assert statistics.sections == 2
    assert statistics.split_sections == 0


def test_section_ordinals_number_the_sections_that_produced_chunks(
    make_document: DocumentFactory,
):
    """A heading with no body of its own contributes no section."""
    body = "Preamble.\n\n# A\n\n## B\n\nBody under B.\n\n## C\n\nBody under C.\n"
    chunks = chunk_document(make_document(body))

    assert [chunk.provenance.section_ordinal for chunk in chunks] == [0, 1, 2]
    assert [chunk.heading_path for chunk in chunks] == [(), ("A", "B"), ("A", "C")]


def test_an_empty_chunk_list_has_zero_everything():
    statistics = ChunkStatistics.of([], DEFAULT)

    assert statistics.sections == 0
    assert statistics.split_sections == 0
    assert statistics.chunks == 0


def test_sizes_and_overlap_counts_are_reported(make_document: DocumentFactory):
    body = "## S\n\n" + "\n\n".join(f"Paragraph {index} of one long section." for index in range(8))
    statistics = _statistics(make_document, body, TIGHT)

    assert statistics.smallest_chars <= statistics.average_chars <= statistics.largest_chars
    assert statistics.largest_chars <= TIGHT.max_chars
    assert statistics.over_max == 0
    assert statistics.chunks_with_overlap >= 1
