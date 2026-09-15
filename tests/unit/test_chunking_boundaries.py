"""Golden boundary tests: exactly where the chunker cuts, and what it produces.

These assert whole expected outputs rather than properties. They are the tests
that fail loudly when someone changes the packing rule, which is the point:
chunk boundaries are a contract with everything downstream, and "roughly the
same chunks" is not a thing.
"""

from __future__ import annotations

from portfolio_rag.ingestion.chunking import ChunkingPolicy, chunk_document
from tests.conftest import DocumentFactory

#: (ordinal, id, heading_path, content) — the whole observable shape of a chunk
#: that a boundary test cares about.
ChunkShape = tuple[int, str, tuple[str, ...], str]


def _summary(document_factory: DocumentFactory, content: str, **kwargs: object) -> list[ChunkShape]:
    chunks = chunk_document(document_factory(content, **kwargs))
    return [(c.ordinal, c.id, c.heading_path, c.content) for c in chunks]


def test_the_documented_example_cuts_exactly_here(make_document: DocumentFactory):
    content = """# APIs

Paragraph A.

Paragraph B.

## REST

Paragraph C.
"""

    assert _summary(make_document, content) == [
        (0, "sample--0000", ("APIs",), "Paragraph A.\n\nParagraph B."),
        (1, "sample--0001", ("APIs", "REST"), "Paragraph C."),
    ]


def test_a_very_short_document_is_a_single_chunk(make_document: DocumentFactory):
    assert _summary(make_document, "Just one sentence.") == [
        (0, "sample--0000", (), "Just one sentence.")
    ]


def test_a_document_without_headings_keeps_an_empty_heading_path(
    make_document: DocumentFactory,
):
    content = "First paragraph.\n\nSecond paragraph.\n"

    assert _summary(make_document, content) == [
        (0, "sample--0000", (), "First paragraph.\n\nSecond paragraph.")
    ]


def test_content_before_the_first_heading_is_kept_in_its_own_section(
    make_document: DocumentFactory,
):
    """No invented "Introduction" heading — the path is simply empty."""
    content = """Preamble text.

# Real Heading

Body text.
"""

    assert _summary(make_document, content) == [
        (0, "sample--0000", (), "Preamble text."),
        (1, "sample--0001", ("Real Heading",), "Body text."),
    ]


def test_short_sections_stay_separate_instead_of_being_merged(
    make_document: DocumentFactory,
):
    """Two subjects under two headings are two units, however small they are."""
    content = """## REST

Short paragraph.

## Webhooks

Another short paragraph.
"""
    chunks = _summary(make_document, content)

    assert [chunk[2] for chunk in chunks] == [("REST",), ("Webhooks",)]
    assert len(chunks) == 2


def test_a_heading_without_a_body_produces_no_chunk(make_document: DocumentFactory):
    content = """# Backend

## APIs

Only this has content.
"""

    assert _summary(make_document, content) == [
        (0, "sample--0000", ("Backend", "APIs"), "Only this has content.")
    ]


def test_a_trailing_heading_with_nothing_under_it_produces_no_chunk(
    make_document: DocumentFactory,
):
    content = "# A\n\nBody.\n\n## Empty Tail\n"

    assert _summary(make_document, content) == [(0, "sample--0000", ("A",), "Body.")]


def test_blocks_are_rejoined_with_a_blank_line(make_document: DocumentFactory):
    content = "Paragraph one.\n\n\n\nParagraph two.\n"

    assert _summary(make_document, content) == [
        (0, "sample--0000", (), "Paragraph one.\n\nParagraph two.")
    ]


def test_a_list_and_its_surrounding_paragraphs_pack_together(
    make_document: DocumentFactory,
):
    content = """## Protocols

We support these:

- REST
- GraphQL
- Webhooks

Ask for details.
"""

    assert _summary(make_document, content) == [
        (
            0,
            "sample--0000",
            ("Protocols",),
            "We support these:\n\n- REST\n- GraphQL\n- Webhooks\n\nAsk for details.",
        )
    ]


def test_a_section_over_the_target_is_split_at_a_block_boundary(
    make_document: DocumentFactory,
):
    """Packing stops at the target and never cuts inside a paragraph that fits."""
    first = "A" * 80
    second = "B" * 80
    third = "C" * 80
    document = make_document(f"## S\n\n{first}\n\n{second}\n\n{third}\n")

    policy = ChunkingPolicy(target_chars=100, max_chars=200, overlap_chars=0)
    chunks = chunk_document(document, policy)

    assert [chunk.content for chunk in chunks] == [f"{first}\n\n{second}", third]
    assert all(chunk.heading_path == ("S",) for chunk in chunks)


def test_markdown_inline_formatting_is_carried_through_untouched(
    make_document: DocumentFactory,
):
    content = "Text with **bold**, *italic*, `code` and a [link](https://example.test).\n"

    assert _summary(make_document, content)[0][3] == (
        "Text with **bold**, *italic*, `code` and a [link](https://example.test)."
    )


def test_a_blockquote_survives_as_one_block(make_document: DocumentFactory):
    content = "## Quote\n\n> First line.\n> Second line.\n"

    assert _summary(make_document, content) == [
        (0, "sample--0000", ("Quote",), "> First line.\n> Second line.")
    ]


def test_umlauts_and_cjk_pass_through_unchanged(make_document: DocumentFactory):
    content = "# Überschrift\n\nGrüße, Straße, 漢字テスト.\n"

    assert _summary(make_document, content) == [
        (0, "sample--0000", ("Überschrift",), "Grüße, Straße, 漢字テスト.")
    ]


def test_no_chunk_is_produced_for_a_document_of_only_headings(
    make_document: DocumentFactory,
):
    assert chunk_document(make_document("# A\n\n## B\n\n### C\n")) == ()
