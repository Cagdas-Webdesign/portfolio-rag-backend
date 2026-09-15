"""Heading hierarchy and Markdown block recognition."""

from __future__ import annotations

import pytest

from portfolio_rag.ingestion.chunking.structure import BlockKind, analyze_structure


def _paths(content: str) -> list[tuple[str, ...]]:
    return [section.heading_path for section in analyze_structure(content)]


def _kinds(content: str) -> list[BlockKind]:
    return [block.kind for section in analyze_structure(content) for block in section.blocks]


# --- the heading stack ------------------------------------------------------


def test_a_single_heading_is_a_one_element_path():
    assert _paths("# A\n\nBody.\n") == [("A",)]


def test_nested_headings_accumulate():
    assert _paths("# A\n\n## B\n\nBody.\n") == [("A", "B")]
    assert _paths("# A\n\n## B\n\n### C\n\nBody.\n") == [("A", "B", "C")]


def test_a_level_jump_nests_without_inventing_the_missing_level():
    """`# A` then `### C` gives ('A', 'C') — the document said nothing about H2."""
    assert _paths("# A\n\n### C\n\nBody.\n") == [("A", "C")]
    assert _paths("## APIs\n\n#### Authentication\n\nBody.\n") == [("APIs", "Authentication")]


def test_returning_to_a_shallower_level_pops_everything_below_it():
    content = """# A

## B

### C

Body under C.

## D

Body under D.
"""

    assert _paths(content) == [("A", "B", "C"), ("A", "D")]


def test_a_deep_jump_back_out_pops_several_levels():
    content = "## APIs\n\n#### Auth\n\nBody one.\n\n## Other\n\nBody two.\n"

    assert _paths(content) == [("APIs", "Auth"), ("Other",)]


def test_repeated_heading_names_produce_repeated_paths():
    """Two sections may legitimately share a name; the path is not an id."""
    content = """## Setup

First body.

## Usage

Second body.

## Setup

Third body.
"""

    assert _paths(content) == [("Setup",), ("Usage",), ("Setup",)]


def test_the_same_name_at_different_depths_stays_distinguishable():
    content = "# Backend\n\n## Backend\n\nBody.\n"

    assert _paths(content) == [("Backend", "Backend")]


def test_all_six_heading_levels_are_understood():
    content = "".join(f"{'#' * level} H{level}\n\n" for level in range(1, 7)) + "Body.\n"

    assert _paths(content) == [("H1", "H2", "H3", "H4", "H5", "H6")]


def test_setext_headings_are_recognised_like_atx_ones():
    assert _paths("Title\n=====\n\nBody.\n") == [("Title",)]
    assert _paths("# A\n\nSub\n---\n\nBody.\n") == [("A", "Sub")]


def test_a_heading_keeps_its_inline_markdown_rather_than_being_rendered():
    assert _paths("## API **v2**\n\nBody.\n") == [("API **v2**",)]


def test_an_empty_heading_closes_levels_but_adds_no_path_entry():
    """`##` with no text is not a section name, so it contributes none."""
    assert _paths("# A\n\n## B\n\n##\n\nBody.\n") == [("A",)]


# --- block recognition ------------------------------------------------------


def test_paragraphs_lists_quotes_and_code_are_told_apart():
    content = """Paragraph.

- item

> quote

```py
code
```
"""

    assert _kinds(content) == [
        BlockKind.PARAGRAPH,
        BlockKind.LIST,
        BlockKind.QUOTE,
        BlockKind.CODE,
    ]


def test_ordered_lists_are_lists_too():
    assert _kinds("1. one\n2. two\n") == [BlockKind.LIST]


def test_indented_code_is_code():
    assert _kinds("Intro:\n\n    indented = True\n") == [BlockKind.PARAGRAPH, BlockKind.CODE]


@pytest.mark.parametrize(
    "fence",
    ["```python\n# comment\n```", "```\n# comment\n```", "~~~js\n// x\n~~~"],
)
def test_a_hash_inside_a_code_fence_is_not_a_heading(fence: str):
    """The reason a real parser is used instead of a line scanner."""
    sections = analyze_structure(f"# Real\n\n{fence}\n")

    assert [section.heading_path for section in sections] == [("Real",)]
    assert sections[0].blocks[0].kind is BlockKind.CODE
    assert sections[0].blocks[0].is_atomic


def test_markdown_syntax_inside_a_code_fence_is_not_parsed_as_structure():
    content = "```md\n# Not a heading\n\n- not a list\n```\n"
    sections = analyze_structure(content)

    assert len(sections) == 1
    assert sections[0].heading_path == ()
    assert len(sections[0].blocks) == 1


def test_a_lists_direct_items_become_its_split_units():
    (section,) = analyze_structure("- one\n- two\n- three\n")

    assert section.blocks[0].split_units == ("- one", "- two", "- three")


def test_a_nested_list_stays_inside_its_parent_item():
    (section,) = analyze_structure("- outer\n  - inner one\n  - inner two\n- second\n")

    assert section.blocks[0].split_units == ("- outer\n  - inner one\n  - inner two", "- second")


def test_only_code_blocks_are_atomic():
    (section,) = analyze_structure("Paragraph.\n\n- item\n\n```\ncode\n```\n")
    atomic = [block.kind for block in section.blocks if block.is_atomic]

    assert atomic == [BlockKind.CODE]


def test_raw_html_is_treated_as_content_not_executed_or_rendered():
    content = '<div onclick="alert(1)">text</div>\n'
    sections = analyze_structure(content)

    assert len(sections) == 1
    assert "onclick" in sections[0].blocks[0].text


def test_an_empty_document_body_yields_no_sections():
    assert analyze_structure("") == ()
    assert analyze_structure("\n\n") == ()
