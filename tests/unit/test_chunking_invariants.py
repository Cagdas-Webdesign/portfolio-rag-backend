"""Invariants that must hold for every document, and proof that nothing is lost.

The golden tests pin down exact boundaries for specific inputs. These check the
promises that have to survive *any* input — including the one nobody thought
of. Source coverage is the important half: a chunker that silently drops the
last paragraph produces a corpus that is quietly wrong, and no amount of
retrieval tuning will ever fix it.
"""

from __future__ import annotations

import re

import pytest

from portfolio_rag.ingestion.chunking import ChunkingPolicy, chunk_document
from tests.conftest import DocumentFactory

DEFAULT = ChunkingPolicy()
TIGHT = ChunkingPolicy(target_chars=100, max_chars=150, overlap_chars=30)
NO_OVERLAP = ChunkingPolicy(target_chars=100, max_chars=150, overlap_chars=0)

SENTENCES = " ".join(f"This is sentence number {index} of a long paragraph." for index in range(20))

DOCUMENTS: dict[str, str] = {
    "single sentence": "One short line.",
    "no headings": "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
    "preamble then heading": "Preamble.\n\n# H\n\nBody.",
    "deep nesting": "# A\n\n## B\n\n### C\n\nBody one.\n\n#### D\n\nBody two.",
    "level jump": "## APIs\n\n#### Auth\n\nBody.",
    "repeated headings": "## X\n\nOne.\n\n## X\n\nTwo.",
    "heading only": "# A\n\n## B\n\nBody.",
    "many short sections": "".join(f"## S{i}\n\nBody {i}.\n\n" for i in range(12)),
    "long paragraph": SENTENCES,
    "wrapped paragraph": "\n".join(f"wrapped line {index} of text" for index in range(30)),
    "no sentence ends": " ".join(f"word{index:03d}" for index in range(60)),
    "single long token": "T" * 500,
    "umlauts": "# Überschrift\n\n" + "Grüße und Straßen. " * 20,
    "cjk": "# 見出し\n\n" + "これはテストです。" * 40,
    "list": "- one\n- two\n- three",
    "big list": "\n".join(f"- item number {index} with descriptive text" for index in range(30)),
    "nested list": "\n".join(f"- parent {i}\n  - child {i}" for i in range(20)),
    "code fence": "Intro.\n\n```py\nx = 1\n```\n\nOutro.",
    "quote": "> quoted line one\n> quoted line two",
    "links and emphasis": "See [docs](https://example.test) for **bold** and *italic* `code`.",
    "mixed": (
        "Preamble.\n\n# Guide\n\nIntro paragraph.\n\n## Setup\n\n"
        + SENTENCES
        + "\n\n```sh\nmake install\n```\n\n- step one\n- step two\n\n## Done\n\nFinal words."
    ),
}


@pytest.mark.parametrize("body", DOCUMENTS.values(), ids=list(DOCUMENTS))
@pytest.mark.parametrize(
    "policy", [DEFAULT, TIGHT, NO_OVERLAP], ids=["default", "tight", "no-overlap"]
)
def test_every_chunk_satisfies_the_basic_invariants(
    make_document: DocumentFactory, body: str, policy: ChunkingPolicy
):
    chunks = chunk_document(make_document(body, document_id="doc"), policy)

    for position, chunk in enumerate(chunks):
        assert chunk.content, "no chunk may be empty"
        assert chunk.content.strip(), "no chunk may be only whitespace"
        assert len(chunk.content) <= policy.max_chars, "the hard ceiling is not negotiable"
        assert chunk.id
        assert chunk.ordinal == position, "ordinals are gapless and start at 0"
        assert chunk.document_id == "doc"
        assert re.fullmatch(r"[0-9a-f]{64}", chunk.provenance.fingerprint)
        assert chunk.provenance.strategy_version
        assert chunk.provenance.document.source_path
        assert chunk.provenance.overlap_prefix_chars >= 0

    assert len({chunk.id for chunk in chunks}) == len(chunks), "ids are unique"


@pytest.mark.parametrize("body", DOCUMENTS.values(), ids=list(DOCUMENTS))
@pytest.mark.parametrize("policy", [DEFAULT, TIGHT], ids=["default", "tight"])
def test_chunking_is_reproducible(
    make_document: DocumentFactory, body: str, policy: ChunkingPolicy
):
    first = chunk_document(make_document(body), policy)
    second = chunk_document(make_document(body), policy)

    assert first == second


@pytest.mark.parametrize("body", DOCUMENTS.values(), ids=list(DOCUMENTS))
def test_no_artificial_overlap_when_it_is_switched_off(make_document: DocumentFactory, body: str):
    chunks = chunk_document(make_document(body), NO_OVERLAP)

    assert all(chunk.provenance.overlap_prefix_chars == 0 for chunk in chunks)


@pytest.mark.parametrize("body", DOCUMENTS.values(), ids=list(DOCUMENTS))
@pytest.mark.parametrize(
    "policy", [DEFAULT, TIGHT, NO_OVERLAP], ids=["default", "tight", "no-overlap"]
)
def test_code_fences_are_never_left_unbalanced(
    make_document: DocumentFactory, body: str, policy: ChunkingPolicy
):
    for chunk in chunk_document(make_document(body), policy):
        assert chunk.content.count("```") % 2 == 0


# --- source coverage --------------------------------------------------------


def _all_content(document_factory: DocumentFactory, body: str, policy: ChunkingPolicy) -> str:
    return "\n".join(chunk.content for chunk in chunk_document(document_factory(body), policy))


@pytest.mark.parametrize(
    "policy", [DEFAULT, TIGHT, NO_OVERLAP], ids=["default", "tight", "no-overlap"]
)
def test_the_first_and_last_blocks_both_survive(
    make_document: DocumentFactory, policy: ChunkingPolicy
):
    """Truncating the tail is the classic silent chunking bug."""
    body = "FIRST BLOCK.\n\n" + SENTENCES + "\n\n## Section\n\n" + SENTENCES + "\n\nLAST BLOCK."
    joined = _all_content(make_document, body, policy)

    assert "FIRST BLOCK." in joined
    assert "LAST BLOCK." in joined


@pytest.mark.parametrize(
    "policy", [DEFAULT, TIGHT, NO_OVERLAP], ids=["default", "tight", "no-overlap"]
)
def test_every_paragraph_of_a_multi_section_document_survives(
    make_document: DocumentFactory, policy: ChunkingPolicy
):
    paragraphs = [f"Paragraph marker {index} carries unique text." for index in range(24)]
    body = "".join(
        f"## Section {index // 4}\n\n{paragraph}\n\n" for index, paragraph in enumerate(paragraphs)
    )
    joined = _all_content(make_document, body, policy)

    for paragraph in paragraphs:
        assert paragraph in joined


@pytest.mark.parametrize("policy", [TIGHT, NO_OVERLAP], ids=["tight", "no-overlap"])
def test_an_oversized_paragraph_is_distributed_not_truncated(
    make_document: DocumentFactory, policy: ChunkingPolicy
):
    sentences = [f"Sentence {index} has its own marker." for index in range(30)]
    joined = _all_content(make_document, " ".join(sentences), policy)

    for sentence in sentences:
        assert sentence in joined


@pytest.mark.parametrize("policy", [TIGHT, NO_OVERLAP], ids=["tight", "no-overlap"])
def test_every_item_of_an_oversized_list_survives(
    make_document: DocumentFactory, policy: ChunkingPolicy
):
    items = [f"- item {index} with enough text to matter" for index in range(30)]
    joined = _all_content(make_document, "\n".join(items), policy)

    for item in items:
        assert item in joined


@pytest.mark.parametrize("policy", [DEFAULT, TIGHT], ids=["default", "tight"])
def test_a_code_block_survives_character_for_character(
    make_document: DocumentFactory, policy: ChunkingPolicy
):
    fence = "```python\ndef f(x):\n    return x * 2\n```"
    body = f"{SENTENCES}\n\n{fence}\n\n{SENTENCES}"

    assert fence in _all_content(make_document, body, policy)


@pytest.mark.parametrize(
    "policy", [DEFAULT, TIGHT, NO_OVERLAP], ids=["default", "tight", "no-overlap"]
)
def test_a_section_body_is_never_dropped_at_a_section_change(
    make_document: DocumentFactory, policy: ChunkingPolicy
):
    body = "".join(f"## S{index}\n\nUnique body {index}.\n\n" for index in range(15))
    joined = _all_content(make_document, body, policy)

    for index in range(15):
        assert f"Unique body {index}." in joined


def test_content_is_source_text_and_carries_no_parser_artifacts(
    make_document: DocumentFactory,
):
    body = "# Title\n\nParagraph with **bold**.\n\n- item\n\n```py\nx = 1\n```"
    joined = _all_content(make_document, body, DEFAULT)

    assert "<p>" not in joined
    assert "<strong>" not in joined
    assert "paragraph_open" not in joined
    assert "**bold**" in joined
    assert "# Title" not in joined, "the heading is structure, kept in heading_path"
