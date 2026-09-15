"""Size bounding: the hard ceiling, the fallback ladder, code and lists."""

from __future__ import annotations

import pytest

from portfolio_rag.ingestion.chunking import (
    ChunkingPolicy,
    UnsplittableBlockError,
    chunk_document,
)
from tests.conftest import DocumentFactory

SMALL = ChunkingPolicy(target_chars=100, max_chars=120, overlap_chars=0)


def _contents(document_factory: DocumentFactory, content: str, policy: ChunkingPolicy) -> list[str]:
    return [chunk.content for chunk in chunk_document(document_factory(content), policy)]


# --- the hard ceiling -------------------------------------------------------


def test_no_chunk_exceeds_the_maximum(make_document: DocumentFactory):
    body = "\n\n".join(f"Paragraph number {index}. " * 4 for index in range(12))

    for content in _contents(make_document, body, SMALL):
        assert len(content) <= SMALL.max_chars


def test_a_paragraph_that_fits_is_never_cut(make_document: DocumentFactory):
    paragraph = "X" * 110

    assert _contents(make_document, paragraph, SMALL) == [paragraph]


# --- oversized paragraph: sentence rung -------------------------------------


def test_a_long_single_line_paragraph_splits_on_sentence_boundaries(
    make_document: DocumentFactory,
):
    sentences = [f"This is sentence number {index}." for index in range(1, 13)]
    contents = _contents(make_document, " ".join(sentences), SMALL)

    assert len(contents) > 1
    for content in contents:
        assert len(content) <= SMALL.max_chars
        # Every piece starts and ends on a whole sentence.
        assert content.endswith(".")
    for sentence in sentences:
        assert sentence in " ".join(contents)


@pytest.mark.parametrize("terminator", [".", "!", "?", "…"])
def test_german_and_other_sentence_terminators_are_recognised(
    make_document: DocumentFactory, terminator: str
):
    sentence = f"Das ist ein Satz{terminator}"
    contents = _contents(make_document, " ".join([sentence] * 12), SMALL)

    assert len(contents) > 1
    assert all(content.endswith(terminator) for content in contents)


def test_a_source_wrapped_paragraph_splits_on_line_boundaries_first(
    make_document: DocumentFactory,
):
    """Line breaks the author wrote are the cleanest available boundary."""
    lines = [f"line {index} of a wrapped paragraph" for index in range(10)]
    contents = _contents(make_document, "\n".join(lines), SMALL)

    assert len(contents) > 1
    for content in contents:
        assert len(content) <= SMALL.max_chars
        for line in content.split("\n"):
            assert line in lines


# --- oversized paragraph: whitespace rung -----------------------------------


def test_a_long_paragraph_without_sentence_ends_falls_back_to_whitespace(
    make_document: DocumentFactory,
):
    words = [f"word{index:03d}" for index in range(40)]
    contents = _contents(make_document, " ".join(words), SMALL)

    assert len(contents) > 1
    for content in contents:
        assert len(content) <= SMALL.max_chars
    # No word was cut in half.
    for content in contents:
        for word in content.split(" "):
            assert word in words
    assert set(words) == {word for content in contents for word in content.split(" ")}


# --- oversized paragraph: hard cut ------------------------------------------


def test_a_single_token_longer_than_the_maximum_is_cut(make_document: DocumentFactory):
    """Last resort: a base64 blob has no boundary to respect."""
    token = "Z" * 400
    contents = _contents(make_document, token, SMALL)

    assert len(contents) == 4
    assert all(len(content) <= SMALL.max_chars for content in contents)
    assert "".join(contents) == token


def test_a_long_token_beside_normal_words_does_not_damage_the_words(
    make_document: DocumentFactory,
):
    token = "Q" * 300
    contents = _contents(make_document, f"before {token} after", SMALL)

    joined = "".join(contents)
    assert "before" in joined
    assert "after" in joined
    assert token in joined.replace("\n", "")


def test_unicode_is_split_on_characters_not_bytes(make_document: DocumentFactory):
    body = "漢" * 300
    contents = _contents(make_document, body, SMALL)

    assert all(len(content) <= SMALL.max_chars for content in contents)
    assert "".join(contents) == body


# --- code blocks ------------------------------------------------------------


def test_a_code_fence_within_the_limit_stays_intact(make_document: DocumentFactory):
    fence = '```python\ndef greet() -> str:\n    return "hello"\n```'
    contents = _contents(make_document, f"Before.\n\n{fence}\n\nAfter.", SMALL)

    assert any(fence in content for content in contents)
    for content in contents:
        assert content.count("```") % 2 == 0


def test_a_code_fence_near_the_limit_is_still_one_piece(make_document: DocumentFactory):
    lines = "\n".join(f"x{index} = {index}" for index in range(12))
    fence = f"```python\n{lines}\n```"
    assert len(fence) <= SMALL.max_chars

    assert fence in _contents(make_document, fence, SMALL)


def test_a_code_fence_over_the_limit_fails_loudly(make_document: DocumentFactory):
    lines = "\n".join(f"variable_{index} = {index}" for index in range(40))
    document = make_document(f"# Examples\n\n```python\n{lines}\n```\n")

    with pytest.raises(UnsplittableBlockError) as caught:
        chunk_document(document, SMALL)

    assert caught.value.document_id == "sample"
    assert caught.value.heading_path == ("Examples",)
    assert "UNSPLITTABLE_BLOCK" in caught.value.describe()


def test_an_oversized_fence_inside_a_list_item_also_fails_loudly(
    make_document: DocumentFactory,
):
    """Better a clear error than an unterminated fence in the index."""
    lines = "\n".join(f"  value_{index} = {index}" for index in range(40))
    document = make_document(f"- item with code:\n\n  ```python\n{lines}\n  ```\n")

    with pytest.raises(UnsplittableBlockError):
        chunk_document(document, SMALL)


def test_a_document_of_many_small_fences_chunks_normally(make_document: DocumentFactory):
    body = "\n\n".join(f"```py\nx = {index}\n```" for index in range(6))
    contents = _contents(make_document, body, SMALL)

    for content in contents:
        assert content.count("```") % 2 == 0


# --- lists ------------------------------------------------------------------


def test_a_list_that_fits_is_never_broken_up(make_document: DocumentFactory):
    body = "- REST\n- GraphQL\n- Webhooks\n- SOAP"

    assert _contents(make_document, body, SMALL) == [body]


def test_an_ordered_list_that_fits_stays_whole(make_document: DocumentFactory):
    body = "1. first\n2. second\n3. third"

    assert _contents(make_document, body, SMALL) == [body]


def test_an_oversized_list_splits_between_items(make_document: DocumentFactory):
    items = [f"- item number {index} with some descriptive text" for index in range(12)]
    contents = _contents(make_document, "\n".join(items), SMALL)

    assert len(contents) > 1
    for content in contents:
        assert len(content) <= SMALL.max_chars
        for line in content.split("\n"):
            assert line in items
    assert set(items) == {line for content in contents for line in content.split("\n")}


def test_splitting_an_oversized_list_keeps_nested_items_with_their_parent(
    make_document: DocumentFactory,
):
    items = [f"- parent {index}\n  - child of {index}" for index in range(10)]
    contents = _contents(make_document, "\n".join(items), SMALL)

    for content in contents:
        for line in content.split("\n"):
            if line.startswith("  - child of "):
                index = line.removeprefix("  - child of ")
                assert f"- parent {index}" in content


def test_a_paragraph_after_an_oversized_list_is_not_lost(make_document: DocumentFactory):
    items = "\n".join(f"- item number {index} with text" for index in range(12))
    contents = _contents(make_document, f"{items}\n\nClosing paragraph.", SMALL)

    assert any("Closing paragraph." in content for content in contents)
