"""The context builder — including golden tests that pin the exact string.

Golden tests are worth their maintenance cost here. The context is the only
thing a model ever sees of the corpus, so a change to it changes every answer
this system produces; pinning it means such a change has to be made on purpose
and shows up in a diff.
"""

from __future__ import annotations

from typing import Any

import pytest

from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.rag.context import CONTEXT_REPRESENTATION_VERSION, build_context
from portfolio_rag.rag.errors import ContextBudgetError
from portfolio_rag.rag.tokens import estimate_tokens
from tests.doubles import make_chunk

GENEROUS = 10_000


def retrieved(
    chunk_id: str, content: str, similarity: float = 0.9, **overrides: Any
) -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(chunk_id, content, **overrides), similarity=similarity)


# --- golden representation ---------------------------------------------------


def test_one_passage_renders_exactly_like_this():
    context = build_context(
        [retrieved("doc--0000", "REST endpoints are documented here.")],
        available_tokens=GENEROUS,
    )

    assert context.text == (
        "[SOURCE S1]\n"
        "Document: Test Document\n"
        "Section: Section\n"
        "Content:\n"
        "REST endpoints are documented here."
    )


def test_several_passages_are_separated_by_one_blank_line():
    context = build_context(
        [
            retrieved("aa--0000", "First fact.", document_id="aa", title="Doc A"),
            retrieved("bb--0000", "Second fact.", document_id="bb", title="Doc B"),
        ],
        available_tokens=GENEROUS,
    )

    assert context.text == (
        "[SOURCE S1]\n"
        "Document: Doc A\n"
        "Section: Section\n"
        "Content:\n"
        "First fact.\n"
        "\n"
        "[SOURCE S2]\n"
        "Document: Doc B\n"
        "Section: Section\n"
        "Content:\n"
        "Second fact."
    )


def test_a_passage_above_the_first_heading_omits_the_section_line():
    """An empty heading path is not rendered as an empty label."""
    context = build_context(
        [retrieved("doc--0000", "Intro text.", heading_path=())],
        available_tokens=GENEROUS,
    )

    assert context.text == ("[SOURCE S1]\nDocument: Test Document\nContent:\nIntro text.")
    assert "Section:" not in context.text


def test_a_nested_heading_path_is_rendered_as_a_path():
    context = build_context(
        [retrieved("doc--0000", "Token rules.", heading_path=("Backend", "APIs", "Auth"))],
        available_tokens=GENEROUS,
    )

    assert "Section: Backend > APIs > Auth" in context.text


def test_unicode_and_umlauts_survive_verbatim():
    content = "Die Prüfung läuft über die Schnittstelle — größtenteils automatisch. 🙂"

    context = build_context(
        [retrieved("doc--0000", content, title="Übersicht", heading_path=("Prüfung",))],
        available_tokens=GENEROUS,
    )

    assert content in context.text
    assert "Document: Übersicht" in context.text
    assert "Section: Prüfung" in context.text


def test_markdown_content_is_carried_through_unchanged():
    content = "Use this:\n\n```python\nprint('hi')\n```\n\n- item one\n- item two"

    context = build_context([retrieved("doc--0000", content)], available_tokens=GENEROUS)

    assert content in context.text


def test_the_same_input_produces_the_same_context_every_time():
    passages = [
        retrieved("aa--0000", "First.", document_id="aa"),
        retrieved("bb--0000", "Second.", document_id="bb"),
    ]

    assert (
        build_context(passages, available_tokens=GENEROUS).text
        == build_context(passages, available_tokens=GENEROUS).text
    )


def test_the_representation_is_versioned():
    assert CONTEXT_REPRESENTATION_VERSION == "grounded-context-v1"


# --- labels ------------------------------------------------------------------


def test_labels_are_assigned_in_order_starting_at_one():
    context = build_context(
        [
            retrieved("aa--0000", "First.", document_id="aa"),
            retrieved("bb--0000", "Second.", document_id="bb"),
            retrieved("cc--0000", "Third.", document_id="cc"),
        ],
        available_tokens=GENEROUS,
    )

    assert context.labels == ("S1", "S2", "S3")


def test_the_same_input_produces_the_same_labels():
    passages = [retrieved("aa--0000", "First.", document_id="aa")]

    assert build_context(passages, available_tokens=GENEROUS).labels == ("S1",)
    assert build_context(passages, available_tokens=GENEROUS).labels == ("S1",)


def test_a_label_is_not_derived_from_any_identifier():
    """Labels are local to one answer; nothing about the corpus leaks through them."""
    context = build_context(
        [retrieved("architecture--0007", "Text.", document_id="architecture")],
        available_tokens=GENEROUS,
    )

    (source,) = context.sources
    assert source.label == "S1"
    assert "architecture" not in source.label


def test_a_label_maps_back_to_the_passage_it_was_minted_for():
    context = build_context(
        [
            retrieved("aa--0000", "First.", document_id="aa"),
            retrieved("bb--0000", "Second.", document_id="bb"),
        ],
        available_tokens=GENEROUS,
    )

    second = context.source("S2")
    assert second is not None
    assert second.chunk_id == "bb--0000"


def test_an_unknown_label_resolves_to_nothing():
    context = build_context([retrieved("aa--0000", "First.")], available_tokens=GENEROUS)

    assert context.source("S9") is None


# --- deduplication -----------------------------------------------------------


def test_the_same_chunk_offered_twice_appears_once():
    passage = retrieved("aa--0000", "First.")

    context = build_context([passage, passage], available_tokens=GENEROUS)

    assert context.labels == ("S1",)
    assert context.duplicates_removed == 1


def test_two_ids_with_identical_rendering_appear_once():
    """The same passage reaching the context by two paths is still one passage."""
    context = build_context(
        [retrieved("aa--0000", "Same text."), retrieved("aa--0000", "Same text.", similarity=0.5)],
        available_tokens=GENEROUS,
    )

    assert len(context.sources) == 1


def test_overlapping_but_different_passages_are_both_kept():
    """Controlled overlap must not be destroyed by naive deduplication."""
    first = retrieved("doc--0000", "Shared sentence. First unique part.")
    second = retrieved("doc--0001", "Shared sentence. Second unique part.")

    context = build_context([first, second], available_tokens=GENEROUS)

    assert context.labels == ("S1", "S2")
    assert context.duplicates_removed == 0
    assert "First unique part." in context.text
    assert "Second unique part." in context.text


def test_two_sections_of_one_document_are_both_kept():
    context = build_context(
        [
            retrieved("doc--0000", "About APIs.", heading_path=("APIs",)),
            retrieved("doc--0001", "About storage.", heading_path=("Storage",)),
        ],
        available_tokens=GENEROUS,
    )

    assert context.labels == ("S1", "S2")


# --- budget ------------------------------------------------------------------


def test_everything_fits_when_the_budget_is_generous():
    context = build_context(
        [retrieved(f"doc--{index:04d}", f"Fact {index}.") for index in range(5)],
        available_tokens=GENEROUS,
    )

    assert len(context.sources) == 5
    assert context.skipped_for_budget == 0


def test_a_passage_that_fits_exactly_is_included():
    passage = retrieved("doc--0000", "A short fact.")
    exact = build_context([passage], available_tokens=GENEROUS).estimated_tokens

    context = build_context([passage], available_tokens=exact)

    assert len(context.sources) == 1


def test_one_token_less_is_not_enough():
    passage = retrieved("doc--0000", "A short fact.")
    exact = build_context([passage], available_tokens=GENEROUS).estimated_tokens

    with pytest.raises(ContextBudgetError):
        build_context([passage], available_tokens=exact - 1)


def test_the_higher_ranked_passage_wins_the_budget():
    strong = retrieved("aa--0000", "Relevant fact. " * 20, similarity=0.9, document_id="aa")
    weak = retrieved("bb--0000", "Less relevant fact. " * 20, similarity=0.4, document_id="bb")
    room = build_context([strong], available_tokens=GENEROUS).estimated_tokens + 5

    context = build_context([strong, weak], available_tokens=room)

    assert [source.chunk_id for source in context.sources] == ["aa--0000"]
    assert context.skipped_for_budget == 1


def test_a_short_passage_after_a_long_one_still_gets_its_place():
    """Skipping is per passage, not a stop: a long fourth must not cost a short fifth."""
    huge = retrieved("aa--0000", "x" * 5_000, document_id="aa")
    small = retrieved("bb--0000", "Tiny.", document_id="bb")
    room = build_context([small], available_tokens=GENEROUS).estimated_tokens + 200

    context = build_context([huge, small], available_tokens=room)

    assert [source.chunk_id for source in context.sources] == ["bb--0000"]
    assert context.skipped_for_budget == 1


def test_no_passage_is_ever_cut_in_half():
    long_content = "Sentence one. Sentence two. Sentence three. " * 30
    passage = retrieved("doc--0000", long_content)
    room = build_context([passage], available_tokens=GENEROUS).estimated_tokens

    context = build_context([passage], available_tokens=room)

    assert long_content in context.text


def test_a_passage_that_cannot_fit_at_all_is_a_controlled_error():
    """Silent truncation is the one outcome a context builder must not have."""
    with pytest.raises(ContextBudgetError):
        build_context([retrieved("doc--0000", "x" * 10_000)], available_tokens=10)


def test_no_budget_at_all_is_a_controlled_error():
    with pytest.raises(ContextBudgetError):
        build_context([retrieved("doc--0000", "Fact.")], available_tokens=0)


def test_nothing_retrieved_is_an_empty_context_and_not_an_error():
    context = build_context([], available_tokens=0)

    assert context.is_empty
    assert context.text == ""
    assert context.estimated_tokens == 0


def test_the_reported_token_count_matches_the_rendered_text():
    context = build_context(
        [retrieved("aa--0000", "First."), retrieved("bb--0000", "Second.", document_id="bb")],
        available_tokens=GENEROUS,
    )

    assert context.estimated_tokens == estimate_tokens(context.text)
