"""Renumbering verified citations for a reader — and nothing else.

Every test here runs `resolve_citations` first and `present_answer` second,
which is the order the answer service uses. That is deliberate: it keeps the
internal `S`-vocabulary visible right up to the moment validation is done, and
makes any test that tried to number an unverified label fail on the validation
step rather than on the presentation one.
"""

from __future__ import annotations

import pytest

from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.rag.citations import resolve_citations
from portfolio_rag.rag.context import GroundedContext, build_context
from portfolio_rag.rag.presentation import present_answer
from tests.unit.test_context_builder import GENEROUS, retrieved


def context_of(*passages: RetrievedChunk) -> GroundedContext:
    return build_context(list(passages), available_tokens=GENEROUS)


def many_sources(count: int = 12) -> GroundedContext:
    """Enough passages that two-digit labels exist and can be told apart."""
    return context_of(
        *(
            retrieved(
                f"d{index:02d}--0000",
                f"Fact number {index}.",
                document_id=f"d{index:02d}",
                title=f"Doc {index}",
                source=f"d{index}.md",
            )
            for index in range(1, count + 1)
        )
    )


def present(answer: str, *labels: str, context: GroundedContext | None = None):
    """Validate first, present second — the production order."""
    outcome = resolve_citations(context or many_sources(), list(labels))
    return present_answer(answer, outcome.citations, outcome.cited), outcome


# --- the basic mapping --------------------------------------------------------


def test_the_first_verified_mark_becomes_one():
    presented, _ = present("FastAPI is used [S1].", "S1")

    assert presented.answer == "FastAPI is used [1]."


def test_numbering_starts_at_one_whatever_the_internal_label_was():
    """`[S3]` as the first mark is `[1]`, never `[3]`."""
    presented, _ = present("Vectorize stores the vectors [S3].", "S3")

    assert presented.answer == "Vectorize stores the vectors [1]."
    assert "[3]" not in presented.answer


def test_marks_are_numbered_in_the_order_the_text_uses_them():
    presented, _ = present("A [S4]. B [S1]. C [S7].", "S4", "S1", "S7")

    assert presented.answer == "A [1]. B [2]. C [3]."


def test_the_text_decides_the_order_not_the_models_source_list():
    """The reader meets the marks in the prose, so the prose numbers them."""
    presented, _ = present("First [S7]. Then [S2].", "S2", "S7")

    assert presented.answer == "First [1]. Then [2]."
    assert [citation.title for citation in presented.citations] == ["Doc 7", "Doc 2"]


def test_the_numbering_is_dense():
    presented, _ = present("A [S5]. B [S9]. C [S12].", "S5", "S9", "S12")

    assert presented.answer == "A [1]. B [2]. C [3]."


# --- repeated sources ---------------------------------------------------------


def test_a_source_used_twice_keeps_one_number():
    presented, _ = present("Python [S4]. FastAPI [S4]. Vectorize [S7].", "S4", "S7")

    assert presented.answer == "Python [1]. FastAPI [1]. Vectorize [2]."


def test_a_source_used_twice_is_listed_once():
    presented, _ = present("Python [S4]. FastAPI [S4].", "S4")

    assert len(presented.citations) == 1


def test_two_labels_resolving_to_one_source_share_a_number():
    """Validation collapses passages from one document and section into one
    citation. The reader should see one number for one source."""
    context = context_of(
        retrieved("doc--0000", "First half.", document_id="doc", title="Doc", source="d.md"),
        retrieved("doc--0001", "Second half.", document_id="doc", title="Doc", source="d.md"),
    )
    presented, outcome = present("A [S1]. B [S2].", "S1", "S2", context=context)

    assert len(outcome.citations) == 1, "validation already made these one source"
    assert presented.answer == "A [1]. B [1]."
    assert len(presented.citations) == 1


# --- S1 vs S10 ----------------------------------------------------------------


def test_a_single_digit_label_is_not_matched_inside_a_two_digit_one():
    presented, _ = present("A [S1]. B [S10].", "S1", "S10")

    assert presented.answer == "A [1]. B [2]."


def test_the_same_holds_when_the_longer_label_comes_first():
    presented, _ = present("A [S10]. B [S1].", "S10", "S1")

    assert presented.answer == "A [1]. B [2]."


@pytest.mark.parametrize("label", ["S1", "S2", "S9", "S10", "S12"])
def test_every_label_width_maps_correctly(label: str):
    presented, _ = present(f"Only this one [{label}].", label)

    assert presented.answer == "Only this one [1]."


def test_a_three_digit_label_maps_correctly():
    presented, _ = present("A [S100].", "S100", context=many_sources(100))

    assert presented.answer == "A [1]."


def test_three_digit_and_single_digit_labels_do_not_collide():
    presented, _ = present(
        "A [S100]. B [S10]. C [S1].", "S100", "S10", "S1", context=many_sources(100)
    )

    assert presented.answer == "A [1]. B [2]. C [3]."


# --- only verified sources become visible -------------------------------------


def test_a_mark_for_an_unverified_label_is_removed_not_renumbered():
    """Renumbering it would invent a citation at the last possible moment."""
    presented, outcome = present("FastAPI is used [S1]. Something else [S99].", "S1", "S99")

    assert outcome.unknown_labels == ("S99",)
    assert presented.answer == "FastAPI is used [1]. Something else."
    assert presented.removed_marks == 1
    assert len(presented.citations) == 1


def test_removing_a_mark_does_not_leave_a_double_space():
    presented, _ = present("Python [S99] and FastAPI [S1].", "S1", "S99")

    assert presented.answer == "Python and FastAPI [1]."
    assert "  " not in presented.answer


def test_a_mark_the_model_never_declared_is_also_removed():
    """Prose can refer to a label the model left out of its source list."""
    presented, _ = present("Stated [S1]. Unstated [S2].", "S1")

    assert presented.answer == "Stated [1]. Unstated."


def test_no_internal_label_survives_in_the_published_text():
    presented, _ = present("A [S1]. B [S10]. C [S99]. D [S4].", "S1", "S10", "S4", "S99")

    assert "S1" not in presented.answer
    assert "S10" not in presented.answer
    assert "S99" not in presented.answer
    assert "[S" not in presented.answer


# --- the list and the numbers agree -------------------------------------------


def test_the_citation_list_is_ordered_by_the_visible_numbering():
    presented, _ = present("A [S7]. B [S2]. C [S5].", "S2", "S5", "S7")

    assert [citation.title for citation in presented.citations] == ["Doc 7", "Doc 2", "Doc 5"]


def test_every_visible_number_indexes_its_own_citation():
    presented, _ = present("A [S4]. B [S1]. C [S4]. D [S9].", "S1", "S4", "S9")

    for position, citation in enumerate(presented.citations, 1):
        assert f"[{position}]" in presented.answer
        assert citation.title


def test_a_verified_source_the_prose_never_marked_is_still_listed():
    """Validation admitted it; presentation does not get to overrule that."""
    presented, outcome = present("Only one mark [S1].", "S1", "S5")

    assert len(outcome.citations) == 2
    assert len(presented.citations) == 2
    assert presented.citations[0].title == "Doc 1", "the marked one is numbered first"
    assert presented.citations[1].title == "Doc 5"


def test_readable_sources_come_from_backend_metadata():
    presented, _ = present("A [S1].", "S1")

    (citation,) = presented.citations
    assert (citation.title, citation.document_id, citation.source) == ("Doc 1", "d01", "d1.md")


# --- answers with nothing to number -------------------------------------------


def test_an_answer_without_citations_keeps_none():
    presented, _ = present("I don't have enough information to answer that.")

    assert presented.citations == ()
    assert presented.answer == "I don't have enough information to answer that."


def test_an_answer_whose_every_mark_failed_publishes_none_of_them():
    presented, outcome = present("Invented [S98] and [S99].", "S98", "S99")

    assert not outcome.is_grounded
    assert presented.citations == ()
    assert presented.answer == "Invented and."
    assert "[S" not in presented.answer


def test_text_without_any_mark_is_returned_unchanged():
    presented, _ = present("A plain sentence with no marks at all.", "S1")

    assert presented.answer == "A plain sentence with no marks at all."


# --- the security boundary ----------------------------------------------------


def test_validation_sees_internal_labels_and_only_the_result_is_renumbered():
    """The ordering this whole change had to preserve.

    Citation validation is handed `S`-labels and answers in `S`-labels; the
    renumbering happens afterwards, over what validation already approved. If
    these two steps were ever swapped, validation would be looking up `1` in a
    context that labels its passages `S1`, and nothing would resolve.
    """
    context = many_sources()

    outcome = resolve_citations(context, ["S1", "S3", "S99"])

    # What validation worked with: the internal vocabulary, intact.
    assert [source.label for source in outcome.cited] == ["S1", "S3"]
    assert outcome.unknown_labels == ("S99",)
    assert context.source("S1") is not None
    assert context.source("1") is None, "the context is labelled S1, never 1"

    presented = present_answer("A [S1]. B [S3]. C [S99].", outcome.citations, outcome.cited)

    # Only now do the numbers appear, and only for what survived.
    assert presented.answer == "A [1]. B [2]. C."
    assert len(presented.citations) == 2


def test_presentation_cannot_add_a_source_validation_rejected():
    context = many_sources()
    outcome = resolve_citations(context, ["S99"])

    presented = present_answer("Everything [S99].", outcome.citations, outcome.cited)

    assert presented.citations == ()
    assert "[1]" not in presented.answer


def test_presentation_never_reaches_past_the_citations_it_was_given():
    """Handed an empty verified set, it can publish nothing, whatever the text says."""
    presented = present_answer("A [S1]. B [S2]. C [S3].", (), ())

    assert presented.citations == ()
    assert presented.answer == "A. B. C."
    assert presented.removed_marks == 3
