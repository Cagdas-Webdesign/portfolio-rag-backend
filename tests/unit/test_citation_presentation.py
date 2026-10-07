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
from portfolio_rag.evaluation.e2e import INTERNAL_LABEL
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


# --- labels used as names, not marks (v1.0.1) ---------------------------------
#
# Asked what it knows, a model lists the passages the way the context shows
# them: `S1: Ausbildung`. Only `[S1]` is citation syntax; every other form is a
# label written as a name, and the reader must never see it.


def test_a_bracketed_mark_is_still_renumbered_as_before():
    """Case A: the existing citation path is untouched."""
    presented, _ = present("FastAPI und Python [S1].", "S1")

    assert presented.answer == "FastAPI und Python [1]."
    assert len(presented.citations) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("S1: Informationen zu seiner Ausbildung.", "Informationen zu seiner Ausbildung."),
        ("S1 : Informationen zu seiner Ausbildung.", "Informationen zu seiner Ausbildung."),
        ("(S1) Informationen zu seiner Ausbildung.", "Informationen zu seiner Ausbildung."),
        ("(S1): Informationen zu seiner Ausbildung.", "Informationen zu seiner Ausbildung."),
        ("S1 - Informationen zu seiner Ausbildung.", "Informationen zu seiner Ausbildung."),
        ("S2 \u2013 Weitere Informationen.", "Weitere Informationen."),
        ("S10 \u2014 Weitere Informationen.", "Weitere Informationen."),
        ("S23: Weitere Informationen.", "Weitere Informationen."),
    ],
)
def test_a_label_opening_a_line_is_removed(text: str, expected: str):
    """Cases B to E."""
    presented, _ = present(text, "S1")

    assert presented.answer == expected


def test_labels_on_consecutive_lines_are_all_removed():
    """Case E."""
    presented, _ = present(
        "S1 - Informationen zu A.\nS2 \u2013 Weitere Informationen.\nS10 \u2014 Noch mehr.", "S1"
    )

    assert presented.answer == "Informationen zu A.\nWeitere Informationen.\nNoch mehr."


def test_a_bullet_list_keeps_its_bullets_and_loses_its_labels():
    """Case F."""
    presented, _ = present("Die Wissensbasis:\n- S1: Ausbildung\n- S2: Skills\n- S3: RAG", "S1")

    assert presented.answer == "Die Wissensbasis:\n- Ausbildung\n- Skills\n- RAG"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("* S1: A\n* S2: B", "* A\n* B"),
        ("1. S1: A\n2. S2: B", "1. A\n2. B"),
        ("1) S1 \u2013 A\n2) S2 \u2013 B", "1) A\n2) B"),
        ("  - S1: eingerückt", "  - eingerückt"),
    ],
)
def test_other_list_markers_are_kept_too(text: str, expected: str):
    presented, _ = present(text, "S1")

    assert presented.answer == expected


def test_labels_inside_a_paragraph_after_punctuation_are_removed():
    presented, _ = present("Die Wissensbasis umfasst: S1: Studium. S2: Skills, S3: RAG.", "S1")

    assert presented.answer == "Die Wissensbasis umfasst: Studium. Skills, RAG."


def test_parenthesised_labels_in_running_text_are_removed():
    presented, _ = present("FastAPI wird genutzt (S1), ebenso Vectorize (S2, S3).", "S1")

    assert presented.answer == "FastAPI wird genutzt, ebenso Vectorize."


def test_a_label_name_and_a_citation_mark_on_one_line():
    """The name goes, the mark is renumbered — neither path disturbs the other."""
    presented, _ = present("- S1: Ausbildung [S1]\n- S2: Skills [S2]", "S1", "S2")

    assert presented.answer == "- Ausbildung [1]\n- Skills [2]"
    assert len(presented.citations) == 2


def test_removing_label_names_does_not_touch_the_citations():
    """Case H: the verified list is exactly what validation produced."""
    presented, outcome = present("S1: Ausbildung.\nS2: Skills.", "S1", "S2")

    assert presented.citations == outcome.citations
    assert [citation.title for citation in presented.citations] == ["Doc 1", "Doc 2"]
    assert presented.removed_marks == 0, "a label name is not a failed citation mark"


def test_plain_text_without_labels_is_unchanged():
    """Case G."""
    text = "Er hat Informatik studiert.\n\n- Python\n- FastAPI: ein Web-Framework\n1. Erstens"
    presented, _ = present(text, "S1")

    assert presented.answer == text


# --- what must not be mistaken for a label ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Ein Samsung Galaxy S23: gutes Telefon.",
        "AWS S3: Objektspeicher.",
        "Audi S3 - sportlich.",
        "Dateien liegen im S3-Bucket.",
        "S3-Bucket für Artefakte.",
        "Python 3.12, v1.0.0 und Release S1.2.",
        "Siehe https://example.com/S1:abc und https://example.com/(S1).",
        "x = S1 - S2",
        'print("S1: x")',
        "Treffen um S1:30 ist kein Label.",
        "SS1: kein Label, S1x: auch nicht.",
        "Die Serie S10 ist ein Produkt.",
        "Stufe S2 bezeichnet eine Stufe.",
    ],
)
def test_legitimate_text_is_not_mistaken_for_a_label(text: str):
    presented, _ = present(text, "S1")

    assert presented.answer == text


# --- labels referred to in prose (v1.2.1) -------------------------------------
#
# A model that talks *about* its context — `Quellen S2, S1 und S3`, `[S2, S3]`,
# `S2, S3 und S5` — writes forms the earlier rules did not know. None of them
# may reach the reader; the citation list is not touched by any of this.


def test_unverified_marks_in_running_text_leave_clean_prose():
    presented, _ = present("Er nutzt FastAPI [S2] und Vectorize [S3].")

    assert presented.answer == "Er nutzt FastAPI und Vectorize."


def test_a_trailing_row_of_unverified_marks_is_removed():
    presented, _ = present("Antworttext. [S2] [S3] [S5]")

    assert presented.answer == "Antworttext."


def test_verified_marks_in_a_row_become_numbers_not_labels():
    presented, _ = present("Antworttext. [S2] [S3] [S5]", "S2", "S3", "S5")

    assert presented.answer == "Antworttext. [1] [2] [3]"
    assert len(presented.citations) == 3


def test_several_labels_in_one_bracket_are_handled_one_by_one():
    presented, _ = present("FastAPI und Vectorize [S2, S3].", "S2")

    assert presented.answer == "FastAPI und Vectorize [1]."
    assert presented.removed_marks == 1


def test_a_sentence_that_only_names_sources_is_removed_whole():
    presented, _ = present(
        "Er richtet den Dienst ein und testet ihn. Diese Schritte werden in den Quellen "
        "S2 (Einrichtung), S1 (Deployment) und S3 (Tests) beschrieben."
    )

    assert presented.answer == "Er richtet den Dienst ein und testet ihn."


@pytest.mark.parametrize(
    "sentence",
    [
        "Das steht in S2, S3 und S5.",
        "Quellen S1, S2, S3.",
        "Quellen: S1, S2.",
        "See sources S1 and S2.",
    ],
)
def test_reference_sentences_are_removed_and_the_rest_stays(sentence: str):
    presented, _ = present(f"Erster Satz.\n\n{sentence}\n\nLetzter Satz.")

    assert presented.answer == "Erster Satz.\n\nLetzter Satz."


def test_an_answer_made_only_of_a_reference_keeps_its_prose():
    """Removing everything would publish an empty answer; only the labels go."""
    presented, _ = present("Die Schritte sind in den Quellen S1 und S2 beschrieben.")

    assert presented.answer == "Die Schritte sind in den Quellen beschrieben."


def test_a_two_digit_bracketed_label_never_survives():
    presented, _ = present("Quelle [S10] bestätigt dies.")

    assert "[S10]" not in presented.answer
    assert presented.answer == "Quelle bestätigt dies."


def test_removing_reference_sentences_does_not_touch_the_citations():
    presented, outcome = present("FastAPI [S1]. Siehe Quellen S1 und S2.", "S1", "S2")

    assert presented.answer == "FastAPI [1]."
    assert presented.citations == outcome.citations


@pytest.mark.parametrize(
    "text",
    [
        "Er nutzt FastAPI, Python 3.12 und Cloudflare Vectorize.",
        "Audi S3 und Stufe S2 sind keine Quellen.",
        "Das Modell S10 ist ein Produkt. Version S1.2 folgt.",
        "Die Quellen liegen im S3-Bucket.",
    ],
)
def test_text_without_internal_labels_is_unchanged_byte_for_byte(text: str):
    presented, _ = present(text, "S1")

    assert presented.answer == text


@pytest.mark.parametrize(
    "leak",
    [
        "Antwort [S2, S3].",
        "Diese Schritte werden in den Quellen S2 (A), S1 (B) beschrieben.",
        "Das steht in S2, S3 und S5.",
        "Quellen: S1",
    ],
)
def test_the_leak_gate_catches_every_form_presentation_removes(leak: str):
    assert INTERNAL_LABEL.search(leak)
    presented, _ = present(f"Satz. {leak}")
    assert not INTERNAL_LABEL.search(presented.answer)


@pytest.mark.parametrize("text", ["Audi S3 und Stufe S2.", "Release S1.2, S3-Bucket."])
def test_the_leak_gate_does_not_flag_ordinary_prose(text: str):
    assert not INTERNAL_LABEL.search(text)
