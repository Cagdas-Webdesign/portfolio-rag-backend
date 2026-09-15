"""What a portfolio visitor actually receives, end to end.

The unit tests in `tests/unit/test_citation_presentation.py` pin the mapping
itself. This file checks the thing that broke: that the renumbering is wired
into the real pipeline in the right place — after a real generation, a real
parse and real citation validation — and that what leaves the HTTP boundary
carries no internal label.

Nothing here is mocked except the language model, because a real one cannot be
asked to cite a specific label on demand.
"""

from __future__ import annotations

import pytest

from portfolio_rag.rag.policy import RetrievalPolicy
from portfolio_rag.rag.service import AnswerOutcome
from tests.doubles import ScriptedLLMProvider, ScriptedReply, grounded
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run

QUESTION = "Which framework serves the API and how are documents stored?"


def answer_with(*replies: ScriptedReply):
    service, _, _ = build_pipeline(llm=ScriptedLLMProvider(*replies))
    return run(service.answer(QUESTION))


def test_a_visitor_never_sees_an_internal_source_label():
    """The complaint this change exists for."""
    answer = answer_with(
        grounded("FastAPI serves the API [S1] and documents are Markdown [S2].", "S1", "S2")
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "FastAPI serves the API [1] and documents are Markdown [2]."
    assert "[S" not in answer.answer


def test_the_citation_list_follows_the_order_of_the_text():
    """Swap the marks in the prose and the list swaps with them.

    Written without naming which document sits behind which label: what is
    being checked is that the list tracks the text, not the retrieval order
    that happened to assign the labels.
    """
    forward = answer_with(grounded("First [S1], second [S2].", "S1", "S2"))
    reversed_ = answer_with(grounded("First [S2], second [S1].", "S1", "S2"))

    assert forward.answer == "First [1], second [2]."
    assert reversed_.answer == "First [1], second [2]."
    assert [c.document_id for c in reversed_.citations] == [
        c.document_id for c in reversed(forward.citations)
    ]


def test_a_source_cited_twice_is_one_number_and_one_entry():
    answer = answer_with(grounded("FastAPI [S1] serves it, and FastAPI [S1] documents it.", "S1"))

    assert answer.answer == "FastAPI [1] serves it, and FastAPI [1] documents it."
    assert len(answer.citations) == 1


def test_an_unverifiable_mark_is_dropped_from_the_published_answer():
    """The model marked a label the backend could not prove. It does not ship."""
    answer = answer_with(
        grounded("FastAPI serves the API [S1], and something else [S99].", "S1", "S99")
    )

    assert answer.answer == "FastAPI serves the API [1], and something else."
    assert answer.unknown_labels == ("S99",), "the internal label is still what gets counted"
    assert len(answer.citations) == 1


def test_validation_still_rejects_an_answer_whose_every_source_is_invented():
    """Presentation runs after this decision and cannot reach it."""
    answer = answer_with(grounded("Everything is fine [S99].", "S99"))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.citations == ()
    assert "[S99]" not in answer.answer
    assert "[1]" not in answer.answer


def test_a_refusal_keeps_its_wording_and_its_empty_citation_list():
    answer = answer_with(ScriptedReply(answer="", sources=()))

    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.citations == ()
    assert "[1]" not in answer.answer


def test_a_question_the_corpus_cannot_answer_publishes_no_numbers():
    """Retrieval short-circuits, so presentation is never even reached."""
    service, _, llm = build_pipeline(
        llm=ScriptedLLMProvider(grounded("Irrelevant [S1].", "S1")),
        retrieval_policy=RetrievalPolicy(min_similarity=0.99),
    )

    answer = run(service.answer("What is the best pizza in Naples?"))

    assert answer.outcome is AnswerOutcome.NO_KNOWLEDGE
    assert answer.citations == ()
    assert "[1]" not in answer.answer
    assert llm.call_count == 0


def test_the_model_is_still_prompted_and_parsed_in_internal_labels():
    """The renumbering must not have leaked backwards into the prompt.

    If it had, the context would be labelled `1` while the prompt still asked
    for `S1`, and every citation would resolve to nothing.
    """
    service, _, llm = build_pipeline(
        llm=ScriptedLLMProvider(grounded("Both [S1] [S2].", "S1", "S2"))
    )

    run(service.answer(QUESTION))

    context_message = "\n".join(message.content for message in llm.requests[0].messages)
    assert "[SOURCE S1]" in context_message
    assert "[SOURCE 1]" not in context_message


@pytest.mark.parametrize(
    ("text", "labels", "expected"),
    [
        ("A [S1]. B [S2].", ("S1", "S2"), "A [1]. B [2]."),
        ("A [S2]. B [S1].", ("S1", "S2"), "A [1]. B [2]."),
        ("Only [S2].", ("S2",), "Only [1]."),
    ],
)
def test_the_numbering_is_dense_whatever_the_labels_were(
    text: str, labels: tuple[str, ...], expected: str
):
    answer = answer_with(grounded(text, *labels))

    assert answer.answer == expected
