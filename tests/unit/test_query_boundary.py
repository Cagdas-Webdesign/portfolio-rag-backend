"""The query input boundary: what a question may be, and what it becomes."""

from __future__ import annotations

import pytest

from portfolio_rag.api.schemas.chat import MAX_MESSAGE_LENGTH
from portfolio_rag.rag.errors import QueryValidationError
from portfolio_rag.rag.query import (
    MAX_QUERY_LENGTH,
    QUERY_REPRESENTATION_VERSION,
    build_query_embedding_text,
    normalize_query,
)


def test_a_plain_question_survives_unchanged():
    query = normalize_query("Which HTTP framework does the service use?")

    assert query.text == "Which HTTP framework does the service use?"


@pytest.mark.parametrize("raw", ["", " ", "\n", "\t\t", "   \n  \r\n "])
def test_a_question_with_nothing_in_it_is_refused(raw: str):
    with pytest.raises(QueryValidationError):
        normalize_query(raw)


def test_the_longest_allowed_question_is_accepted():
    query = normalize_query("y" * MAX_QUERY_LENGTH)

    assert len(query.text) == MAX_QUERY_LENGTH


def test_one_character_more_is_refused():
    with pytest.raises(QueryValidationError) as caught:
        normalize_query("y" * (MAX_QUERY_LENGTH + 1))

    assert str(MAX_QUERY_LENGTH) in caught.value.message


def test_the_public_limit_can_only_ever_be_the_stricter_of_the_two():
    """The HTTP bound is tighter than the pipeline's, and must stay that way.

    These bound different things: `MAX_QUERY_LENGTH` is what the pipeline can
    process at all — the CLI and the evaluation harness go through it too —
    while `MAX_MESSAGE_LENGTH` is what an anonymous web page may send. Two
    numbers are only safe in the one direction asserted here; the other way
    round, HTTP would accept questions the pipeline then refuses, and the
    public endpoint would be paying for inputs it cannot use.
    """
    assert MAX_MESSAGE_LENGTH <= MAX_QUERY_LENGTH
    assert MAX_MESSAGE_LENGTH == 2000, "the public bound is a deliberate number, not a default"


def test_the_rejection_never_repeats_the_question():
    """A rejected question is untrusted input and must not be echoed anywhere."""
    secret = "sk-live-not-a-real-token"  # noqa: S105 - a marker, not a credential

    with pytest.raises(QueryValidationError) as caught:
        normalize_query(f"{secret} {'y' * MAX_QUERY_LENGTH}")

    assert secret not in caught.value.message


def test_whitespace_is_collapsed_so_padding_cannot_change_the_embedding():
    padded = normalize_query("  Which   HTTP\n\nframework\tis used?  ")
    plain = normalize_query("Which HTTP framework is used?")

    assert padded.text == plain.text


def test_unicode_is_normalized_to_a_single_composition():
    composed = normalize_query("Wie funktioniert die Prüfung?")
    decomposed = normalize_query("Wie funktioniert die Prüfung?")

    assert composed.text == decomposed.text
    assert composed.text == "Wie funktioniert die Prüfung?"


@pytest.mark.parametrize(
    "raw",
    [
        "Welche Technologien verwendest du?",
        "サービスは何を使っていますか?",
        "Что использует этот сервис?",
        "¿Qué framework se usa?",
        "emoji are text too 🙂",
    ],
)
def test_unicode_questions_are_accepted(raw: str):
    assert normalize_query(raw).text == raw


def test_case_punctuation_and_wording_are_left_alone():
    """Normalization is not the place to decide a question meant something else."""
    raw = "FastAPI, Uvicorn — WHICH one serves HTTP?!"

    assert normalize_query(raw).text == raw


def test_the_original_length_is_kept_for_diagnostics():
    query = normalize_query("  padded  question  ")

    assert query.original_length == 20
    assert query.text == "padded question"


def test_the_representation_is_the_question_and_nothing_else():
    """No `search query:` prefix, no keyword stuffing, no expansion."""
    query = normalize_query("Which database is used?")

    assert build_query_embedding_text(query) == "Which database is used?"


def test_the_representation_is_deterministic():
    first = build_query_embedding_text(normalize_query(" Which  database is used? "))
    second = build_query_embedding_text(normalize_query("Which database is used?"))

    assert first == second


def test_the_representation_is_versioned():
    assert QUERY_REPRESENTATION_VERSION == "query-text-v1"


@pytest.mark.parametrize(
    "raw",
    [
        "​",  # zero-width space — not matched by `\s`
        "​‌‍",  # zero-width space, non-joiner, joiner
        "﻿",  # byte order mark
        "   ",  # non-breaking spaces
        "‎‏",  # bidi marks
        "\x00\x01",  # control characters
    ],
)
def test_a_question_made_only_of_invisible_characters_is_empty(raw: str):
    """Trimming is not enough: a zero-width space survives `.strip()`.

    Without this, a message of pure invisibles is a valid one-character
    question, and every one of them costs an embedding call.
    """
    with pytest.raises(QueryValidationError):
        normalize_query(raw)


def test_invisible_characters_inside_a_real_question_are_left_alone():
    """A joiner is meaningful inside a word and inside an emoji sequence."""
    raw = "Welche​Technologien? 👨‍👩‍👧"

    assert normalize_query(raw).text == raw
