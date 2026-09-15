"""Which prepared sentence a question gets, and what those sentences say."""

from __future__ import annotations

import unicodedata

import pytest

from portfolio_rag.rag.language import (
    DEFAULT_ANSWER_LANGUAGE,
    INSUFFICIENT_KNOWLEDGE_ANSWERS,
    AnswerLanguage,
    detect_language,
    insufficient_knowledge_answer,
)

GERMAN_QUESTIONS = [
    "Welche medizinischen Zertifizierungen besitzt du?",
    "Was kannst du mir über das Backend erzählen?",
    "Wie kann man ihn kontaktieren?",
    "Mit welchen Datenbanken arbeitet er?",
    "Hat er Erfahrung mit WordPress?",
    "Wofür nutzt er ElevenLabs?",
    "Warum ist der Similarity Threshold nicht die Sicherheitsgrenze?",
    "Kann er serverseitige Logik betreiben, ohne einen Server zu mieten?",
]

ENGLISH_QUESTIONS = [
    "Which medical certifications do you have?",
    "What is the maintainer's favourite pizza?",
    "Which HTTP framework does the service use?",
    "How are documents stored?",
    "Tell me about the backend.",
    "Why is the similarity threshold not the security boundary?",
    "Can he run server-side logic without renting a server?",
]


# --- detection ---------------------------------------------------------------


@pytest.mark.parametrize("question", GERMAN_QUESTIONS)
def test_a_german_question_selects_german(question: str):
    assert detect_language(question) is AnswerLanguage.GERMAN


@pytest.mark.parametrize("question", ENGLISH_QUESTIONS)
def test_an_english_question_selects_english(question: str):
    assert detect_language(question) is AnswerLanguage.ENGLISH


def test_umlauts_are_a_signal_even_without_a_function_word():
    assert detect_language("Zertifizierungen für Qualitätsmanagement") is AnswerLanguage.GERMAN


def test_a_german_proper_noun_does_not_make_an_english_question_german():
    """One umlaut is a signal, not a verdict — the English markers outvote it."""
    assert detect_language("Which projects did he build in Düsseldorf?") is AnswerLanguage.ENGLISH


@pytest.mark.parametrize(
    "question",
    [
        pytest.param("", id="empty"),
        pytest.param("Zertifizierungen?", id="one word, no marker"),
        pytest.param("42", id="digits"),
        pytest.param("¿Qué certificaciones tienes?", id="a third language"),
        pytest.param("!!! ??? ...", id="punctuation only"),
    ],
)
def test_anything_undecidable_falls_back_to_the_default(question: str):
    assert detect_language(question) is DEFAULT_ANSWER_LANGUAGE
    assert DEFAULT_ANSWER_LANGUAGE is AnswerLanguage.ENGLISH


def test_shouting_is_still_german():
    assert detect_language("WELCHE ZERTIFIZIERUNGEN BESITZT DU?") is AnswerLanguage.GERMAN


def test_a_decomposed_umlaut_counts_like_a_composed_one():
    """Two keyboards, one question: u plus a combining diaeresis is still an umlaut."""
    composed = "Wofür nutzt er das?"
    decomposed = unicodedata.normalize("NFD", composed)

    assert decomposed != composed
    assert detect_language(decomposed) is AnswerLanguage.GERMAN


def test_detection_is_deterministic():
    question = "Welche medizinischen Zertifizierungen besitzt du?"

    assert detect_language(question) is detect_language(question)


# --- the sentences -----------------------------------------------------------


def test_every_language_has_a_prepared_sentence():
    assert set(INSUFFICIENT_KNOWLEDGE_ANSWERS) == set(AnswerLanguage)


@pytest.mark.parametrize("language", list(AnswerLanguage))
def test_a_prepared_sentence_states_the_limit_and_nothing_else(language: AnswerLanguage):
    sentence = insufficient_knowledge_answer(language)

    assert sentence.strip() == sentence
    assert sentence.endswith(".")
    assert len(sentence.split()) < 25


def test_the_two_sentences_are_actually_different_wording():
    english = insufficient_knowledge_answer(AnswerLanguage.ENGLISH)
    german = insufficient_knowledge_answer(AnswerLanguage.GERMAN)

    assert english != german
    assert "knowledge base" in english
    assert "Wissensbasis" in german


@pytest.mark.parametrize("language", list(AnswerLanguage))
def test_a_refusal_is_answered_in_its_own_language(language: AnswerLanguage):
    """Each prepared refusal contains enough signal to select its own language."""
    assert detect_language(insufficient_knowledge_answer(language)) is language
