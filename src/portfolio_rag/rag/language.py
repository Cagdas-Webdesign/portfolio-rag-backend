"""Which language a refusal is written in.

The system has exactly one sentence it says when it cannot answer, and that
sentence is written by the backend rather than by a model — fixed wording is
what makes it predictable for a client and incapable of containing a claim
(:mod:`portfolio_rag.rag.service`). Fixed wording in *one* language, however,
answers a German question in English, which is a visible defect in a bilingual
portfolio: every grounded answer comes back in the language it was asked in,
because the prompt says so, and only the refusal switches to English.

So the sentence is fixed **per language**, and this module decides which one.
Two languages are supported, German and English, because those are the two the
corpus and its audience actually use. Anything else falls back to English —
stating that limit is more honest than pretending to identify fifty languages
with a word list.

**Detection is a heuristic over function words, and is only ever allowed to
choose between two prepared sentences.** It never edits the question, never
reaches a provider, and never changes what is retrieved, embedded or generated:
the worst case is a refusal in the wrong one of two languages. That bounded
consequence is what justifies a word list here instead of a
language-identification dependency.

Why function words rather than characters alone. ``ä``/``ö``/``ü``/``ß`` are a
strong signal, but a perfectly ordinary German question contains none of them
("Was kannst du?"), while an English question about *Düsseldorf* contains one.
The markers below are therefore short, high-frequency function words, and each
list is kept free of words that also exist in the other language — ``die``,
``was``, ``will``, ``hat``, ``man``, ``war``, ``also`` and ``fast`` are all
German words *and* English words, so none of them votes.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Final


class AnswerLanguage(StrEnum):
    """A language the backend can write its own sentences in.

    Deliberately not "the language of the question": this enum exists to select
    a prepared sentence, so it has a member only where a prepared sentence
    exists.
    """

    ENGLISH = "en"
    GERMAN = "de"


#: The one thing this system says when it cannot say anything else, per
#: language. The two sentences state the same limit — no knowledge, therefore
#: no answer — and neither can carry a fact about the question.
INSUFFICIENT_KNOWLEDGE_ANSWERS: Final[dict[AnswerLanguage, str]] = {
    AnswerLanguage.ENGLISH: (
        "I don't have enough information in the available knowledge base to answer that."
    ),
    AnswerLanguage.GERMAN: (
        "Dazu habe ich in der verfügbaren Wissensbasis nicht genug Informationen."
    ),
}

#: Used when the markers below find nothing to go on: a question too short to
#: contain a function word, or one written in a third language.
DEFAULT_ANSWER_LANGUAGE: Final = AnswerLanguage.ENGLISH

#: Letters German has and English does not. Worth more than one marker because
#: they are rare in English text — but not decisive on their own, since a
#: proper noun such as "Düsseldorf" turns up in English questions too.
_GERMAN_LETTERS: Final[frozenset[str]] = frozenset("äöüß")
_GERMAN_LETTER_WEIGHT: Final = 2


def _markers(words: str) -> frozenset[str]:
    """Read one whitespace-separated word list.

    A written-out list rather than a set literal: these are word lists, they
    are meant to be read and edited as prose, and a hundred quoted strings one
    per line is not that.
    """
    return frozenset(words.split())


#: German function words that are not also English words.
_GERMAN_MARKERS: Final[frozenset[str]] = _markers(
    """
    aber auch auf aus bei beim besitzt bin bist damit dann das dass dein deine
    deinen dem den denn der des dich diese diesen dir du durch ein eine einen
    einer eines er es etwas euch für gibt haben hast hatte hier ich ihm ihn ihr
    ihre im ist kann kannst kein keine können machen machst mehr mich mir mit
    nach nicht noch nur oder ohne schon sehr sein seine sich sie sind über uns
    und viele vom von vor wann warum welche welchem welchen welcher welches wem
    wen wer weshalb wie wieso wir wird wo wofür womit wurde würde zum zur
    """
)

#: English function words that are not also German words.
_ENGLISH_MARKERS: Final[frozenset[str]] = _markers(
    """
    about and any anything are at be but can could did do does each for from
    has have how is it its many me more much my not of on or please should
    some tell than that the their there these they this those to used using
    what when where which who whose why with would you your
    """
)

#: Word-shaped runs, letters and digits only. Punctuation and symbols carry no
#: vote either way, so they are simply not tokens.
_WORD: Final = re.compile(r"\w+", re.UNICODE)


def detect_language(text: str) -> AnswerLanguage:
    """Decide which prepared sentence *text* should be answered with.

    Counts the German and English markers in the text, gives each word carrying
    a German-only letter extra weight, and picks the higher score. Ties and a
    complete absence of markers both fall to :data:`DEFAULT_ANSWER_LANGUAGE`:
    "no evidence" and "evidence for both" are the same situation, and neither
    justifies leaving the default.
    """
    # ``lower`` rather than ``casefold``: casefolding maps "ß" to "ss" and
    # would erase one of the two German-only letters this counts.
    words = _WORD.findall(unicodedata.normalize("NFC", text).lower())

    german_score = sum(1 for word in words if word in _GERMAN_MARKERS)
    german_score += _GERMAN_LETTER_WEIGHT * sum(
        1 for word in words if not _GERMAN_LETTERS.isdisjoint(word)
    )
    english_score = sum(1 for word in words if word in _ENGLISH_MARKERS)

    return AnswerLanguage.GERMAN if german_score > english_score else DEFAULT_ANSWER_LANGUAGE


def insufficient_knowledge_answer(language: AnswerLanguage) -> str:
    """The refusal sentence for *language*."""
    return INSUFFICIENT_KNOWLEDGE_ANSWERS[language]
