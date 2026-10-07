"""Turning verified citations into something a reader wants to look at.

``S1``, ``S7``, ``S10`` are how the context labels its passages and how a model
points back at them. They are load-bearing: the prompt asks for them, the
parser reads them, and citation validation decides what is publishable by
looking them up. What they are not is readable. A portfolio visitor should see

    Das Backend nutzt FastAPI [1] und Cloudflare Vectorize [2].

and never learn that the backend calls those passages S4 and S7.

**This runs last, and only on an answer that already passed.** It takes a
validated answer and renumbers it; it cannot admit a source, and it is never
consulted about whether one is admissible. The order is
generation → parsing → citation validation → *this*, and nothing here can be
reached on a path that produced no citations — a refusal has fixed wording and
an empty citation list, and both stay exactly as they were.

Three rules, and all three exist to keep the numbers honest:

* **A number names a published citation, not a label.** Validation collapses
  several passages from one document and section into one citation, so two
  labels can legitimately be the same source — and then they are the same
  number.
* **Numbering follows the answer text**, not the order the model happened to
  list its sources in, because the reader meets the marks in the text. The
  citation list is reordered to match, so ``[2]`` is always the second entry.
* **A mark the backend could not verify is removed, never renumbered.** Turning
  an unknown ``[S9]`` into ``[3]`` would be inventing a citation at the last
  possible moment, which is precisely what the rest of this package exists to
  prevent.

A label a model wrote as a *name* — ``S1: Ausbildung``, ``(S2)`` — is not a
mark at all and is removed outright; the reader was never meant to see it.

A sentence that only talks *about* labels — ``Diese Schritte werden in den
Quellen S2 (…) und S1 (…) beschrieben.`` — is removed whole (v1.2.1). It tells
the reader nothing the citation list does not, and with its labels cut out it
would no longer be a sentence. Removing text can never admit a source; if it
would leave nothing at all, only the labels go and the sentence stays.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from portfolio_rag.domain.retrieval import SourceCitation
from portfolio_rag.rag.context import ContextSource

#: An internal source mark as the context writes it and a model repeats it.
#: ``\d+`` is greedy inside the brackets, which is what keeps ``[S1]`` from
#: matching the start of ``[S10]``; the brackets do the rest. Leading blanks
#: are captured so that removing a mark does not leave a double space behind —
#: spaces and tabs only, never a newline, so no line is joined to the one above.
_MARK: Final = re.compile(r"([ \t]*)\[(S\d+)\]")

#: A label a model used as a *name* rather than a mark: ``S1: Ausbildung``,
#: ``- S2 - Skills``, ``(S3) Details``. The prompt shows passages under
#: ``[SOURCE S1]``, and asked what it knows, a model lists them that way.
#: These are never citations — only ``[S1]`` is — so they are removed, not
#: renumbered. Deliberately narrow, so ``Galaxy S23: …`` or ``S3-Bucket``
#: survive: the label must open a line (after an optional list marker) or
#: follow punctuation, and a bare label must be followed by ``:``, a dash
#: with a blank after it, or an en/em dash. The kept ``lead`` preserves list
#: markers and sentence punctuation.
_LABEL_AS_NAME: Final = re.compile(
    r"(?P<lead>^[ \t]*(?:(?:[-*+•]|\d+[.)])[ \t]+)?|[.,;:!?][ \t]+)"
    r"(?:\(S\d+\)|S\d+(?=[ \t]*(?::(?!\d)|[\u2013\u2014]|-[ \t])))"
    r"[ \t]*(?::(?!\d)|[\u2013\u2014]|-(?=[ \t]))?[ \t]*",
    re.MULTILINE,
)

#: A parenthesised label anywhere else — ``FastAPI (S1)`` or ``(S1, S2)``.
#: Parentheses are not the citation syntax, so there is nothing to renumber.
#: Not after a ``/``, which is a URL path rather than prose.
_PARENTHESISED_LABELS: Final = re.compile(r"[ \t]*(?<!/)\(S\d+(?:[ \t]*,[ \t]*S\d+)*\)")

#: Several labels in one bracket — ``[S2, S3]``. Split into single marks so
#: that each is renumbered or removed exactly like ``[S2] [S3]``.
_MARK_GROUP: Final = re.compile(r"\[(S\d+(?:[ \t]*[,;][ \t]*S\d+)+)\]")
_GROUP_SEPARATOR: Final = re.compile(r"[ \t]*[,;][ \t]*")

#: A bare label as a word: not inside another word, a path or a version number,
#: so ``SS1``, ``S1x``, ``S3-Bucket``, ``/S1`` and ``S1.2`` are not labels. Not
#: inside brackets or parentheses either — those are marks, handled elsewhere.
_BARE_LABEL: Final = r"(?<![\w/\-\[(])S\d+(?![\w\-\])]|[.:]\d)"
#: A bare label with an optional description: ``S2 (Setup)``.
_DESCRIBED_LABEL: Final = rf"{_BARE_LABEL}(?:[ \t]*\([^()\n]*\))?"
_LIST_JOIN: Final = r"[ \t]*(?:,|;|&|\bund\b|\bsowie\b|\boder\b|\band\b|\bor\b)[ \t]*"

#: Labels written as a reference in prose, the way a model talks *about* its
#: context: a source word followed by labels — ``Quellen S1, S2``, ``Quelle:
#: S3``, ``sources S1 and S2`` — or a list of two or more bare labels —
#: ``S2, S3 und S5``. A single bare label without a source word is left alone,
#: because ``Audi S3`` and ``Stufe S2`` are ordinary prose.
_LABEL_REFERENCE: Final = re.compile(
    rf"(?:\b(?:Quellen?|Quellenangaben?|[Ss]ources?|SOURCES?)[ \t]*:?[ \t]*{_DESCRIBED_LABEL}"
    rf"(?:{_LIST_JOIN}{_DESCRIBED_LABEL})*"
    rf"|{_DESCRIBED_LABEL}(?:{_LIST_JOIN}{_DESCRIBED_LABEL})+)"
)
_SOURCE_WORD: Final = re.compile(r"\b(?:Quellen?|Quellenangaben?|[Ss]ources?|SOURCES?)\b")

#: A sentence: ends at ``.``, ``!`` or ``?`` followed by a blank or the end of
#: the line — so ``3.12`` and ``z.B.`` inside a sentence do not end it.
_SENTENCE: Final = re.compile(r"(?:[^.!?\n]|[.!?]+(?![ \t]|$))+(?:[.!?]+|$)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class PresentedAnswer:
    """An answer as a client should see it, with its citations in that order."""

    answer: str
    citations: tuple[SourceCitation, ...]
    removed_marks: int
    """Marks dropped because no verified citation stood behind them.

    Diagnostics. Non-zero means a model referred in its prose to a label it did
    not declare, or declared one that validation rejected — worth counting,
    never worth publishing.
    """


def present_answer(
    answer: str, citations: Sequence[SourceCitation], cited: Sequence[ContextSource]
) -> PresentedAnswer:
    """Renumber *answer*'s source marks over the already-verified *citations*.

    *cited* is the audit trail validation produced: the context sources the
    model named, still carrying their labels. It is what connects a mark in the
    text to a published citation — this function reads it and adds nothing to
    it.

    An answer with no citations is returned untouched apart from having any
    stray marks removed: there is nothing to number against, and a refusal has
    no marks to begin with.
    """
    citation_of_label = _labels_to_citations(citations, cited)
    order: list[SourceCitation] = []
    removed = 0

    def renumber(match: re.Match[str]) -> str:
        nonlocal removed
        blanks, label = match.group(1), match.group(2)
        citation = citation_of_label.get(label)
        if citation is None:
            removed += 1
            return ""
        if citation not in order:
            order.append(citation)
        return f"{blanks}[{order.index(citation) + 1}]"

    presented = _MARK_GROUP.sub(
        lambda match: " ".join(f"[{label}]" for label in _GROUP_SEPARATOR.split(match.group(1))),
        answer,
    )
    presented = _drop_label_references(presented)
    presented = _MARK.sub(renumber, presented)
    presented = _LABEL_AS_NAME.sub(r"\g<lead>", presented)
    presented = _PARENTHESISED_LABELS.sub("", presented)

    # A verified citation the prose never marked still belongs in the list —
    # validation admitted it, and this step does not get to overrule that. It
    # goes after the marked ones, keeping the numbers the reader can see dense
    # and starting at one.
    order.extend(citation for citation in citations if citation not in order)

    return PresentedAnswer(answer=presented, citations=tuple(order), removed_marks=removed)


def _labels_to_citations(
    citations: Sequence[SourceCitation], cited: Sequence[ContextSource]
) -> dict[str, SourceCitation]:
    """Map each verified label onto the citation that was published for it.

    Several labels can arrive at the same citation. That is not a collision to
    resolve: validation already decided those passages are one source, and the
    reader should see one number for it.
    """
    published = {(citation.document_id, citation.section): citation for citation in citations}
    mapping: dict[str, SourceCitation] = {}
    for source in cited:
        citation = source.retrieved.citation()
        match = published.get((citation.document_id, citation.section))
        if match is not None:
            mapping[source.label] = match
    return mapping


def _drop_label_references(answer: str) -> str:
    """Remove every sentence that refers to the context by its labels.

    Runs before renumbering, so a mark in a removed sentence is never numbered
    and the numbers the reader sees stay dense. A line whose sentences are all
    removed goes with them. Text without such a reference is returned as is.
    """
    if not _LABEL_REFERENCE.search(answer):
        return answer
    lines: list[str] = []
    for line in answer.split("\n"):
        if not _LABEL_REFERENCE.search(line):
            lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        kept = [
            sentence.group(0).strip()
            for sentence in _SENTENCE.finditer(line)
            if sentence.group(0).strip() and not _LABEL_REFERENCE.search(sentence.group(0))
        ]
        if kept:
            lines.append(indent + " ".join(kept))
    presented = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
    if presented.strip():
        return presented
    # Every sentence was a reference: keep the prose, lose only the labels.
    stripped = _LABEL_REFERENCE.sub(_strip_labels, answer)
    stripped = re.sub(r"[ \t]+(?=[.,;:!?])", "", stripped)
    return re.sub(r"(?<=\S)[ \t]{2,}", " ", stripped)


def _strip_labels(match: re.Match[str]) -> str:
    """``in den Quellen S1 und S2`` → ``in den Quellen``; a bare list → nothing."""
    word = _SOURCE_WORD.match(match.group(0))
    return word.group(0) if word else ""
