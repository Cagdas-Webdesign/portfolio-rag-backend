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

    presented = _MARK.sub(renumber, answer)
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
