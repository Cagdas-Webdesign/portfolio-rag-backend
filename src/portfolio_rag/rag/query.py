"""The query input boundary, and what a question looks like to a model.

Two jobs, both small and both deliberate.

**Bounding the input.** This is the first place in the system that handles text
a stranger wrote. A question must be non-empty once whitespace stops counting,
and short enough that no single request can push an unbounded payload into an
embedding provider. :data:`MAX_QUERY_LENGTH` is the one number that says so;
the HTTP schema imports it rather than declaring a second one, because a limit
that exists twice eventually exists at two different values.

**Deciding what gets embedded.** Version 1 is the normalized question and
nothing else: no ``"search query:"`` prefix, no keyword stuffing, no rewriting,
no expansion. Those are all plausible, all cheap to add, and none of them has
been measured on this corpus — inventing retrieval tricks before there is an
evaluation to judge them is how a pipeline acquires behaviour nobody can
explain. Version 2 is where evidence goes.

The query representation is **not** the corpus representation. A chunk is
embedded under its document title and heading path
(:mod:`portfolio_rag.ingestion.embedding`) because a passage out of context is
often unintelligible; a question already is what it is. Manufacturing a fake
``KnowledgeChunk`` to reuse the corpus path would couple two things that only
happen to share a provider.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

from portfolio_rag.rag.errors import QueryValidationError

#: Upper bound on one question, in characters. Bounded input is the cheapest
#: defence against both accidental and deliberate oversized payloads, and it is
#: enforced identically at the HTTP boundary and here.
MAX_QUERY_LENGTH: Final = 4000

#: Identifies how a question is turned into embeddable text. Changing what this
#: module composes means changing this string too.
QUERY_REPRESENTATION_VERSION: Final = "query-text-v1"

_WHITESPACE_RUN: Final = re.compile(r"\s+")

#: Unicode categories that carry no visible content: control characters, format
#: characters (zero-width space, joiners, bidi marks, the BOM) and separators.
#: A question made *only* of these is empty however many code points it has —
#: ``\s`` does not match a zero-width space, so trimming alone would let one
#: through as a one-character question and buy an embedding call for nothing.
#:
#: This decides emptiness; it never edits the text. Zero-width joiners are
#: meaningful inside real words and inside emoji sequences, and a question that
#: contains anything else keeps every character it arrived with.
_INVISIBLE_CATEGORIES: Final[frozenset[str]] = frozenset({"Cc", "Cf", "Zs", "Zl", "Zp"})


@dataclass(frozen=True, slots=True)
class UserQuery:
    """A question that passed the input boundary.

    Constructing one anywhere but through :func:`normalize_query` defeats the
    point of having a boundary.
    """

    text: str
    """The normalized question: NFC, trimmed, internal whitespace collapsed."""

    original_length: int
    """Characters as received. Kept for diagnostics — counts may be logged, the
    text may not."""


def normalize_query(raw: str) -> UserQuery:
    """Validate and normalize a user's question.

    Normalization is deliberately shallow: Unicode NFC so the same question
    typed on two keyboards is the same question, and whitespace collapsed so
    that padding cannot change what gets embedded. Case, punctuation, accents
    and wording are left exactly as asked — a question is prose, and this is not
    the place to decide it meant something else.

    Raises :class:`~portfolio_rag.rag.errors.QueryValidationError` for input
    that is empty, whitespace-only or over :data:`MAX_QUERY_LENGTH`.
    """
    original_length = len(raw)
    if original_length > MAX_QUERY_LENGTH:
        raise QueryValidationError(f"The question is longer than {MAX_QUERY_LENGTH} characters.")

    text = _WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFC", raw)).strip()
    if not _has_visible_content(text):
        raise QueryValidationError("The question is empty.")

    return UserQuery(text=text, original_length=original_length)


def _has_visible_content(text: str) -> bool:
    """Whether *text* contains anything a reader could see."""
    return any(unicodedata.category(character) not in _INVISIBLE_CATEGORIES for character in text)


def build_query_embedding_text(query: UserQuery) -> str:
    """Compose the text handed to the embedding provider.

    Version 1 is the identity function over the normalized question, and it is
    a named function anyway: this is the exact point at which text leaves the
    process for an external provider, and a boundary you can grep for is a
    boundary you can review.
    """
    return query.text
