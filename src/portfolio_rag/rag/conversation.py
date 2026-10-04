"""The turns before a question: what a follow-up refers to, never evidence.

A visitor asks "Which technologies does the backend use?", then "And how is it
deployed?". The second question is unintelligible on its own — "it" is the
backend only because of the first. This module bounds the few turns a client
may send along with a question so that a model can read such a reference, and
nothing more.

**Where it goes, and where it does not.** Earlier turns reach exactly one place:
the generation prompt, in a section of their own that the instructions name as
not being a source (:mod:`portfolio_rag.rag.prompt`). They are not embedded, so
retrieval searches for the question as asked and an earlier topic cannot pull
the search towards itself. They are not shown to the grounding check, so an
earlier answer — the model's own words — can never be what confirms a new one.
They produce no citation: labels resolve against the retrieved context only.

**This service still keeps no conversation state.** The client sends the turns
it wants considered; nothing is stored between requests, and nothing here is
memory.

**Bounded, deterministically.** At most :data:`MAX_CONVERSATION_TURNS` turns,
each at most :data:`MAX_TURN_LENGTH` characters, normalized like a question.
Of those, the newest whole turns that fit :data:`MAX_CONVERSATION_CHARACTERS`
are kept and the older ones are left out and counted — a turn is never cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from portfolio_rag.rag.errors import QueryValidationError
from portfolio_rag.rag.query import normalize_query

#: How many earlier turns a request may carry: three exchanges. Enough for a
#: reference to the previous question and its answer, far too few to be a
#: transcript.
MAX_CONVERSATION_TURNS: Final = 6

#: Upper bound on one earlier turn, in characters — the same bound a question
#: has over HTTP, so an answer a client received can be sent back whole.
MAX_TURN_LENGTH: Final = 2000

#: How much earlier conversation one prompt may spend, in characters, after
#: normalization. About 1000 estimated tokens, which leaves the retrieved
#: passages the larger share of the prompt budget.
MAX_CONVERSATION_CHARACTERS: Final = 3000


class ConversationRole(StrEnum):
    """Who said an earlier turn. There is no system role, by construction."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One earlier turn, as a caller supplied it."""

    role: ConversationRole
    text: str


@dataclass(frozen=True, slots=True)
class Conversation:
    """The earlier turns a prompt will carry, oldest first."""

    turns: tuple[ConversationTurn, ...] = ()
    dropped: int = 0
    """Valid turns left out because the newer ones already filled the budget."""

    @property
    def is_empty(self) -> bool:
        return not self.turns


#: No earlier turns: the single-question path, exactly as it was.
NO_CONVERSATION: Final = Conversation()


def bound_conversation(turns: Sequence[ConversationTurn]) -> Conversation:
    """Validate, normalize and bound the turns sent with a question.

    Raises :class:`~portfolio_rag.rag.errors.QueryValidationError` for more
    than :data:`MAX_CONVERSATION_TURNS` turns, a turn over
    :data:`MAX_TURN_LENGTH` characters, or a turn with nothing visible in it.
    The message names the rule, never the content.
    """
    if len(turns) > MAX_CONVERSATION_TURNS:
        raise QueryValidationError(
            f"The conversation has more than {MAX_CONVERSATION_TURNS} turns."
        )

    normalized: list[ConversationTurn] = []
    for turn in turns:
        if len(turn.text) > MAX_TURN_LENGTH:
            raise QueryValidationError(
                f"A conversation turn is longer than {MAX_TURN_LENGTH} characters."
            )
        try:
            # The same boundary a question passes: NFC, whitespace collapsed.
            # Collapsing also means a turn is one line, so no turn can open a
            # section header of its own in the prompt.
            text = normalize_query(turn.text).text
        except QueryValidationError:
            raise QueryValidationError("A conversation turn is empty.") from None
        normalized.append(ConversationTurn(role=ConversationRole(turn.role), text=text))

    kept: list[ConversationTurn] = []
    spent = 0
    for turn in reversed(normalized):
        if spent + len(turn.text) > MAX_CONVERSATION_CHARACTERS:
            break
        kept.append(turn)
        spent += len(turn.text)

    return Conversation(turns=tuple(reversed(kept)), dropped=len(normalized) - len(kept))
