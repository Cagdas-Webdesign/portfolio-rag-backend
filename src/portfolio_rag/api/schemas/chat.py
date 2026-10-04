"""Wire format of the chat endpoint.

Both models are live: a request is validated on every call, and a successful
call returns a :class:`ChatResponse`.

**What a response deliberately does not carry.** No similarity scores, no
source labels, no context, no prompt, no embedding space, no vector ids, no
fingerprints, no timings, no provider name, no model name. All of those exist
and are useful — to a developer, through the CLI and the logs. None of them is
something a browser client needs, and every one of them is a detail about the
inside of this service that a public API is better off not promising.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from portfolio_rag.domain.retrieval import SourceCitation
from portfolio_rag.rag.conversation import (
    MAX_CONVERSATION_TURNS,
    MAX_TURN_LENGTH,
    ConversationRole,
)

#: Upper bound on a single user message over HTTP.
#:
#: Deliberately *stricter* than the application's own
#: :data:`~portfolio_rag.rag.query.MAX_QUERY_LENGTH`, and the one place in this
#: project where a second limit is the right answer rather than a drifting
#: copy. They bound different things: the domain limit is what the pipeline can
#: process at all — it also applies to the CLI and to the evaluation harness —
#: while this one is what a public web page has any business sending. A
#: portfolio visitor asks a question; nobody types two thousand characters of
#: one by accident, and an anonymous endpoint that accepts four thousand is
#: paying for prompt-sized inputs it will never legitimately receive.
#:
#: The invariant that keeps the two honest is that this can only ever be the
#: tighter of the pair, which a test asserts.
MAX_MESSAGE_LENGTH = 2000

#: Conversation ids are minted by clients; constrain them so they are safe to
#: log and to echo.
CONVERSATION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class ChatTurn(BaseModel):
    """One earlier turn of the conversation, as the client saw it."""

    model_config = ConfigDict(extra="forbid")

    role: ConversationRole = Field(
        description="Who said it: `user` or `assistant`. No other role is accepted."
    )
    content: str = Field(
        min_length=1,
        max_length=MAX_TURN_LENGTH,
        description="What was said. Never logged.",
    )


class ChatRequest(BaseModel):
    """A user question, optionally with the turns immediately before it."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"message": "Which backend technologies are covered here?"}]
        },
    )

    message: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="The user's question. Never logged in full.",
    )
    conversation_id: str | None = Field(
        default=None,
        pattern=CONVERSATION_ID_PATTERN,
        description=(
            "Client-chosen identifier, echoed back unchanged so a client can "
            "correlate turns. This service keeps no conversation state: every "
            "request is answered on its own, from the knowledge base alone."
        ),
    )
    conversation: list[ChatTurn] = Field(
        default_factory=list,
        max_length=MAX_CONVERSATION_TURNS,
        description=(
            "Optional. The turns immediately before `message`, oldest first, so "
            'that a follow-up such as "and how is it deployed?" can be read. '
            "Used only to understand what the question refers to: it is never a "
            "source, never cited, and not used to search the knowledge base. "
            "When the turns exceed the conversation budget, the oldest whole "
            "turns are left out. Omitting it answers the question on its own."
        ),
    )


class ChatResponse(BaseModel):
    """A grounded answer with the sources it was built from.

    ``citations`` is empty exactly when the knowledge base does not support an
    answer. That is a normal, successful response — the honest one — and not an
    error.

    **The marks in ``answer`` index ``citations``.** A grounded answer carries
    ``[1]``, ``[2]``, … where it draws on a source, and ``[n]`` is the ``n``-th
    entry of the list — numbered from one, in the order the answer uses them,
    with no gaps. A source used twice keeps its number and is listed once. The
    labels the backend uses internally to verify a citation never appear here.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "answer": "The service exposes its HTTP API with FastAPI [1].",
                    "citations": [
                        {
                            "document_id": "backend-stack",
                            "title": "Backend Stack",
                            "source": "docs/backend-stack.md",
                            "section": "HTTP layer",
                        }
                    ],
                    "conversation_id": None,
                }
            ]
        },
    )

    answer: str = Field(description="The generated answer, grounded in the citations below.")
    citations: list[SourceCitation] = Field(
        default_factory=list,
        description=(
            "Sources the answer is grounded in, in the order the answer used "
            "them, so the first entry is the `[1]` in the answer text. Every "
            "entry refers to a document that was actually retrieved; empty "
            "when the knowledge base does not cover the question."
        ),
    )
    conversation_id: str | None = Field(
        default=None,
        description="Echo of the request's conversation id, when one was supplied.",
    )
