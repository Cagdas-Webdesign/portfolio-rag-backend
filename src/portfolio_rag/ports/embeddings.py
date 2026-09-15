"""Port for embedding providers.

Text in, vectors out — and deliberately nothing else. A provider does not know
what a chunk is, does not compose the text it embeds, and does not decide what
belongs in it. It receives finished :class:`EmbeddingInput` values and returns
one vector for each, in an order it must not get wrong.

Batching is part of the contract because it is part of the problem: embedding a
corpus one HTTP request at a time is slow and, with a metered provider,
needlessly expensive. How a provider splits a batch internally is its own
business; the caller passes the whole sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from portfolio_rag.domain.embedding import EmbeddingSpec, EmbeddingVector


class EmbeddingInput(BaseModel):
    """One piece of text to embed, with a caller-chosen handle.

    The id exists so results can be matched back without relying on position
    alone. Providers that return results out of order are not hypothetical, and
    a silently mismatched vector is the worst kind of bug: everything keeps
    working, and every answer is subtly wrong.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class EmbeddingResult(BaseModel):
    """One vector, carrying back the id of the input that produced it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200)
    vector: EmbeddingVector


class EmbeddingProvider(Protocol):
    """Turns text into vectors within one embedding space."""

    @property
    def spec(self) -> EmbeddingSpec:
        """The space this provider produces vectors in.

        Recorded on every indexed record, so a later run can tell whether the
        index it is looking at is even comparable to what this provider makes.
        """
        ...

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed every input, returning one result per input, in input order.

        Implementations must fail rather than return a partial or reordered
        batch: the caller checks, but a provider that guesses has already lost
        the information needed to check anything.
        """
        ...
