"""Port for vector stores.

A storage contract, not a retrieval engine. :meth:`VectorStore.query` is a
primitive — a vector goes in, scored records come out — and turning a user's
question into that vector is the query side's job.

The operations are what indexing genuinely needs: write records, read their
state to decide what to change, fetch whole records when metadata has to be
rewritten without re-embedding, delete what is no longer wanted, and search.

Enumeration is a *capability*, not an assumption. Some real stores — Cloudflare
Vectorize among them — can fetch by id but cannot list what they contain. A
port that pretended otherwise would produce an adapter that lies. Stores
declare :attr:`VectorStore.supports_enumeration`, and planning adapts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from portfolio_rag.domain.embedding import (
    EmbeddingVector,
    VectorIndexSpec,
    VectorRecord,
    VectorRecordState,
)


class VectorQuery(BaseModel):
    """A similarity search.

    ``filters`` are exact-match constraints on record metadata, for example
    ``{"visibility": "public"}``. Every store supports at least that; richer
    query languages stay provider-specific and out of this contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding: EmbeddingVector
    top_k: int = Field(default=5, ge=1, le=100)
    filters: Mapping[str, str] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    """A record the store considers similar, with its score.

    Carries the record's state rather than its vector: a caller comparing
    results has no use for the embedding, and returning it would move a lot of
    numbers for nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    record: VectorRecordState
    score: float = Field(description="Higher is closer, in the index's configured metric.")


class VectorStore(Protocol):
    """Persists and searches knowledge vectors."""

    @property
    def index_spec(self) -> VectorIndexSpec:
        """What this index holds and how it scores. Checked before any write."""
        ...

    @property
    def supports_enumeration(self) -> bool:
        """Whether :meth:`list_state` can list the whole index.

        ``False`` means stale records cannot be discovered by this store, only
        overwritten by id — which planning has to know rather than assume.
        """
        ...

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records, keyed by ``record.id``."""
        ...

    async def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        """Return the records that exist, whole. Unknown ids are simply absent."""
        ...

    async def fetch_states(self, ids: Sequence[str]) -> list[VectorRecordState]:
        """Return the state of the records that exist, without their vectors."""
        ...

    async def list_state(self) -> list[VectorRecordState]:
        """Return the state of every record in the index.

        Raises ``UnsupportedVectorStoreOperationError`` when
        :attr:`supports_enumeration` is ``False``.
        """
        ...

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove records by id. Unknown ids are ignored, not an error."""
        ...

    async def query(self, query: VectorQuery) -> list[VectorMatch]:
        """Return the best matches, most similar first, deterministically ordered."""
        ...
