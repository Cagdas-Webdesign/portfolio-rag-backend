"""An in-memory vector store — the reference implementation of the port.

Not a stub and not a test double: it enforces every rule the contract makes,
including the ones a careless real adapter would skip. Dimension checks,
embedding-space checks, non-finite rejection, deterministic ordering — if this
store accepts something, the contract permits it.

That strictness is the point. The whole indexing pipeline runs against this
store with no account, no credentials and no network, so a developer or CI can
prove that planning, idempotency, stale deletion and metadata-only updates work
before a single vector is sent anywhere.

State lives in a dictionary and disappears with the process. There is
deliberately no file or SQLite persistence: a hand-rolled vector database would
be throwaway infrastructure competing with the real adapters for trust.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from portfolio_rag.domain.embedding import (
    EmbeddingVector,
    SimilarityMetric,
    VectorIndexSpec,
    VectorRecord,
    VectorRecordState,
)
from portfolio_rag.ports.errors import VectorStoreError
from portfolio_rag.ports.vector_store import VectorMatch, VectorQuery


class InMemoryVectorStore:
    """A complete, strict, non-persistent implementation of :class:`VectorStore`."""

    def __init__(self, index_spec: VectorIndexSpec) -> None:
        self._index_spec = index_spec
        self._records: dict[str, VectorRecord] = {}

    @property
    def index_spec(self) -> VectorIndexSpec:
        return self._index_spec

    @property
    def supports_enumeration(self) -> bool:
        """True: this store knows everything it holds."""
        return True

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or replace records, after checking every one of them.

        Validation happens for the whole batch before anything is written, so a
        rejected batch leaves the index exactly as it was rather than half
        applied.
        """
        for record in records:
            self._require_compatible(record)
        for record in records:
            self._records[record.id] = record

    async def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        return [self._records[identifier] for identifier in ids if identifier in self._records]

    async def fetch_states(self, ids: Sequence[str]) -> list[VectorRecordState]:
        return [record.state() for record in await self.fetch(ids)]

    async def list_state(self) -> list[VectorRecordState]:
        """Every record, ordered by id so callers never depend on insertion order."""
        return [self._records[identifier].state() for identifier in sorted(self._records)]

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove records. Deleting something absent is not an error — the
        caller's intent ("this must not be in the index") is already satisfied."""
        for identifier in ids:
            self._records.pop(identifier, None)

    async def query(self, query: VectorQuery) -> list[VectorMatch]:
        """Score every record that passes the filters, best first.

        Ties are broken by ascending id. Two records with an identical score is
        not a rare case — identical text produces identical vectors — and an
        arbitrary order there would make results irreproducible.
        """
        self._require_query_dimensions(query.embedding)

        scored = [
            (self._similarity(query.embedding, record.embedding), record)
            for record in self._records.values()
            if _matches(record, query.filters)
        ]
        scored.sort(key=lambda entry: (-entry[0], entry[1].id))
        return [
            VectorMatch(record=record.state(), score=score)
            for score, record in scored[: query.top_k]
        ]

    # --- validation ---------------------------------------------------------

    def _require_compatible(self, record: VectorRecord) -> None:
        if not self._index_spec.embedding.is_compatible_with(record.spec):
            raise VectorStoreError(
                f"Record `{record.id}` was produced in embedding space "
                f"`{record.spec.identity}`, but this index holds "
                f"`{self._index_spec.embedding.identity}`. Mixing them would make "
                "every similarity in this index meaningless."
            )
        if len(record.embedding) != self._index_spec.dimensions:
            raise VectorStoreError(
                f"Record `{record.id}` has {len(record.embedding)} dimensions; "
                f"this index holds {self._index_spec.dimensions}."
            )
        # `EmbeddingVector` already rejects non-finite values, so reaching this
        # store with one means the record was built around validation.
        for position, value in enumerate(record.embedding):
            if not math.isfinite(value):
                raise VectorStoreError(
                    f"Record `{record.id}` has a non-finite value at index {position}."
                )

    def _require_query_dimensions(self, embedding: EmbeddingVector) -> None:
        if len(embedding) != self._index_spec.dimensions:
            raise VectorStoreError(
                f"Query vector has {len(embedding)} dimensions; "
                f"this index holds {self._index_spec.dimensions}."
            )

    def _similarity(self, left: EmbeddingVector, right: EmbeddingVector) -> float:
        if self._index_spec.metric is not SimilarityMetric.COSINE:  # pragma: no cover
            raise VectorStoreError(f"Unsupported metric: {self._index_spec.metric}")
        return _cosine_similarity(left, right)


def _cosine_similarity(left: EmbeddingVector, right: EmbeddingVector) -> float:
    """Cosine similarity, with zero vectors scoring 0 rather than dividing by 0."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _matches(record: VectorRecord, filters: Mapping[str, str]) -> bool:
    """Exact-match filtering over the record's metadata.

    Values are compared as strings so that enums, ids and free text all behave
    the same way — which is also how a real store's metadata filtering works.
    """
    for field, expected in filters.items():
        if not hasattr(record.metadata, field):
            return False
        actual = getattr(record.metadata, field)
        if isinstance(actual, tuple):
            if expected not in [str(item) for item in actual]:
                return False
        elif str(actual) != expected:
            return False
    return True
