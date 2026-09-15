"""Bringing a vector index into line with the corpus.

**Convergent, not transactional.** An embedding provider and a vector store are
two remote systems; there is no transaction across them, and pretending
otherwise with compensating writes would add machinery that fails in its own
interesting ways. The guarantee made here is weaker and honest: a run either
completes or reports a failure — never a quiet partial success — and the next
run re-derives desired-versus-actual from scratch and converges. Nothing
persists between runs except the index itself.

**Write order: upserts before deletes.** Deleting first would mean that a
provider failure halfway through leaves the index missing records it used to
serve correctly, trading a stale index for an incomplete one. Upserting first
means the worst case is an index that still contains something outdated, which
the next run removes.

**The index is the cache.** A record already carrying the desired embedding
fingerprint already holds the vector for it, so nothing needs re-embedding.
There is deliberately no second cache layer — no Redis, no file of vectors — to
disagree with the index about what exists.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

from portfolio_rag.application.indexing.errors import (
    EmbeddingValidationError,
    IncompatibleEmbeddingSpaceError,
    IndexSynchronizationError,
)
from portfolio_rag.application.indexing.plan import (
    DesiredRecord,
    IndexPlan,
    build_desired_state,
    plan_index,
)
from portfolio_rag.core.logging import get_logger
from portfolio_rag.domain.embedding import EmbeddingSpec, VectorRecord, VectorRecordState
from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingProvider
from portfolio_rag.ports.errors import UnsupportedVectorStoreOperationError
from portfolio_rag.ports.vector_store import VectorStore

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """What a run actually did. Counts and durations — nothing evaluative."""

    desired: int
    created: int
    reembedded: int
    metadata_updated: int
    unchanged: int
    deleted: int
    embeddings_generated: int
    embeddings_reused: int
    stale_detection: bool
    duration_seconds: float

    @property
    def records_written(self) -> int:
        return self.created + self.reembedded + self.metadata_updated

    @property
    def is_noop(self) -> bool:
        return self.records_written == 0 and self.deleted == 0

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("desired chunks", str(self.desired)),
            ("created", str(self.created)),
            ("re-embedded", str(self.reembedded)),
            ("metadata only", str(self.metadata_updated)),
            ("unchanged", str(self.unchanged)),
            ("deleted", str(self.deleted)),
            ("embeddings generated", str(self.embeddings_generated)),
            ("embeddings reused", str(self.embeddings_reused)),
            ("records written", str(self.records_written)),
        )


class IndexingService:
    """Plans and applies index changes for one corpus and one embedding space."""

    def __init__(self, provider: EmbeddingProvider, store: VectorStore) -> None:
        self._provider = provider
        self._store = store

    @property
    def spec(self) -> EmbeddingSpec:
        return self._provider.spec

    async def plan(self, chunks: Sequence[KnowledgeChunk], *, rebuild: bool = False) -> IndexPlan:
        """Work out what would change. Reads the index; writes nothing.

        No embedding call is made here — the plan is derived from fingerprints
        alone, which is exactly why it can be shown to a developer before any
        money or time is spent.
        """
        self._require_compatible_index()
        desired = build_desired_state(chunks, self._provider.spec)
        actual, stale_detection = await self._read_actual_state(desired)

        if not rebuild:
            self._require_compatible_records(actual)

        return plan_index(desired, actual, stale_detection=stale_detection, rebuild=rebuild)

    async def synchronize(
        self, chunks: Sequence[KnowledgeChunk], *, rebuild: bool = False
    ) -> IndexingResult:
        """Bring the index in line with *chunks*, or fail loudly trying."""
        started = time.perf_counter()
        plan = await self.plan(chunks, rebuild=rebuild)

        vectors = await self._embed(plan)
        records = [
            *self._records_from(plan.create, vectors),
            *self._records_from(plan.reembed, vectors),
            *await self._records_for_metadata_updates(plan),
        ]

        try:
            if records:
                await self._store.upsert(records)
            if plan.delete:
                await self._store.delete(plan.delete)
        except Exception as exc:
            # Whatever already succeeded stays; the next run re-plans against
            # the index as it now is and finishes the job.
            raise IndexSynchronizationError(
                f"Index synchronization failed after writing {len(records)} record(s): {exc}"
            ) from exc

        result = IndexingResult(
            desired=plan.desired_count,
            created=len(plan.create),
            reembedded=len(plan.reembed),
            metadata_updated=len(plan.metadata_updates),
            unchanged=len(plan.unchanged),
            deleted=len(plan.delete),
            embeddings_generated=len(vectors),
            embeddings_reused=len(plan.unchanged) + len(plan.metadata_updates),
            stale_detection=plan.stale_detection,
            duration_seconds=round(time.perf_counter() - started, 3),
        )
        # Field names avoid `logging.LogRecord`'s own attributes (`created`,
        # `msecs`, `module`, …); colliding with one raises at emit time.
        _logger.info(
            "index synchronized",
            extra={
                "embedding_space": self._provider.spec.identity,
                "desired_records": result.desired,
                "records_created": result.created,
                "records_reembedded": result.reembedded,
                "records_metadata_updated": result.metadata_updated,
                "records_deleted": result.deleted,
                "embeddings_generated": result.embeddings_generated,
                "duration_seconds": result.duration_seconds,
            },
        )
        return result

    # --- steps --------------------------------------------------------------

    async def _read_actual_state(
        self, desired: Sequence[DesiredRecord]
    ) -> tuple[list[VectorRecordState], bool]:
        """Read what the index holds, using the cheapest sufficient call.

        A store that can enumerate gives the whole picture, which is what stale
        detection needs. One that cannot is asked only about the ids we care
        about, and the caller is told that stale records are undiscoverable
        rather than left to assume there are none.
        """
        if self._store.supports_enumeration:
            return await self._store.list_state(), True
        try:
            states = await self._store.fetch_states([record.id for record in desired])
        except UnsupportedVectorStoreOperationError:  # pragma: no cover - defensive
            return [], False
        return states, False

    async def _embed(self, plan: IndexPlan) -> dict[str, tuple[float, ...]]:
        """Embed everything the plan requires, keyed by embedding fingerprint."""
        if not plan.embeddings_required:
            return {}

        inputs = [
            EmbeddingInput(id=record.embedding_fingerprint, text=record.embedding_text)
            for record in plan.embeddings_required
        ]
        results = await self._provider.embed(inputs)
        return self._validate(inputs, results)

    def _validate(
        self, inputs: Sequence[EmbeddingInput], results: Sequence[object]
    ) -> dict[str, tuple[float, ...]]:
        """Check the port's promises before a single vector is stored.

        A provider adapter validates its own responses, but this check exists
        for every provider — including the next one, written by someone who
        assumed nobody downstream was looking.
        """
        from portfolio_rag.ports.embeddings import EmbeddingResult

        if len(results) != len(inputs):
            raise EmbeddingValidationError(
                f"Provider returned {len(results)} embeddings for {len(inputs)} inputs."
            )

        dimensions = self._provider.spec.dimensions
        vectors: dict[str, tuple[float, ...]] = {}
        expected_ids = {item.id for item in inputs}

        for position, result in enumerate(results):
            if not isinstance(result, EmbeddingResult):  # pragma: no cover - typing guard
                raise EmbeddingValidationError(f"Result {position} is not an embedding result.")
            if result.id not in expected_ids:
                raise EmbeddingValidationError(
                    f"Provider returned an embedding for an unknown input `{result.id}`."
                )
            if len(result.vector) != dimensions:
                raise EmbeddingValidationError(
                    f"Embedding for `{result.id}` has {len(result.vector)} dimensions, "
                    f"expected {dimensions}."
                )
            if any(not math.isfinite(value) for value in result.vector):
                raise EmbeddingValidationError(
                    f"Embedding for `{result.id}` contains a non-finite value."
                )
            vectors[result.id] = result.vector

        missing = expected_ids - set(vectors)
        if missing:
            raise EmbeddingValidationError(
                f"Provider did not return embeddings for {len(missing)} input(s)."
            )
        return vectors

    def _records_from(
        self, desired: Sequence[DesiredRecord], vectors: dict[str, tuple[float, ...]]
    ) -> list[VectorRecord]:
        return [self._record(record, vectors[record.embedding_fingerprint]) for record in desired]

    async def _records_for_metadata_updates(self, plan: IndexPlan) -> list[VectorRecord]:
        """Rewrite metadata around vectors the index already holds.

        The existing vector is fetched and re-upserted unchanged. Asking the
        provider for it again would produce exactly the same numbers at exactly
        the same cost as not needing to.
        """
        if not plan.metadata_updates:
            return []

        wanted = {record.id: record for record in plan.metadata_updates}
        existing = await self._store.fetch(list(wanted))
        found = {record.id: record for record in existing}

        missing = set(wanted) - set(found)
        if missing:
            raise IndexSynchronizationError(
                f"{len(missing)} record(s) needed a metadata update but could not be read back."
            )
        return [
            self._record(wanted[identifier], found[identifier].embedding) for identifier in wanted
        ]

    def _record(self, desired: DesiredRecord, vector: tuple[float, ...]) -> VectorRecord:
        return VectorRecord(
            id=desired.id,
            spec=self._provider.spec,
            embedding_fingerprint=desired.embedding_fingerprint,
            chunk_fingerprint=desired.chunk_fingerprint,
            document_fingerprint=desired.document_fingerprint,
            metadata=desired.metadata,
            embedding=vector,
        )

    # --- compatibility ------------------------------------------------------

    def _require_compatible_index(self) -> None:
        index_spec = self._store.index_spec.embedding
        provider_spec = self._provider.spec
        if not index_spec.is_compatible_with(provider_spec):
            raise IncompatibleEmbeddingSpaceError(
                f"The index holds `{index_spec.identity}` but the provider makes "
                f"`{provider_spec.identity}`. Point at an index for this embedding "
                "space, or rebuild."
            )

    def _require_compatible_records(self, actual: Sequence[VectorRecordState]) -> None:
        provider_spec = self._provider.spec
        stale = {record.spec.identity for record in actual if record.spec != provider_spec}
        if stale:
            raise IncompatibleEmbeddingSpaceError(
                f"The index already contains vectors from {sorted(stale)}, which cannot "
                f"be compared with `{provider_spec.identity}`. Rebuild the index "
                "explicitly rather than mixing embedding spaces."
            )
