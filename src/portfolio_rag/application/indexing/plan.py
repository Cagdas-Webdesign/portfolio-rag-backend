"""Deciding what an index run should do, before it does anything.

Planning compares two things:

* the **desired state** — what the current corpus and the current embedding
  space say the index ought to contain;
* the **actual state** — what it contains now.

The output is a plan, and computing it performs no writes and no embedding
calls. That separation is what makes ``--dry-run`` truthful rather than a
promise, and what makes the interesting logic testable without a provider.

The four outcomes are not three. "Update" splits in two, and the distinction
earns its keep every time somebody flips a `visibility` flag:

* **create** — no record with this id exists.
* **re-embed** — the embedding fingerprint differs, so the text that would be
  embedded has changed. A provider call is needed.
* **metadata update** — the embedding fingerprint matches but the stored
  metadata does not. The vector is still correct; only the record around it is
  stale. Re-embedding here would spend money to produce a vector identical to
  the one already stored.
* **unchanged** — fingerprint and metadata both match. Do nothing.

Plus **delete**, for records the corpus no longer wants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    VectorMetadata,
    VectorRecordState,
)
from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.ingestion.embedding import build_embedding_text, compute_embedding_fingerprint


@dataclass(frozen=True, slots=True)
class DesiredRecord:
    """One retrieval unit as the corpus currently wants it indexed."""

    id: str
    embedding_text: str
    embedding_fingerprint: str
    chunk_fingerprint: str
    document_fingerprint: str
    metadata: VectorMetadata


def build_desired_state(
    chunks: Sequence[KnowledgeChunk], spec: EmbeddingSpec
) -> tuple[DesiredRecord, ...]:
    """Project chunks into what the index should hold, in corpus order."""
    return tuple(_desired_record(chunk, spec) for chunk in chunks)


def _desired_record(chunk: KnowledgeChunk, spec: EmbeddingSpec) -> DesiredRecord:
    text = build_embedding_text(chunk)
    return DesiredRecord(
        id=chunk.id,
        embedding_text=text,
        embedding_fingerprint=compute_embedding_fingerprint(text, spec),
        chunk_fingerprint=chunk.provenance.fingerprint,
        document_fingerprint=chunk.provenance.document.document_fingerprint,
        metadata=VectorMetadata(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=chunk.document_metadata.title,
            heading_path=chunk.heading_path,
            source_path=chunk.provenance.document.source_path,
            document_type=chunk.document_metadata.document_type,
            language=chunk.document_metadata.language,
            visibility=chunk.document_metadata.visibility,
            trust_level=chunk.document_metadata.trust_level,
            topics=chunk.document_metadata.topics,
            technologies=chunk.document_metadata.technologies,
        ),
    )


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """What a synchronization run would do. Computing this changes nothing."""

    create: tuple[DesiredRecord, ...] = ()
    reembed: tuple[DesiredRecord, ...] = ()
    metadata_updates: tuple[DesiredRecord, ...] = ()
    unchanged: tuple[DesiredRecord, ...] = ()
    delete: tuple[str, ...] = ()
    stale_detection: bool = True
    """False when the store cannot enumerate its contents, which means the
    `delete` list is not evidence that nothing is stale — only that nothing
    stale could be found."""

    embeddings_required: tuple[DesiredRecord, ...] = field(default=())
    """Records that need a provider call. Deduplicated by embedding
    fingerprint: two chunks with identical text in the same space share a
    vector, and paying twice for it would be waste."""

    @property
    def is_noop(self) -> bool:
        """True when the index already matches the corpus."""
        return not (self.create or self.reembed or self.metadata_updates or self.delete)

    @property
    def desired_count(self) -> int:
        return (
            len(self.create) + len(self.reembed) + len(self.metadata_updates) + len(self.unchanged)
        )

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("create", str(len(self.create))),
            ("re-embed", str(len(self.reembed))),
            ("metadata only", str(len(self.metadata_updates))),
            ("unchanged", str(len(self.unchanged))),
            ("delete", str(len(self.delete))),
        )


def plan_index(
    desired: Sequence[DesiredRecord],
    actual: Sequence[VectorRecordState],
    *,
    stale_detection: bool = True,
    rebuild: bool = False,
) -> IndexPlan:
    """Compare desired against actual and decide what has to happen.

    With *rebuild*, everything desired is treated as a create: the incremental
    reuse is deliberately switched off, which is the escape hatch for an index
    whose embedding space no longer matches.
    """
    actual_by_id: Mapping[str, VectorRecordState] = {record.id: record for record in actual}

    create: list[DesiredRecord] = []
    reembed: list[DesiredRecord] = []
    metadata_updates: list[DesiredRecord] = []
    unchanged: list[DesiredRecord] = []

    for record in desired:
        existing = actual_by_id.get(record.id)
        if rebuild or existing is None:
            create.append(record)
        elif existing.embedding_fingerprint != record.embedding_fingerprint:
            reembed.append(record)
        elif _metadata_differs(existing, record):
            metadata_updates.append(record)
        else:
            unchanged.append(record)

    desired_ids = {record.id for record in desired}
    delete = tuple(
        sorted(identifier for identifier in actual_by_id if identifier not in desired_ids)
    )

    return IndexPlan(
        create=tuple(create),
        reembed=tuple(reembed),
        metadata_updates=tuple(metadata_updates),
        unchanged=tuple(unchanged),
        delete=delete,
        stale_detection=stale_detection,
        embeddings_required=_deduplicate_by_fingerprint([*create, *reembed]),
    )


def _metadata_differs(existing: VectorRecordState, desired: DesiredRecord) -> bool:
    """Anything stored alongside the vector that no longer matches the corpus.

    The lineage fingerprints count: a chunk whose document changed elsewhere
    still needs its `document_fingerprint` corrected, even though its own text
    — and therefore its vector — is untouched.
    """
    return (
        existing.metadata != desired.metadata
        or existing.chunk_fingerprint != desired.chunk_fingerprint
        or existing.document_fingerprint != desired.document_fingerprint
    )


def _deduplicate_by_fingerprint(records: Sequence[DesiredRecord]) -> tuple[DesiredRecord, ...]:
    seen: set[str] = set()
    unique: list[DesiredRecord] = []
    for record in records:
        if record.embedding_fingerprint in seen:
            continue
        seen.add(record.embedding_fingerprint)
        unique.append(record)
    return tuple(unique)
