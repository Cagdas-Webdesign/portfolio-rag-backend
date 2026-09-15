"""Builders for embedding and vector-store value objects.

Vector records carry a lot of required identity, almost none of which a given
test cares about. These keep the noise out of the assertions.
"""

from __future__ import annotations

from typing import Any

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    VectorMetadata,
    VectorRecord,
    VectorRecordState,
)
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION

DEFAULT_SPEC = EmbeddingSpec(
    provider="deterministic",
    model="sha256-derived-v1",
    dimensions=3,
    representation_version=EMBEDDING_REPRESENTATION_VERSION,
)


def make_metadata(chunk_id: str = "doc--0000", **overrides: Any) -> VectorMetadata:
    fields: dict[str, Any] = {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "document_title": "Test Document",
        "heading_path": ("Section",),
        "source_path": "doc.md",
        "document_type": "reference",
        "language": "en",
        "visibility": "public",
        "trust_level": "verified",
        "topics": (),
        "technologies": (),
    }
    fields.update(overrides)
    return VectorMetadata.model_validate(fields)


def make_vector_record(
    record_id: str = "doc--0000",
    *,
    spec: EmbeddingSpec = DEFAULT_SPEC,
    embedding: tuple[float, ...] = (0.1, 0.2, 0.3),
    embedding_fingerprint: str | None = None,
    chunk_fingerprint: str | None = None,
    document_fingerprint: str | None = None,
    metadata: VectorMetadata | None = None,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        spec=spec,
        embedding=embedding,
        embedding_fingerprint=embedding_fingerprint or _digest(record_id, "embedding"),
        chunk_fingerprint=chunk_fingerprint or _digest(record_id, "chunk"),
        document_fingerprint=document_fingerprint or _digest(record_id, "document"),
        metadata=metadata or make_metadata(record_id),
    )


def make_record_state(record_id: str = "doc--0000", **overrides: Any) -> VectorRecordState:
    return make_vector_record(record_id, **overrides).state()


def _digest(record_id: str, kind: str) -> str:
    """A syntactically valid, readable, stable stand-in for a real fingerprint."""
    import hashlib

    return hashlib.sha256(f"{kind}:{record_id}".encode()).hexdigest()
