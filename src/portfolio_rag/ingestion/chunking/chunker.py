"""The chunker: validated documents in, retrieval units out.

Pure functions. No filesystem, no network, no provider, no global state — the
input is a validated :class:`KnowledgeDocument`, and the
output is fully determined by that document plus the policy. Reading files is
:mod:`portfolio_rag.ingestion.loader`'s job, and the two are composed at the
edge (the CLI), not inside each other.

The pipeline for one document:

    content → structure analysis → sections → block packing → overlap
            → identity + provenance → KnowledgeChunk[]
"""

from __future__ import annotations

from collections.abc import Sequence

from portfolio_rag.domain.knowledge import ChunkProvenance, KnowledgeChunk, KnowledgeDocument
from portfolio_rag.ingestion.chunking.errors import ChunkingInvariantError
from portfolio_rag.ingestion.chunking.fingerprint import build_chunk_id, compute_fingerprint
from portfolio_rag.ingestion.chunking.packing import pack_section
from portfolio_rag.ingestion.chunking.policy import (
    DEFAULT_CHUNKING_POLICY,
    MARKDOWN_CHUNKING_STRATEGY_VERSION,
    ChunkingPolicy,
)
from portfolio_rag.ingestion.chunking.structure import analyze_structure


def chunk_document(
    document: KnowledgeDocument,
    policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> tuple[KnowledgeChunk, ...]:
    """Cut one document into retrieval units, in source order.

    Raises :class:`~portfolio_rag.ingestion.chunking.errors.ChunkingError` when
    the document cannot be divided safely under *policy* — never silently
    returns something that violates the budget.
    """
    chunks: list[KnowledgeChunk] = []

    for section_ordinal, section in enumerate(analyze_structure(document.content)):
        for packed in pack_section(section, policy, document_id=document.metadata.id):
            chunks.append(
                _build_chunk(
                    document=document,
                    policy=policy,
                    ordinal=len(chunks),
                    section_ordinal=section_ordinal,
                    heading_path=section.heading_path,
                    content=packed.text,
                    overlap_prefix_chars=packed.overlap_prefix_chars,
                )
            )

    _verify_document_invariants(chunks, policy, document_id=document.metadata.id)
    return tuple(chunks)


def chunk_knowledge_base(
    documents: Sequence[KnowledgeDocument],
    policy: ChunkingPolicy = DEFAULT_CHUNKING_POLICY,
) -> tuple[KnowledgeChunk, ...]:
    """Cut a whole corpus, preserving the order the documents arrive in.

    No re-sorting: the loader already returns documents in a deterministic
    order, and quietly reordering them here would make the corpus depend on two
    sort rules instead of one.

    A document that cannot be chunked aborts the run. Returning the other
    documents would hand back a corpus that looks complete and is not, and
    every answer built on it would be confidently missing something.
    """
    chunks: list[KnowledgeChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, policy))

    _verify_corpus_invariants(chunks)
    return tuple(chunks)


def _build_chunk(
    *,
    document: KnowledgeDocument,
    policy: ChunkingPolicy,
    ordinal: int,
    section_ordinal: int,
    heading_path: tuple[str, ...],
    content: str,
    overlap_prefix_chars: int,
) -> KnowledgeChunk:
    document_id = document.metadata.id
    return KnowledgeChunk(
        id=build_chunk_id(document_id, ordinal),
        document_id=document_id,
        ordinal=ordinal,
        content=content,
        heading_path=heading_path,
        document_metadata=document.metadata,
        provenance=ChunkProvenance(
            document=document.provenance,
            strategy_version=MARKDOWN_CHUNKING_STRATEGY_VERSION,
            fingerprint=compute_fingerprint(
                strategy_version=MARKDOWN_CHUNKING_STRATEGY_VERSION,
                policy=policy,
                document_id=document_id,
                heading_path=heading_path,
                content=content,
            ),
            section_ordinal=section_ordinal,
            overlap_prefix_chars=overlap_prefix_chars,
        ),
    )


def _verify_document_invariants(
    chunks: Sequence[KnowledgeChunk],
    policy: ChunkingPolicy,
    *,
    document_id: str,
) -> None:
    """Check what this module promises, rather than trusting it.

    None of these should ever fire. They exist so that a future change which
    breaks packing fails loudly here instead of shipping oversized or
    misnumbered units into an index.
    """
    for position, chunk in enumerate(chunks):
        if chunk.ordinal != position:
            raise ChunkingInvariantError(
                f"Chunk ordinals must be gapless from 0; found {chunk.ordinal} at {position}.",
                document_id=document_id,
            )
        if len(chunk.content) > policy.max_chars:
            raise ChunkingInvariantError(
                f"Chunk {chunk.id} has {len(chunk.content)} characters, "
                f"over the {policy.max_chars}-character limit.",
                document_id=document_id,
                heading_path=chunk.heading_path,
            )
        if not chunk.content.strip():
            raise ChunkingInvariantError(
                f"Chunk {chunk.id} is empty.",
                document_id=document_id,
                heading_path=chunk.heading_path,
            )


def _verify_corpus_invariants(chunks: Sequence[KnowledgeChunk]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id in seen:
            raise ChunkingInvariantError(
                f"Chunk id `{chunk.id}` is not unique across the corpus.",
                document_id=chunk.document_id,
            )
        seen.add(chunk.id)
