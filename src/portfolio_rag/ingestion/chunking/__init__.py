"""Structure-aware chunking: validated documents into retrieval units.

    KnowledgeDocument
      → markdown structure analysis   headings and blocks, from a real parser
      → heading hierarchy             a path, not a single section label
      → block-aware packing           whole blocks while the budget allows
      → size bounding                 soft target, hard maximum
      → controlled overlap            within a split section only
      → metadata + provenance         inherited, never re-read from disk
      → stable identity               readable id, SHA-256 fingerprint
      → KnowledgeChunk[]

Chunking belongs to ingestion — the corpus side — because a chunk boundary is a
property of the corpus, decided once when a document is ingested. Deciding it
per query would mean re-deciding it on every request, with no way to know what
was embedded.

Two things this module deliberately does *not* do:

* **Open anything.** It receives documents; it never reads a file, parses YAML
  or resolves a path. That is :mod:`portfolio_rag.ingestion.loader`'s job.
* **Compose an embedding representation.** ``chunk.content`` is source text.
  What gets composed for an embedding is
  :mod:`portfolio_rag.ingestion.embedding`'s job — the heading path and title
  are kept beside the content as data, not baked into it.

Public API: :func:`chunk_document`, :func:`chunk_knowledge_base`,
:class:`ChunkingPolicy`.
"""

from portfolio_rag.ingestion.chunking.chunker import chunk_document, chunk_knowledge_base
from portfolio_rag.ingestion.chunking.errors import (
    ChunkingError,
    ChunkingErrorCode,
    ChunkingInvariantError,
    InvalidChunkingPolicyError,
    UnsplittableBlockError,
)
from portfolio_rag.ingestion.chunking.policy import (
    DEFAULT_CHUNKING_POLICY,
    MARKDOWN_CHUNKING_STRATEGY_VERSION,
    ChunkingPolicy,
)
from portfolio_rag.ingestion.chunking.statistics import ChunkStatistics

__all__ = [
    "DEFAULT_CHUNKING_POLICY",
    "MARKDOWN_CHUNKING_STRATEGY_VERSION",
    "ChunkStatistics",
    "ChunkingError",
    "ChunkingErrorCode",
    "ChunkingInvariantError",
    "ChunkingPolicy",
    "InvalidChunkingPolicyError",
    "UnsplittableBlockError",
    "chunk_document",
    "chunk_knowledge_base",
]
