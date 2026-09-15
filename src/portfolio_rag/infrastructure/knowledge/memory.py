"""An in-process corpus snapshot, resolving chunk ids back to their text.

The corpus is a few dozen Markdown files in the repository, chunked
deterministically. Loading it once and keeping it in a dictionary is the whole
implementation, and that is the right size for the problem: a database whose
only job is to hand back text that is already on disk, in git, and identical on
every machine would be infrastructure bought to solve nothing.

**The snapshot is not a cache.** There is no second copy of the corpus to go
stale against: this *is* the corpus, read from the same files the index was
built from. A record whose chunk id is absent means the index and the source
tree are out of step — which the retrieval service reports rather than papers
over.

Built by the composition root at startup, from the loader and the chunker.
Nothing here opens a file: it receives chunks, exactly as the chunker produces
them.
"""

from __future__ import annotations

from collections.abc import Sequence

from portfolio_rag.domain.knowledge import KnowledgeChunk


class InMemoryChunkResolver:
    """Resolves chunk ids against a fixed snapshot of the corpus."""

    def __init__(self, chunks: Sequence[KnowledgeChunk]) -> None:
        self._chunks: dict[str, KnowledgeChunk] = {chunk.id: chunk for chunk in chunks}

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    async def resolve(self, chunk_ids: Sequence[str]) -> list[KnowledgeChunk]:
        """Return the chunks that exist, in the order they were asked for."""
        found = (self._chunks.get(chunk_id) for chunk_id in chunk_ids)
        return [chunk for chunk in found if chunk is not None]
