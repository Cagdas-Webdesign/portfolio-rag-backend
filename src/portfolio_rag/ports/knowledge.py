"""Port for reading chunk text back at query time.

A vector index answers *"which units resemble this query?"* — it returns ids,
scores and the metadata retrieval filters on. It deliberately does not store
chunk text ([ADR 0006](../../../docs/adr/0006-versioned-embeddings-and-incremental-indexing.md)):
:class:`~portfolio_rag.domain.embedding.VectorMetadata` is a chosen list of
filter and citation fields, not a copy of the corpus.

So something has to turn a match back into the text an answer can be built
from, and that is this port. The source of truth is ``knowledge/`` — the same
Markdown the index was derived from — which means there is exactly one place a
passage can come from and no second copy to drift out of step with it.

Narrow on purpose: resolution by id, nothing else. No search, no listing, no
filtering — the index does that.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from portfolio_rag.domain.knowledge import KnowledgeChunk


class ChunkResolver(Protocol):
    """Returns the corpus text behind chunk ids."""

    async def resolve(self, chunk_ids: Sequence[str]) -> list[KnowledgeChunk]:
        """Return the chunks that exist, in the order they were asked for.

        Unknown ids are **absent**, not an error: an index can legitimately be
        one deploy ahead of or behind the corpus it was built from, and a
        retrieval path that raises in that window would turn a stale record
        into an outage. The caller sees fewer chunks than it asked for and
        decides what that means.
        """
        ...
