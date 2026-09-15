"""Descriptive numbers about a set of chunks, for developer inspection.

Counts and sizes only. There is deliberately no "quality score" here: nothing
in these structural measurements establishes retrieval quality. That is
measured separately, against questions with known answers.

Not observability, not analytics, not persisted anywhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.ingestion.chunking.policy import ChunkingPolicy


@dataclass(frozen=True, slots=True)
class ChunkStatistics:
    """A summary of one chunking run."""

    documents: int
    chunks: int
    sections: int
    split_sections: int
    chunks_with_overlap: int
    average_chars: int
    smallest_chars: int
    largest_chars: int
    over_max: int

    @classmethod
    def of(
        cls,
        chunks: Sequence[KnowledgeChunk],
        policy: ChunkingPolicy,
    ) -> ChunkStatistics:
        if not chunks:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0)

        sizes = [len(chunk.content) for chunk in chunks]
        # Keyed by the *source* section, not by the heading path: a document
        # with two `## Setup` sections has two sections, and neither of them was
        # split. Counting by path would report one section split in two.
        chunks_per_section: dict[tuple[str, int], int] = {}
        for chunk in chunks:
            key = (chunk.document_id, chunk.provenance.section_ordinal)
            chunks_per_section[key] = chunks_per_section.get(key, 0) + 1

        return cls(
            documents=len({chunk.document_id for chunk in chunks}),
            chunks=len(chunks),
            sections=len(chunks_per_section),
            split_sections=sum(1 for count in chunks_per_section.values() if count > 1),
            chunks_with_overlap=sum(
                1 for chunk in chunks if chunk.provenance.overlap_prefix_chars > 0
            ),
            average_chars=round(sum(sizes) / len(sizes)),
            smallest_chars=min(sizes),
            largest_chars=max(sizes),
            over_max=sum(1 for size in sizes if size > policy.max_chars),
        )

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("Documents", str(self.documents)),
            ("Chunks", str(self.chunks)),
            ("Sections", str(self.sections)),
            ("Split sections", str(self.split_sections)),
            ("With overlap", str(self.chunks_with_overlap)),
            ("Average chars", str(self.average_chars)),
            ("Smallest", str(self.smallest_chars)),
            ("Largest", str(self.largest_chars)),
            ("Over max", str(self.over_max)),
        )
