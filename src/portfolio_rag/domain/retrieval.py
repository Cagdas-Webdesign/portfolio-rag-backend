"""Retrieval results and the citations derived from them.

Two models, two audiences, and the gap between them is the point.

:class:`RetrievedChunk` is **internal**: the whole chunk — content, heading
path, metadata, provenance — plus how similar the store found it. It is what
context building, grounding and diagnostics work with.

:class:`SourceCitation` is the **client-facing** projection: a chosen handful
of fields a reader can act on. No chunk text, no similarity, no fingerprints,
no chunk id, no internal classification. Anything a client is allowed to see
is listed here explicitly, which is what makes the boundary reviewable.

**Similarity is not confidence.** ``similarity`` is a cosine score in the
index's metric: a number that orders results, not a probability that an answer
is correct. Nothing in this system renders it as a percentage.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from portfolio_rag.domain.knowledge import KnowledgeChunk, Slug


class RetrievedChunk(BaseModel):
    """A chunk a vector search returned, with how close it scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: KnowledgeChunk
    similarity: float = Field(
        description=(
            "Similarity in the index's metric, as reported by the vector store. "
            "Higher is closer. Not a probability and not a confidence."
        )
    )

    @property
    def chunk_id(self) -> str:
        return self.chunk.id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    def citation(self) -> SourceCitation:
        """The public projection of where this passage came from."""
        return SourceCitation.from_chunk(self.chunk)


class SourceCitation(BaseModel):
    """A reference the client can show next to an answer.

    Built by the backend from a chunk that was actually retrieved — never from
    anything a language model wrote. A model may point at a source label; it
    may not name a document, a path or a URL and have that believed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: Slug = Field(description="Stable id of the document the passage came from.")
    title: str = Field(
        min_length=1, max_length=200, description="Title of that document, as its author wrote it."
    )
    source: str = Field(
        min_length=1,
        max_length=500,
        description="URL, repository path or file name of the original document.",
    )
    section: str | None = Field(
        default=None,
        max_length=200,
        description="Innermost heading the passage sat under, when it had one.",
    )

    @classmethod
    def from_chunk(cls, chunk: KnowledgeChunk) -> SourceCitation:
        """Project a chunk down to what a client is allowed to see."""
        return cls(
            document_id=chunk.document_id,
            title=chunk.document_metadata.title,
            source=chunk.document_metadata.source,
            section=chunk.section,
        )
