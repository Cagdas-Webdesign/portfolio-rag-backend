"""What can go wrong while chunking a *valid* document.

Kept separate from :mod:`portfolio_rag.ingestion.errors` because the two answer
different questions:

* ingestion errors — "is this source document valid?"
* chunking errors — "can this valid document be cut into retrieval units under
  this policy without breaking something?"

Developer-facing, like ingestion errors: they name documents and headings, and
none of them is ever routed to an HTTP response.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class ChunkingErrorCode(StrEnum):
    """Stable, machine-readable reasons chunking refuses to produce output."""

    INVALID_CHUNKING_POLICY = "INVALID_CHUNKING_POLICY"
    UNSPLITTABLE_BLOCK = "UNSPLITTABLE_BLOCK"
    CHUNKING_INVARIANT_VIOLATION = "CHUNKING_INVARIANT_VIOLATION"


class ChunkingError(Exception):
    """Base for chunking failures."""

    code: ClassVar[ChunkingErrorCode] = ChunkingErrorCode.CHUNKING_INVARIANT_VIOLATION

    def __init__(
        self,
        message: str,
        *,
        document_id: str | None = None,
        heading_path: tuple[str, ...] = (),
    ) -> None:
        self.message = message
        self.document_id = document_id
        self.heading_path = heading_path
        super().__init__(message)

    def describe(self) -> str:
        """One-line rendering for developer output."""
        location = f" [{' > '.join(self.heading_path)}]" if self.heading_path else ""
        document = f" in {self.document_id}" if self.document_id else ""
        return f"{self.code.value}{document}{location}: {self.message}"


class InvalidChunkingPolicyError(ChunkingError):
    """The requested size budget is not coherent."""

    code = ChunkingErrorCode.INVALID_CHUNKING_POLICY


class UnsplittableBlockError(ChunkingError):
    """An atomic block exceeds ``max_chars`` and cannot be divided.

    A fenced code block is the realistic case. Cutting one would emit broken
    Markdown — an unterminated fence — into the index, so chunking stops and
    says which document and section is at fault. The fix belongs in the
    knowledge base (split the example), not in the chunker.
    """

    code = ChunkingErrorCode.UNSPLITTABLE_BLOCK


class ChunkingInvariantError(ChunkingError):
    """A guarantee this module makes about its own output did not hold.

    Never expected. It exists so that a future change which quietly breaks
    packing fails loudly instead of shipping oversized or duplicated units.
    """

    code = ChunkingErrorCode.CHUNKING_INVARIANT_VIOLATION
