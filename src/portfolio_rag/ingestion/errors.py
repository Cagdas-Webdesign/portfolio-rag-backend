"""What can go wrong while ingesting a knowledge base, and how it is reported.

Two shapes, deliberately kept small:

* :class:`DocumentIngestionError` — an exception raised by a pipeline stage the
  moment one document turns out to be unusable. Stages fail fast; they do not
  return sentinel values.
* :class:`IngestionIssue` — the same failure as data, with the file it happened
  in attached. This is what a batch run collects and prints.

These are developer-facing, not client-facing: they are free to name files and
fields, which API errors (``portfolio_rag.core.errors``) are not. That is why
they form their own taxonomy instead of extending ``AppError``.

Plain dataclasses rather than Pydantic models: nothing here validates untrusted
input, it only carries values the pipeline has already established.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IngestionErrorCode(StrEnum):
    """Stable, machine-readable reasons a document or a knowledge base fails."""

    INVALID_KNOWLEDGE_ROOT = "INVALID_KNOWLEDGE_ROOT"
    UNREADABLE_SOURCE = "UNREADABLE_SOURCE"
    INVALID_ENCODING = "INVALID_ENCODING"
    MISSING_FRONTMATTER = "MISSING_FRONTMATTER"
    INVALID_FRONTMATTER = "INVALID_FRONTMATTER"
    UNSUPPORTED_SCHEMA_VERSION = "UNSUPPORTED_SCHEMA_VERSION"
    INVALID_METADATA = "INVALID_METADATA"
    EMPTY_DOCUMENT = "EMPTY_DOCUMENT"
    DUPLICATE_DOCUMENT_ID = "DUPLICATE_DOCUMENT_ID"


class DocumentIngestionError(Exception):
    """One document cannot be ingested.

    Raised by a stage, caught by the loader, which knows the file path and
    turns it into an :class:`IngestionIssue`.
    """

    def __init__(
        self,
        code: IngestionErrorCode,
        message: str,
        *,
        field: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.field = field
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class IngestionIssue:
    """A failure, located in a specific file."""

    code: IngestionErrorCode
    source_path: str
    message: str
    field: str | None = None
    reason: str | None = None

    def describe(self) -> str:
        """One-line rendering: what is wrong, and where in the document."""
        detail = f"{self.field}: {self.reason}" if self.field and self.reason else None
        detail = detail or self.reason or self.message
        return f"{self.code.value}  {detail}"


class KnowledgeBaseError(Exception):
    """The knowledge base as a whole is not usable.

    Raised by :func:`portfolio_rag.ingestion.load_knowledge_base`, which refuses
    to hand back a partially valid corpus. Callers that want to see every
    problem at once use ``collect_knowledge_base`` instead and read
    :attr:`issues`.
    """

    def __init__(self, issues: tuple[IngestionIssue, ...]) -> None:
        self.issues = issues
        count = len(issues)
        noun = "document" if count == 1 else "documents"
        super().__init__(f"{count} {noun} could not be ingested")
