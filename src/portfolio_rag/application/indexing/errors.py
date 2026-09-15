"""Failures the indexing orchestration detects itself.

Distinct from :mod:`portfolio_rag.ports.errors`, which covers "the adapter
could not do its job". These are the ones the application discovers by
checking: a provider that answered, but not with what it promised; an index
that holds vectors from a different world; a synchronization run that did not
finish.

These errors are developer-facing. None is routed to an HTTP response because
indexing has no HTTP surface.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar


class IndexingErrorCode(StrEnum):
    """Stable, machine-readable reasons an indexing run stops."""

    EMBEDDING_VALIDATION_ERROR = "EMBEDDING_VALIDATION_ERROR"
    INCOMPATIBLE_EMBEDDING_SPACE = "INCOMPATIBLE_EMBEDDING_SPACE"
    INDEX_SYNCHRONIZATION_ERROR = "INDEX_SYNCHRONIZATION_ERROR"


class IndexingError(Exception):
    """Base for indexing failures."""

    code: ClassVar[IndexingErrorCode] = IndexingErrorCode.INDEX_SYNCHRONIZATION_ERROR

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def describe(self) -> str:
        """One-line rendering for developer output."""
        return f"{self.code.value}: {self.message}"


class EmbeddingValidationError(IndexingError):
    """A provider returned something that cannot be trusted as an embedding.

    Checked here rather than only in each adapter, because the guarantee has to
    hold for every provider — including one written later by someone who
    assumed the caller was not looking.
    """

    code = IndexingErrorCode.EMBEDDING_VALIDATION_ERROR


class IncompatibleEmbeddingSpaceError(IndexingError):
    """The index and the provider do not describe the same embedding space.

    Refusing is the safe answer. Writing the new vectors anyway would leave an
    index whose contents are individually valid and collectively meaningless —
    a failure that shows up as mediocre retrieval, months later, with no
    error anywhere.
    """

    code = IndexingErrorCode.INCOMPATIBLE_EMBEDDING_SPACE


class IndexSynchronizationError(IndexingError):
    """A synchronization run could not be completed."""

    code = IndexingErrorCode.INDEX_SYNCHRONIZATION_ERROR
