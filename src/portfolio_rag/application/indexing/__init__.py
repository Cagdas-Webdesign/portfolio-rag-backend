"""Indexing: keeping a vector index in step with the corpus.

    KnowledgeChunk[]
      → embedding representation   title + heading path + content
      → embedding fingerprint      what identifies this unit in this space
      → desired state              what the index ought to contain
      → index plan                 create / re-embed / metadata / unchanged / delete
      → embedding provider         only for what actually changed
      → vector store               upserts first, then deletes

Planning and execution are separate on purpose: a plan can be shown to a
developer, asserted in a test, and computed without spending a provider call.

``application.indexing`` works against ports only. It never imports an adapter,
never opens a file and never knows which provider it is talking to.
"""

from portfolio_rag.application.indexing.errors import (
    EmbeddingValidationError,
    IncompatibleEmbeddingSpaceError,
    IndexingError,
    IndexingErrorCode,
    IndexSynchronizationError,
)
from portfolio_rag.application.indexing.plan import (
    DesiredRecord,
    IndexPlan,
    build_desired_state,
    plan_index,
)
from portfolio_rag.application.indexing.service import IndexingResult, IndexingService

__all__ = [
    "DesiredRecord",
    "EmbeddingValidationError",
    "IncompatibleEmbeddingSpaceError",
    "IndexPlan",
    "IndexSynchronizationError",
    "IndexingError",
    "IndexingErrorCode",
    "IndexingResult",
    "IndexingService",
    "build_desired_state",
    "plan_index",
]
