"""Knowledge ingestion: the corpus side of the system.

Turns a directory of authored Markdown into validated
:class:`~portfolio_rag.domain.knowledge.KnowledgeDocument` objects:

    knowledge root
      → discovery      which files are knowledge documents at all
      → reading        bytes off disk, once
      → decoding       UTF-8, BOM, line endings, Unicode NFC
      → frontmatter    split the YAML block from the Markdown body
      → validation     schema version, then every authored field
      → normalization  trim file artifacts, never touch the prose
      → provenance     where it came from, and its canonical hash
      → conflicts      duplicate document ids across the whole base
      → KnowledgeDocument[]

Everything below is an implementation detail. Callers import from this module
and stay unaware of paths, YAML, encodings and delimiters — which is what lets
the chunker take a valid document for granted.

Chunking belongs to this package too: it is a property of the corpus, decided
once at ingestion time, not re-decided per query.

``ingestion`` may import from ``core`` and ``domain``. It is offline work and
is never reachable from an HTTP request handler.
"""

from portfolio_rag.ingestion.errors import (
    DocumentIngestionError,
    IngestionErrorCode,
    IngestionIssue,
    KnowledgeBaseError,
)
from portfolio_rag.ingestion.loader import (
    KnowledgeBaseReport,
    collect_knowledge_base,
    load_knowledge_base,
)
from portfolio_rag.ingestion.metadata import CURRENT_SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "DocumentIngestionError",
    "IngestionErrorCode",
    "IngestionIssue",
    "KnowledgeBaseError",
    "KnowledgeBaseReport",
    "collect_knowledge_base",
    "load_knowledge_base",
]
