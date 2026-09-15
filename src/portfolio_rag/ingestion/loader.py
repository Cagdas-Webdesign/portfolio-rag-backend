"""The knowledge loader: the only way into the corpus.

Everything downstream receives :class:`KnowledgeDocument` objects from here
and never touches the filesystem, YAML, encodings or duplicate ids itself.
The messy part of the pipeline has exactly one place to live.

Two entry points, for two different callers:

* :func:`collect_knowledge_base` gathers everything it can and reports every
  problem it found. This is what a developer running ``knowledge validate``
  wants: fixing five documents needs five error messages, not the first one.
* :func:`load_knowledge_base` returns documents or raises. This is what a
  program wants: a partially valid corpus silently missing three documents
  would produce confidently incomplete answers later.

Neither holds state, caches, or touches globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_rag.domain.knowledge import DocumentProvenance, KnowledgeDocument
from portfolio_rag.ingestion.discovery import DiscoveredDocument, discover_documents
from portfolio_rag.ingestion.errors import (
    DocumentIngestionError,
    IngestionErrorCode,
    IngestionIssue,
    KnowledgeBaseError,
)
from portfolio_rag.ingestion.frontmatter import parse_frontmatter, split_frontmatter
from portfolio_rag.ingestion.metadata import validate_metadata
from portfolio_rag.ingestion.normalization import decode_source, normalize_body
from portfolio_rag.ingestion.provenance import compute_document_fingerprint


@dataclass(frozen=True, slots=True)
class KnowledgeBaseReport:
    """The outcome of one ingestion run: what loaded, and what did not."""

    root: Path
    documents: tuple[KnowledgeDocument, ...]
    issues: tuple[IngestionIssue, ...]

    @property
    def is_valid(self) -> bool:
        """True when every discovered document was ingested successfully."""
        return not self.issues

    @property
    def document_count(self) -> int:
        """Documents that were discovered, whether or not they were usable."""
        return len(self.documents) + len(self.issues)


def collect_knowledge_base(root: Path) -> KnowledgeBaseReport:
    """Ingest every document under *root*, collecting failures instead of raising."""
    try:
        discovered = discover_documents(root)
    except DocumentIngestionError as exc:
        return KnowledgeBaseReport(
            root=root,
            documents=(),
            issues=(_as_issue(exc, source_path=str(root)),),
        )

    documents: list[KnowledgeDocument] = []
    issues: list[IngestionIssue] = []
    # Document id -> path of the file that claimed it first. Discovery is
    # sorted, so "first" is deterministic rather than filesystem-dependent.
    claimed_ids: dict[str, str] = {}

    for candidate in discovered:
        try:
            document = _ingest_document(candidate)
        except DocumentIngestionError as exc:
            issues.append(_as_issue(exc, source_path=candidate.relative_path))
            continue

        document_id = document.metadata.id
        owner = claimed_ids.get(document_id)
        if owner is not None:
            issues.append(
                IngestionIssue(
                    code=IngestionErrorCode.DUPLICATE_DOCUMENT_ID,
                    source_path=candidate.relative_path,
                    message=f"Document id `{document_id}` is already used by another document.",
                    field="id",
                    reason=f"`{document_id}` first claimed by {owner}",
                )
            )
            continue

        claimed_ids[document_id] = candidate.relative_path
        documents.append(document)

    return KnowledgeBaseReport(root=root, documents=tuple(documents), issues=tuple(issues))


def load_knowledge_base(root: Path) -> tuple[KnowledgeDocument, ...]:
    """Return every document under *root*, or raise :class:`KnowledgeBaseError`.

    All-or-nothing on purpose: an assistant answering from a corpus that
    quietly lost three documents is worse than one that refuses to start.
    """
    report = collect_knowledge_base(root)
    if not report.is_valid:
        raise KnowledgeBaseError(report.issues)
    return report.documents


def _ingest_document(candidate: DiscoveredDocument) -> KnowledgeDocument:
    """Run one file through the whole pipeline.

    The file is read exactly once; every stage after that works on text already
    in memory.
    """
    try:
        source_bytes = candidate.path.read_bytes()
    except OSError as exc:
        raise DocumentIngestionError(
            IngestionErrorCode.UNREADABLE_SOURCE,
            "File could not be read.",
            reason=exc.strerror or type(exc).__name__,
        ) from exc

    text = decode_source(source_bytes)
    frontmatter_block, raw_body = split_frontmatter(text)
    metadata = validate_metadata(parse_frontmatter(frontmatter_block))
    content = normalize_body(raw_body)

    if not content:
        raise DocumentIngestionError(
            IngestionErrorCode.EMPTY_DOCUMENT,
            "Document has valid frontmatter but no content.",
        )

    return KnowledgeDocument(
        metadata=metadata,
        provenance=DocumentProvenance(
            source_path=candidate.relative_path,
            document_fingerprint=compute_document_fingerprint(metadata, content),
        ),
        content=content,
    )


def _as_issue(exc: DocumentIngestionError, *, source_path: str) -> IngestionIssue:
    return IngestionIssue(
        code=exc.code,
        source_path=source_path,
        message=exc.message,
        field=exc.field,
        reason=exc.reason,
    )
