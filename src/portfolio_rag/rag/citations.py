"""Deciding which sources an answer is actually allowed to claim.

**The backend owns citation truth.** A model may point at a label; it may not
name a source. Every citation in a response is built from a
:class:`~portfolio_rag.domain.retrieval.RetrievedChunk` that this request
actually retrieved and actually put in the context — the model's contribution
is the selection, never the content. A document title, a path or a URL that a
model produced is text, not provenance, and is discarded here.

That is what makes a citation checkable rather than decorative: every one of
them can be traced back to a chunk id, a document fingerprint and a file in
``knowledge/``, and a label that maps to nothing is dropped rather than
rendered.

Three cases, each with a decided answer:

* **unknown label** — dropped, counted, logged. The alternative is publishing a
  reference to something that does not exist.
* **duplicate label** — collapsed to one. A model repeating itself is not
  evidence twice.
* **several passages, one document and section** — one citation. A reader wants
  the source, not the chunking. The granular mapping stays in
  :attr:`CitationOutcome.cited` for diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from portfolio_rag.core.logging import get_logger
from portfolio_rag.domain.retrieval import SourceCitation
from portfolio_rag.rag.context import ContextSource, GroundedContext

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CitationOutcome:
    """The verified sources, and what had to be rejected to get them."""

    citations: tuple[SourceCitation, ...]
    """Public, deduplicated, safe to return."""

    cited: tuple[ContextSource, ...]
    """The context sources behind them, in the order the model named them.
    Granular on purpose: this is the audit trail, not the response."""

    unknown_labels: tuple[str, ...]
    """Labels the model produced that were not in the context."""

    duplicate_labels: int

    @property
    def is_grounded(self) -> bool:
        """Whether at least one claim survived verification."""
        return bool(self.citations)


def resolve_citations(context: GroundedContext, labels: Sequence[str]) -> CitationOutcome:
    """Map claimed labels onto the passages that were really in *context*."""
    cited: list[ContextSource] = []
    unknown: list[str] = []
    seen_labels: set[str] = set()
    duplicates = 0

    for label in labels:
        if label in seen_labels:
            duplicates += 1
            continue
        source = context.source(label)
        if source is None:
            unknown.append(label)
            continue
        seen_labels.add(label)
        cited.append(source)

    if unknown:
        # The labels themselves are model output, not user content, and are
        # short and structural — logging how many is what matters, and a
        # provider inventing labels is worth noticing.
        _logger.warning("generation cited unknown sources", extra={"count": len(unknown)})

    return CitationOutcome(
        citations=_public_citations(cited),
        cited=tuple(cited),
        unknown_labels=tuple(unknown),
        duplicate_labels=duplicates,
    )


def _public_citations(cited: Sequence[ContextSource]) -> tuple[SourceCitation, ...]:
    """Project verified passages to what a client may see, without repeats."""
    citations: list[SourceCitation] = []
    seen: set[tuple[str, str | None]] = set()

    for source in cited:
        citation = source.retrieved.citation()
        key = (citation.document_id, citation.section)
        if key in seen:
            continue
        seen.add(key)
        citations.append(citation)
    return tuple(citations)
