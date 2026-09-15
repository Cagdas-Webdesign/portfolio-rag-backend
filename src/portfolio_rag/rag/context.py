"""Turning ranked passages into the bounded text a model is allowed to read.

A pure function of its inputs: no store, no provider, no filesystem, no clock,
no randomness. Same passages and same budget, same string, byte for byte —
which is what makes the golden tests in ``tests/unit/test_context_builder.py``
worth having, and what makes a context bug reproducible from the arguments
alone.

**The context representation is not the chunk content**, exactly as the
embedding representation is not the chunk content. ``chunk.content`` stays
source text; what a model sees is composed here, versioned here, and can change
here without touching the corpus:

    [SOURCE S1]
    Document: Integration Guide
    Section: Backend > APIs
    Content:
    REST endpoints are documented here.

**Labels are backend-owned.** ``S1`` is minted here, positionally, and means
nothing outside this one answer. It is deliberately not a document id, not a
chunk id and not a UUID: a model that can only say "S2" cannot invent a source,
and a label that carries no external meaning cannot leak one. The mapping back
to real provenance never leaves the process — see
:mod:`portfolio_rag.rag.citations`.

**Nothing is truncated.** A passage either fits whole or is left out and
counted. Half a paragraph of evidence reads exactly like whole evidence to a
model, and that is the one failure mode a context builder must not have.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.rag.errors import ContextBudgetError
from portfolio_rag.rag.tokens import estimate_tokens

#: Identifies how retrieved passages are rendered for a model. Recorded
#: alongside the prompt version so an evaluation run can say what it tested.
CONTEXT_REPRESENTATION_VERSION: Final = "grounded-context-v1"

#: Joins the heading path into one line: ``Backend > APIs > Authentication``.
#:
#: Deliberately its own constant rather than an import from
#: :mod:`portfolio_rag.ingestion.embedding`, which happens to use the same
#: string today. The two representations are versioned separately and must stay
#: free to diverge: sharing the constant would quietly couple what a model reads
#: at query time to what was embedded at index time.
HEADING_SEPARATOR: Final = " > "

#: Separates rendered sources.
_SOURCE_SEPARATOR: Final = "\n\n"


@dataclass(frozen=True, slots=True)
class ContextSource:
    """One passage as it appears in the context, under the label it was given."""

    label: str
    retrieved: RetrievedChunk
    text: str
    """The rendered block, exactly as it appears in the context."""

    @property
    def chunk_id(self) -> str:
        return self.retrieved.chunk.id


@dataclass(frozen=True, slots=True)
class GroundedContext:
    """The bounded knowledge one answer may be built from."""

    sources: tuple[ContextSource, ...]
    text: str
    estimated_tokens: int
    duplicates_removed: int
    """Passages dropped because an identical one was already included."""

    skipped_for_budget: int
    """Passages dropped because they did not fit in what was left."""

    @property
    def is_empty(self) -> bool:
        return not self.sources

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(source.label for source in self.sources)

    def source(self, label: str) -> ContextSource | None:
        """Look a label up. The only way a label ever becomes a citation."""
        for source in self.sources:
            if source.label == label:
                return source
        return None

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("representation", CONTEXT_REPRESENTATION_VERSION),
            ("sources", str(len(self.sources))),
            ("estimated tokens", str(self.estimated_tokens)),
            ("duplicates removed", str(self.duplicates_removed)),
            ("skipped for budget", str(self.skipped_for_budget)),
        )


def build_context(retrieved: Sequence[RetrievedChunk], *, available_tokens: int) -> GroundedContext:
    """Render *retrieved* into a context that fits *available_tokens*.

    Passages are taken in the order given — which is rank order — so the budget
    is spent on the strongest evidence first. A passage that does not fit is
    skipped rather than cut, and the ones after it are still tried: a long
    fourth passage should not cost a short fifth one its place.

    Raises :class:`~portfolio_rag.rag.errors.ContextBudgetError` when passages
    were offered and not one of them fits. An empty input is not an error — it
    is the "nothing was retrieved" case, and the caller decides what that means.
    """
    if available_tokens <= 0 and retrieved:
        raise ContextBudgetError

    sources: list[ContextSource] = []
    blocks: list[str] = []
    rendered_text = ""
    seen_chunk_ids: set[str] = set()
    seen_blocks: set[str] = set()
    duplicates = 0
    skipped = 0

    for candidate in retrieved:
        body = _render_body(candidate)
        if candidate.chunk.id in seen_chunk_ids or body in seen_blocks:
            # Exact duplicates only. Phase 3 gives neighbouring chunks a
            # deliberate overlap, and dropping passages for *resembling* each
            # other would throw away the text that overlap exists to preserve.
            duplicates += 1
            continue

        block = _render_block(f"S{len(sources) + 1}", body)
        candidate_text = _SOURCE_SEPARATOR.join([*blocks, block])
        if estimate_tokens(candidate_text) > available_tokens:
            skipped += 1
            continue

        seen_chunk_ids.add(candidate.chunk.id)
        seen_blocks.add(body)
        blocks.append(block)
        rendered_text = candidate_text
        sources.append(ContextSource(label=f"S{len(sources) + 1}", retrieved=candidate, text=block))

    if retrieved and not sources and skipped:
        raise ContextBudgetError

    return GroundedContext(
        sources=tuple(sources),
        text=rendered_text,
        estimated_tokens=estimate_tokens(rendered_text),
        duplicates_removed=duplicates,
        skipped_for_budget=skipped,
    )


def _render_body(retrieved: RetrievedChunk) -> str:
    """Everything about a passage except its label.

    Separated from the label so deduplication compares passages rather than
    positions: the same passage offered twice renders the same body and
    different labels.
    """
    chunk = retrieved.chunk
    lines = [f"Document: {chunk.document_metadata.title}"]
    if chunk.heading_path:
        lines.append(f"Section: {HEADING_SEPARATOR.join(chunk.heading_path)}")
    lines.append("Content:")
    lines.append(chunk.content)
    return "\n".join(lines)


def _render_block(label: str, body: str) -> str:
    return f"[SOURCE {label}]\n{body}"
