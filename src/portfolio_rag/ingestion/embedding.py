"""Turning a chunk into the text that actually gets embedded.

``KnowledgeChunk.content`` is source text and stays that way. What a model sees
is a separate, explicit decision made here — because a chunk in the middle of a
document is often unintelligible on its own:

    Ich verwende es für die Anbindung externer Systeme.

The same chunk under its title and heading path is a different question
entirely:

    API & Systemintegration

    Backend > Cloudflare

    Ich verwende es für die Anbindung externer Systeme.

**Version 1 is title, heading path, chunk content** — separated by blank lines,
in that order. Nothing else. Internal metadata (visibility, trust level, source
paths, fingerprints, ids, licences, dates) is deliberately *not* embedded: it is
structured data that retrieval filters on, and turning it into prose would both
pollute the vector and hand an external provider information it has no reason
to see.

Topics and technologies are also left out for now. They might help; nobody has
measured it, and version 2 is the honest place for an answer that evidence
supports.

Determinism is the contract: the same chunk under the same version always
produces the same text, byte for byte. No timestamps, no ids, no paths, no
provider-specific formatting — a provider adapter receives finished text and
never composes its own.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.domain.knowledge import KnowledgeChunk

#: Identifies how a chunk is turned into embeddable text. Part of
#: :class:`~portfolio_rag.domain.embedding.EmbeddingSpec`, so changing it makes
#: every existing vector recognisably stale rather than quietly incomparable.
#: Defined once, here.
EMBEDDING_REPRESENTATION_VERSION: Final = "embedding-text-v1"

#: Joins the heading path into one line: ``Backend > APIs > Authentication``.
HEADING_SEPARATOR: Final = " > "

#: Separates the parts of the representation.
_PART_SEPARATOR: Final = "\n\n"


def build_embedding_text(chunk: KnowledgeChunk) -> str:
    """Compose the canonical text for *chunk* under the current version.

    A chunk above the document's first heading has an empty heading path; the
    heading line is then omitted rather than rendered as an empty one.
    """
    parts = [chunk.document_metadata.title]
    if chunk.heading_path:
        parts.append(HEADING_SEPARATOR.join(chunk.heading_path))
    parts.append(chunk.content)
    return _PART_SEPARATOR.join(parts)


def canonical_form(text: str, spec: EmbeddingSpec) -> bytes:
    """Serialize what an embedding's identity is taken over.

    The text *and* the space that would produce it: the same words sent to a
    different model, or composed under a different representation version, are
    a different vector and must fingerprint differently.
    """
    payload = {
        "spec": {
            "dimensions": spec.dimensions,
            "model": spec.model,
            "provider": spec.provider,
            "representation_version": spec.representation_version,
        },
        "text": text,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return serialized.encode("utf-8")


def compute_embedding_fingerprint(text: str, spec: EmbeddingSpec) -> str:
    """SHA-256 over the canonical form.

    This is the value that decides whether a provider call is needed at all:
    an index already holding a record with this fingerprint already holds the
    vector for it.
    """
    return hashlib.sha256(canonical_form(text, spec)).hexdigest()
