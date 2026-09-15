"""The canonical fingerprint of a document.

A document's hash answers one question: *has anything that matters changed?*
"Matters" means the validated metadata and the normalized body — the two things
a later phase would have to re-chunk, re-embed and re-index if they changed.

What is deliberately **excluded** from the hash:

* the file's path — moving a document does not change what it says, and
  re-embedding a corpus because a directory was renamed would be waste;
* modification times, ingestion timestamps, machine names, random ids — none of
  them are properties of the document, and any of them would make two runs of
  the same input disagree.

Metadata is included on purpose, not just the body: changing ``visibility`` or
``trust_level`` changes how retrieval is allowed to use a document, so it has to
count as a change even when the prose is identical.
"""

from __future__ import annotations

import hashlib
import json

from portfolio_rag.domain.knowledge import DocumentMetadata

#: Separates metadata from body in the canonical form, so that no combination
#: of field values can be confused with the start of the body.
_SEPARATOR = b"\x00"


def canonical_form(metadata: DocumentMetadata, body: str) -> bytes:
    """Serialize a document to the exact bytes its hash is taken over.

    JSON with sorted keys is used rather than Python's own representation: it
    is stable across releases, and every value is already JSON-native after
    ``mode="json"`` (dates become ISO strings, enums their values).
    """
    serialized_metadata = json.dumps(
        metadata.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return serialized_metadata.encode("utf-8") + _SEPARATOR + body.encode("utf-8")


def compute_document_fingerprint(metadata: DocumentMetadata, body: str) -> str:
    """Return the SHA-256 hex digest of the document's canonical form."""
    return hashlib.sha256(canonical_form(metadata, body)).hexdigest()
