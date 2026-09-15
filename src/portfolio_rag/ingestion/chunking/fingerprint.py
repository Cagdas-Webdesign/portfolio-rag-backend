"""Chunk identity: a readable id for *where*, a hash for *what*.

Two different questions, answered by two different values.

**The id** says where a unit sits: ``api-integrations--0003``. Document ids are
unique across the corpus and ordinals are gapless within a document, so chunk
ids are unique across the corpus without any registry. They are readable on
purpose — a chunk id in a log or a citation should be traceable by a human
without a lookup.

**The fingerprint** says what a unit contains and makes changes to chunking
output detectable. It covers:

* the *recipe* — strategy version and the policy values that decide boundaries;
* the *content* — document id, heading path, and the chunk text itself.

Two exclusions are deliberate:

* **The ordinal is not hashed.** Position is the id's job. Two chunks with
  identical text under an identical heading path are the same retrieval unit,
  and should be recognisable as such.
* **The document hash is not hashed**, even though every chunk carries it for
  lineage. Including it would make chunk 7's fingerprint change because a typo
  was fixed in chunk 2 — invalidating embeddings that did not change. The two
  hashes work at different levels: the document hash says "re-chunk this
  document", while the separately versioned embedding fingerprint decides
  which units need new vectors.

Nothing time-, path- or machine-dependent enters either value, so a second run
on the same input reproduces both exactly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from portfolio_rag.ingestion.chunking.policy import ChunkingPolicy

#: Separates document id from ordinal. Two dashes, because a single one is
#: legal inside a document slug and would make ids ambiguous to read.
ID_SEPARATOR: Final = "--"

#: Ordinals are zero-padded so ids sort lexicographically for small documents.
_ORDINAL_WIDTH: Final = 4


def build_chunk_id(document_id: str, ordinal: int) -> str:
    """``profile`` + 3 -> ``profile--0003``."""
    return f"{document_id}{ID_SEPARATOR}{ordinal:0{_ORDINAL_WIDTH}d}"


def canonical_form(
    *,
    strategy_version: str,
    policy: ChunkingPolicy,
    document_id: str,
    heading_path: tuple[str, ...],
    content: str,
) -> bytes:
    """Serialize a chunk to the exact bytes its fingerprint is taken over.

    JSON with sorted keys, matching the document-level canonicalization in
    :mod:`portfolio_rag.ingestion.provenance`: stable across releases, and
    unambiguous about where one field ends and the next begins.
    """
    payload = {
        "content": content,
        "document_id": document_id,
        "heading_path": list(heading_path),
        "policy": {
            "max_chars": policy.max_chars,
            "overlap_chars": policy.overlap_chars,
            "target_chars": policy.target_chars,
        },
        "strategy_version": strategy_version,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return serialized.encode("utf-8")


def compute_fingerprint(
    *,
    strategy_version: str,
    policy: ChunkingPolicy,
    document_id: str,
    heading_path: tuple[str, ...],
    content: str,
) -> str:
    """SHA-256 hex digest of the chunk's canonical form."""
    return hashlib.sha256(
        canonical_form(
            strategy_version=strategy_version,
            policy=policy,
            document_id=document_id,
            heading_path=heading_path,
            content=content,
        )
    ).hexdigest()
