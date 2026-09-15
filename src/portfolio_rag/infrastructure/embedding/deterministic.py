"""A deterministic local embedding provider.

**This is not a semantic embedding model.** It does not understand text, and
similarity between two of its vectors means nothing about meaning. It exists so
that the entire indexing pipeline — representation, fingerprints, planning,
upserts, deletes, idempotency — can be developed, tested and demonstrated with
no API key, no network, no account and no cost.

What it does guarantee is what those tests need:

* **Determinism across processes.** Vectors are derived with SHA-256, not with
  Python's ``hash()`` (randomised per process) or ``random`` (needs seeding
  discipline nobody maintains). The same text produces the same vector on any
  machine, in any run, forever.
* **Distinguishable outputs.** Different text produces genuinely different
  vectors, so ordering, top-k and tie-breaking tests are meaningful. A provider
  returning zeros would make every similarity test vacuous.
* **Unit length.** Vectors are normalized, so cosine scores land in a familiar
  range and are easy to reason about in a failing assertion.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Final

from portfolio_rag.domain.embedding import EmbeddingSpec, EmbeddingVector
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingResult

PROVIDER_NAME: Final = "deterministic"
MODEL_NAME: Final = "sha256-derived-v1"

#: Small enough to keep test output readable, large enough that unrelated texts
#: do not collide into near-identical vectors.
DEFAULT_DIMENSIONS: Final = 256

_BYTES_PER_VALUE: Final = 4
_SCALE: Final = float(1 << (8 * _BYTES_PER_VALUE - 1))


class DeterministicEmbeddingProvider:
    """Derives a stable pseudo-embedding from the text itself."""

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._spec = EmbeddingSpec(
            provider=PROVIDER_NAME,
            model=MODEL_NAME,
            dimensions=dimensions,
            representation_version=EMBEDDING_REPRESENTATION_VERSION,
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed every input. No I/O, so batching changes nothing here."""
        return [EmbeddingResult(id=item.id, vector=self._vector_for(item.text)) for item in inputs]

    def _vector_for(self, text: str) -> EmbeddingVector:
        """Expand SHA-256 over the text into a unit vector of the right length.

        The counter is mixed into each round so that a vector longer than one
        digest keeps producing fresh bytes rather than repeating.
        """
        values: list[float] = []
        counter = 0
        while len(values) < self._spec.dimensions:
            digest = hashlib.sha256(f"{text}\x00{counter}".encode()).digest()
            for offset in range(0, len(digest), _BYTES_PER_VALUE):
                if len(values) == self._spec.dimensions:
                    break
                raw = int.from_bytes(digest[offset : offset + _BYTES_PER_VALUE], "big")
                values.append(raw / _SCALE - 1.0)
            counter += 1

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:  # pragma: no cover - needs 256 exact zeros from SHA-256
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)
