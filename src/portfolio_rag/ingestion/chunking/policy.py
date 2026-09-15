"""The size budget that governs where chunks are cut.

Deliberately counted in **characters, not tokens**. Token counts are a property
of a particular model's tokenizer. Coupling chunk boundaries to one would make
the corpus a function of a vendor choice. Characters are provider-neutral,
exactly reproducible, and sufficiently precise to bound a retrieval unit.

The defaults remain *starting values*, not universal optima. Change them only
when evaluation demonstrates an improvement for the target corpus.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Identifies the chunking algorithm. Recorded on every chunk and folded into
#: its fingerprint, so a future change to how documents are cut is detectable
#: rather than silently producing incomparable units. Defined once, here.
MARKDOWN_CHUNKING_STRATEGY_VERSION: Final = "markdown-structure-v1"

DEFAULT_TARGET_CHARS: Final = 1200
DEFAULT_MAX_CHARS: Final = 1800
DEFAULT_OVERLAP_CHARS: Final = 150


class ChunkingPolicy(BaseModel):
    """Size budget for one chunking run.

    ``target_chars`` is the soft goal: packing stops adding blocks once a chunk
    has reached it. ``max_chars`` is a hard ceiling that no chunk may exceed —
    if honouring it is impossible, chunking fails rather than emits an
    oversized unit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_chars: int = Field(
        default=DEFAULT_TARGET_CHARS,
        gt=0,
        description="Soft goal. Packing stops adding blocks once a chunk reaches it.",
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS,
        gt=0,
        description="Hard ceiling. No chunk may exceed it, overlap included.",
    )
    overlap_chars: int = Field(
        default=DEFAULT_OVERLAP_CHARS,
        ge=0,
        description=(
            "Upper bound on text repeated from the previous chunk of the same "
            "section. 0 disables overlap entirely."
        ),
    )

    @model_validator(mode="after")
    def _check_budget_is_coherent(self) -> ChunkingPolicy:
        if self.max_chars < self.target_chars:
            raise ValueError(
                f"max_chars ({self.max_chars}) must be at least target_chars ({self.target_chars})"
            )
        if self.overlap_chars >= self.target_chars:
            # Overlap at or above the target would mean a chunk could carry more
            # repeated than new text, which defeats the point of splitting.
            raise ValueError(
                f"overlap_chars ({self.overlap_chars}) must be smaller than "
                f"target_chars ({self.target_chars})"
            )
        return self

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("strategy", MARKDOWN_CHUNKING_STRATEGY_VERSION),
            ("target chars", str(self.target_chars)),
            ("max chars", str(self.max_chars)),
            ("overlap chars", str(self.overlap_chars)),
        )


#: Used when no policy is supplied.
DEFAULT_CHUNKING_POLICY: Final = ChunkingPolicy()
