"""Embeddings and the vector index: the vocabulary of the corpus's last stage.

Three ideas live here, and keeping them apart is the point of the module.

**An embedding space** (:class:`EmbeddingSpec`) is *who produced a vector and
from what*: a provider, a model, a dimensionality, and the representation
version that decided which text was sent. Vectors are only comparable inside
one space. Note what this implies: two models that both output 1024 numbers do
not share a space. Dimensionality is a shape, not an identity, and treating it
as one is how a corpus quietly ends up with meaningless similarities.

**A vector record** (:class:`VectorRecord`) is one retrieval unit as it exists
in an index: its vector, the space it came from, the three fingerprints that
say whether it is current, and the metadata retrieval will filter on.

**Record state** (:class:`VectorRecordState`) is the same thing without the
vector — everything needed to decide what to do, and nothing more. Planning an
index update should not have to download every embedding it is not going to
change.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from portfolio_rag.domain.knowledge import (
    DocumentType,
    LanguageCode,
    Slug,
    TrustLevel,
    Visibility,
)

#: SHA-256 hex digest — the shape every fingerprint in this system has.
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def _reject_non_finite(values: tuple[float, ...]) -> tuple[float, ...]:
    """A NaN or an infinity poisons every similarity it ever takes part in.

    Providers do occasionally return them — on a degenerate input, or through a
    serialization bug. Rejecting at the boundary means the failure surfaces
    where it can be understood, rather than as inexplicably bad retrieval.
    """
    for position, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError(f"embedding contains a non-finite value at index {position}: {value}")
    return values


#: A dense embedding. A plain tuple keeps numeric libraries out of the core;
#: adapters convert to and from whatever their API speaks. Values are
#: deliberately unbounded — real models do not promise a [-1, 1] range.
EmbeddingVector = Annotated[
    tuple[float, ...],
    Field(min_length=1),
    AfterValidator(_reject_non_finite),
]


class SimilarityMetric(StrEnum):
    """How closeness between two vectors is measured."""

    COSINE = "cosine"


class EmbeddingSpec(BaseModel):
    """Identity of an embedding space.

    Two vectors may only be compared when they were produced under an *equal*
    spec. Equality is over every field, not just :attr:`dimensions`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(
        min_length=1,
        max_length=50,
        description="Which adapter produced the vector, e.g. `mistral`, `deterministic`.",
    )
    model: str = Field(
        min_length=1,
        max_length=200,
        description="Provider-specific model identifier, e.g. `mistral-embed`.",
    )
    dimensions: int = Field(gt=0, description="Length of every vector in this space.")
    representation_version: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Which rule turned a chunk into the text that was embedded, e.g. "
            "`embedding-text-v1`. Part of the space because the same model over "
            "differently composed text produces vectors that are not comparable."
        ),
    )

    @property
    def identity(self) -> str:
        """Short, stable, human-readable name for this space."""
        return f"{self.provider}:{self.model}:{self.dimensions}:{self.representation_version}"

    def is_compatible_with(self, other: EmbeddingSpec) -> bool:
        """Whether vectors from *other* may share an index with vectors from this space."""
        return self == other

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("provider", self.provider),
            ("model", self.model),
            ("dimensions", str(self.dimensions)),
            ("representation", self.representation_version),
        )


class VectorIndexSpec(BaseModel):
    """What an index holds and how it compares its contents.

    The metric lives here rather than being assumed by whoever happens to be
    computing a score: a store configured for cosine and a caller assuming dot
    product would disagree silently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    embedding: EmbeddingSpec
    metric: SimilarityMetric = SimilarityMetric.COSINE

    @property
    def dimensions(self) -> int:
        return self.embedding.dimensions

    def describe(self) -> tuple[tuple[str, str], ...]:
        return (*self.embedding.describe(), ("metric", self.metric.value))


class VectorMetadata(BaseModel):
    """The chunk attributes an index carries alongside a vector.

    Deliberately a chosen list rather than a dump of the domain model. Every
    field here is either something retrieval will filter on, or something a
    citation needs — and anything stored is something a vector store operator
    can read.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1, max_length=200)
    document_id: Slug
    document_title: str = Field(min_length=1, max_length=200)
    heading_path: tuple[str, ...] = ()
    source_path: str = Field(min_length=1, max_length=500)
    document_type: DocumentType
    language: LanguageCode
    visibility: Visibility
    trust_level: TrustLevel
    topics: tuple[str, ...] = ()
    technologies: tuple[str, ...] = ()


class VectorRecordState(BaseModel):
    """A record's identity and metadata, without its vector.

    This is what index planning compares. Fetching thousands of embeddings only
    to discover that nothing changed would be the wrong shape of question.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200, description="The chunk id this record holds.")
    spec: EmbeddingSpec
    embedding_fingerprint: Fingerprint = Field(
        description="Changes exactly when this unit needs a new vector."
    )
    chunk_fingerprint: Fingerprint = Field(
        description="Lineage: which structural retrieval unit this came from."
    )
    document_fingerprint: Fingerprint = Field(
        description="Lineage: which revision of which document this came from."
    )
    metadata: VectorMetadata


class VectorRecord(VectorRecordState):
    """A complete record: its state plus the embedding itself."""

    embedding: EmbeddingVector

    def state(self) -> VectorRecordState:
        """Project away the vector."""
        return VectorRecordState(
            id=self.id,
            spec=self.spec,
            embedding_fingerprint=self.embedding_fingerprint,
            chunk_fingerprint=self.chunk_fingerprint,
            document_fingerprint=self.document_fingerprint,
            metadata=self.metadata,
        )
