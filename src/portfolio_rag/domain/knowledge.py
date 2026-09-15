"""Knowledge documents and the chunks derived from them.

The field set of :class:`DocumentMetadata` is the machine-readable half of the
knowledge document standard described in ``knowledge/README.md``: every
document in ``knowledge/`` carries exactly these keys in its YAML frontmatter.
:mod:`portfolio_rag.ingestion` is what turns a file into a validated
:class:`KnowledgeDocument`; nothing else in the system parses YAML or reads the
knowledge directory.

Two kinds of metadata live here, and keeping them apart matters:

* :class:`DocumentMetadata` is **authored** — a human wrote it in the
  frontmatter and is accountable for it.
* :class:`DocumentProvenance` is **derived** — ingestion computed it from the
  file. It is never authored and never trusted from input.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints

#: Stable, human-readable, git-friendly identifier (e.g. ``about-me``).
Slug = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", min_length=2, max_length=100),
]

#: ISO 639-1 language code (e.g. ``en``, ``de``).
LanguageCode = Annotated[str, StringConstraints(pattern=r"^[a-z]{2}$")]


class DocumentType(StrEnum):
    """What kind of content a document holds. Drives prompting and filtering."""

    PROFILE = "profile"
    SKILL = "skill"
    PROJECT = "project"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    ARTICLE = "article"
    FAQ = "faq"
    REFERENCE = "reference"


class SourceType(StrEnum):
    """Where the content originally came from — part of its provenance."""

    AUTHORED = "authored"
    REPOSITORY = "repository"
    WEBSITE = "website"
    IMPORTED_FILE = "imported_file"


class Visibility(StrEnum):
    """Whether a document may be used to answer requests from public clients."""

    PUBLIC = "public"
    INTERNAL = "internal"


class TrustLevel(StrEnum):
    """Authored assessment of a document's reliability, preserved as metadata."""

    AUTHORITATIVE = "authoritative"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class DocumentMetadata(BaseModel):
    """Authored classification of a knowledge document, from its frontmatter.

    ``extra="forbid"``: a misspelled key must fail ingestion loudly rather than
    vanish and leave the author believing a document is tagged when it is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: StrictInt = Field(
        ge=1,
        description=(
            "Version of the frontmatter format this document is written against. "
            "Which versions ingestion accepts is decided in "
            "`portfolio_rag.ingestion.metadata`, not here."
        ),
    )
    id: Slug
    title: str = Field(min_length=1, max_length=200)
    document_type: DocumentType
    language: LanguageCode
    topics: tuple[str, ...] = ()
    technologies: tuple[str, ...] = ()
    source: str = Field(
        min_length=1,
        max_length=500,
        description="Where the content came from: a URL, a repository path or a file name.",
    )
    source_type: SourceType
    # Strict: YAML already yields a real integer for `version: 2`. Accepting
    # `"2"` or `true` here would only ever hide an authoring mistake.
    version: StrictInt = Field(ge=1, description="Bumped whenever the content changes materially.")
    updated_at: date
    visibility: Visibility = Visibility.INTERNAL
    trust_level: TrustLevel = TrustLevel.UNVERIFIED
    license: str | None = None


class DocumentProvenance(BaseModel):
    """Where a document came from and exactly which revision of it this is.

    Derived by ingestion, never authored. Everything in here is deterministic:
    the same file produces the same provenance on every machine and every run,
    which is what later phases will rely on to decide what actually changed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_path: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Path relative to the knowledge root, with forward slashes on every "
            "platform (e.g. `skills/api-integrations.md`)."
        ),
    )
    document_fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "SHA-256 over the canonical form of the whole document — its "
            "normalized metadata *and* its normalized body. Named a fingerprint "
            "rather than a content hash because it covers more than the body: "
            "changing `visibility` changes it too."
        ),
    )


class KnowledgeDocument(BaseModel):
    """A validated source document: what it says, and where it came from.

    Constructing one means the metadata validated, the body is non-empty and
    the provenance was computed — which is why later stages of the pipeline can
    take all three for granted.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: DocumentMetadata
    provenance: DocumentProvenance
    content: str = Field(min_length=1)


class ChunkProvenance(BaseModel):
    """How a chunk came to exist, and how to tell when it has changed.

    Derived by :mod:`portfolio_rag.ingestion.chunking`, never authored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    document: DocumentProvenance = Field(
        description="Provenance of the document this chunk was cut from, carried "
        "along so citations never have to reopen the source file."
    )
    strategy_version: str = Field(
        min_length=1,
        max_length=100,
        description=(
            "Identifies the algorithm *and* the size budget that produced this "
            "chunk (e.g. `markdown-structure-v1`). Chunks produced by different "
            "strategy versions are not comparable units."
        ),
    )
    fingerprint: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "SHA-256 over what this chunk *contains*: strategy, boundary-relevant "
            "policy, document id, heading path and content. Position is the "
            "chunk id's job, so the ordinal is deliberately not part of it."
        ),
    )
    section_ordinal: int = Field(
        ge=0,
        description=(
            "Which source section this chunk came from, counted in document "
            "order. Two sections that happen to share a heading path are still "
            "two sections — the path is a label, not an identity."
        ),
    )
    overlap_prefix_chars: int = Field(
        default=0,
        ge=0,
        description=(
            "How many leading characters of `content` are text repeated verbatim "
            "from the previous chunk of the same section: `content[:n]` is "
            "exactly that text. Counts repeated *source* characters only — the "
            "blank line joining it to the rest is not repeated content. "
            "0 means none."
        ),
    )


class KnowledgeChunk(BaseModel):
    """A retrievable slice of a document — the unit that later gets embedded.

    Document metadata is carried along (denormalized) because retrieval filters
    and citations need it at query time, when only the chunk is in hand.

    :attr:`content` is source text and nothing else. The heading path, the
    title and the metadata are kept *beside* it as data, never spliced into it:
    what a chunk says and how a later phase chooses to represent it for an
    embedding model are two different decisions, and only the first one belongs
    here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=200, description="Unique across the whole corpus.")
    document_id: Slug
    ordinal: int = Field(ge=0, description="Position of the chunk within its document.")
    content: str = Field(min_length=1)
    heading_path: tuple[str, ...] = Field(
        default=(),
        description=(
            "Enclosing headings from outermost to innermost, e.g. "
            "`('Backend', 'APIs', 'Authentication')`. Empty for content that "
            "appears before the document's first heading."
        ),
    )
    document_metadata: DocumentMetadata
    provenance: ChunkProvenance

    @property
    def section(self) -> str | None:
        """The innermost heading, or ``None`` above the first one."""
        return self.heading_path[-1] if self.heading_path else None
