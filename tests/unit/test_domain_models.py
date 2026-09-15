"""Knowledge and retrieval models shared across the pipeline."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from portfolio_rag.domain.knowledge import (
    ChunkProvenance,
    DocumentMetadata,
    DocumentProvenance,
    DocumentType,
    KnowledgeChunk,
    KnowledgeDocument,
    SourceType,
    TrustLevel,
    Visibility,
)
from portfolio_rag.domain.retrieval import RetrievedChunk, SourceCitation


def _metadata(**overrides: object) -> DocumentMetadata:
    payload: dict[str, object] = {
        "schema_version": 1,
        "id": "architecture-overview",
        "title": "Architecture Overview",
        "document_type": DocumentType.REFERENCE,
        "language": "en",
        "source": "docs/ARCHITECTURE.md",
        "source_type": SourceType.AUTHORED,
        "version": 1,
        "updated_at": date(2026, 8, 7),
    }
    return DocumentMetadata(**(payload | overrides))


def _provenance(**overrides: object) -> DocumentProvenance:
    payload: dict[str, object] = {
        "source_path": "architecture-overview.md",
        "document_fingerprint": "a" * 64,
    }
    return DocumentProvenance(**(payload | overrides))


def _chunk(**overrides: object) -> KnowledgeChunk:
    payload: dict[str, object] = {
        "id": "architecture-overview#0",
        "document_id": "architecture-overview",
        "ordinal": 0,
        "content": "The application is a modular monolith.",
        "heading_path": ("Architecture", "Module boundaries"),
        "document_metadata": _metadata(),
        "provenance": ChunkProvenance(
            document=_provenance(),
            strategy_version="markdown-structure-v1",
            fingerprint="b" * 64,
            section_ordinal=0,
        ),
    }
    return KnowledgeChunk(**(payload | overrides))


def test_metadata_defaults_are_the_cautious_ones():
    metadata = _metadata()

    assert metadata.visibility is Visibility.INTERNAL
    assert metadata.trust_level is TrustLevel.UNVERIFIED
    assert metadata.license is None
    assert metadata.topics == ()


@pytest.mark.parametrize(
    "document_id",
    ["Architecture", "with spaces", "trailing-", "a", "under_score", "ümlaut"],
)
def test_document_ids_must_be_git_friendly_slugs(document_id: str):
    with pytest.raises(ValidationError):
        _metadata(id=document_id)


def test_language_must_be_an_iso_639_1_code():
    assert _metadata(language="de").language == "de"
    with pytest.raises(ValidationError):
        _metadata(language="german")


def test_unknown_metadata_keys_are_rejected():
    """Frontmatter typos must fail ingestion instead of being silently dropped."""
    with pytest.raises(ValidationError):
        _metadata(documenttype="profile")


def test_version_starts_at_one():
    with pytest.raises(ValidationError):
        _metadata(version=0)


def test_documents_need_content():
    with pytest.raises(ValidationError):
        KnowledgeDocument(metadata=_metadata(), provenance=_provenance(), content="")


def test_documents_always_know_where_they_came_from():
    """Provenance is required: an untraceable document must not be constructible."""
    with pytest.raises(ValidationError):
        KnowledgeDocument(metadata=_metadata(), content="text")  # type: ignore[call-arg]


def test_chunk_ordinals_are_non_negative():
    assert _chunk(ordinal=3).ordinal == 3
    with pytest.raises(ValidationError):
        _chunk(ordinal=-1)


def test_domain_models_are_immutable():
    chunk = _chunk()

    with pytest.raises(ValidationError):
        chunk.content = "rewritten"  # type: ignore[misc]


def test_the_section_of_a_chunk_is_its_innermost_heading():
    assert _chunk().section == "Module boundaries"
    assert _chunk(heading_path=()).section is None


def test_a_citation_exposes_provenance_but_never_chunk_text():
    citation = SourceCitation.from_chunk(_chunk())

    assert citation.document_id == "architecture-overview"
    assert citation.title == "Architecture Overview"
    assert citation.source == "docs/ARCHITECTURE.md"
    assert citation.section == "Module boundaries"
    assert "content" not in citation.model_dump()


def test_a_retrieved_chunk_keeps_the_similarity_next_to_the_chunk():
    retrieved = RetrievedChunk(chunk=_chunk(), similarity=0.87)

    assert retrieved.similarity == pytest.approx(0.87)
    assert retrieved.chunk_id == retrieved.chunk.id
    assert retrieved.document_id == "architecture-overview"
    assert retrieved.chunk.document_metadata.trust_level is TrustLevel.UNVERIFIED
