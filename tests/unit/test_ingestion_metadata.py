"""Schema-version gating and authored-field validation."""

from __future__ import annotations

from typing import Any

import pytest

from portfolio_rag.domain.knowledge import DocumentType, TrustLevel, Visibility
from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode
from portfolio_rag.ingestion.metadata import (
    CURRENT_SCHEMA_VERSION,
    SUPPORTED_SCHEMA_VERSIONS,
    validate_metadata,
)


def _frontmatter(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "id": "alpha",
        "title": "Test Document Alpha",
        "document_type": "reference",
        "language": "en",
        "topics": ["testing"],
        "technologies": ["Python"],
        "source": "fixture",
        "source_type": "authored",
        "version": 1,
        "updated_at": "2026-08-07",
    }
    payload.update(overrides)
    return {key: value for key, value in payload.items() if value is not ...}


def test_a_complete_frontmatter_validates():
    metadata = validate_metadata(_frontmatter())

    assert metadata.id == "alpha"
    assert metadata.document_type is DocumentType.REFERENCE
    assert metadata.topics == ("testing",)
    assert metadata.updated_at.isoformat() == "2026-08-07"


def test_optional_fields_fall_back_to_the_cautious_defaults():
    metadata = validate_metadata(_frontmatter())

    assert metadata.visibility is Visibility.INTERNAL
    assert metadata.trust_level is TrustLevel.UNVERIFIED
    assert metadata.license is None


def test_version_one_is_the_supported_schema_version():
    assert set(SUPPORTED_SCHEMA_VERSIONS) == {1}
    assert CURRENT_SCHEMA_VERSION == 1


def test_a_future_schema_version_is_refused_rather_than_guessed_at():
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(schema_version=2))

    assert caught.value.code is IngestionErrorCode.UNSUPPORTED_SCHEMA_VERSION
    assert caught.value.field == "schema_version"


def test_a_missing_schema_version_is_a_metadata_error():
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(schema_version=...))

    assert caught.value.code is IngestionErrorCode.INVALID_METADATA
    assert caught.value.field == "schema_version"


@pytest.mark.parametrize("declared", ["1", 1.0, True, None, [1]])
def test_a_non_integer_schema_version_is_rejected(declared: object):
    """`schema_version: true` is a YAML accident, not version 1."""
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(schema_version=declared))

    assert caught.value.code is IngestionErrorCode.INVALID_METADATA


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Alpha Document"),
        ("id", "../escape"),
        ("id", "topics/alpha"),
        ("id", "trailing-"),
        ("language", "english"),
        ("language", "EN"),
        ("document_type", "blogpost"),
        ("source_type", "invented"),
        ("visibility", "secret"),
        ("trust_level", "probably"),
        ("version", 0),
        ("version", "one"),
        ("updated_at", "07.08.2026"),
        ("updated_at", "2026-13-01"),
        ("title", ""),
        ("topics", "testing"),
    ],
)
def test_invalid_field_values_are_rejected_and_named(field: str, value: object):
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(**{field: value}))

    assert caught.value.code is IngestionErrorCode.INVALID_METADATA
    assert caught.value.field == field


@pytest.mark.parametrize("field", ["id", "title", "document_type", "language", "source"])
def test_missing_required_fields_are_rejected(field: str):
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(**{field: ...}))

    assert caught.value.code is IngestionErrorCode.INVALID_METADATA
    assert caught.value.field == field


def test_an_unknown_field_is_rejected_so_typos_cannot_hide():
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(technolgies=["Python"]))

    assert caught.value.code is IngestionErrorCode.INVALID_METADATA
    assert caught.value.field == "technolgies"


def test_several_invalid_fields_are_summarized_in_one_issue():
    """One issue per document; the author still learns there is more to fix."""
    with pytest.raises(DocumentIngestionError) as caught:
        validate_metadata(_frontmatter(language="english", version=0))

    assert caught.value.field == "language"
    assert "and 1 more invalid field" in (caught.value.reason or "")


def test_a_valid_document_type_from_the_extended_set_is_accepted():
    metadata = validate_metadata(_frontmatter(document_type="skill"))

    assert metadata.document_type is DocumentType.SKILL


def test_numeric_strings_are_not_silently_coerced_into_integers():
    """Pydantic strict-ish behaviour matters here: `version: "1"` is an authoring bug."""
    with pytest.raises(DocumentIngestionError):
        validate_metadata(_frontmatter(version="1"))
