"""The canonical document hash: what changes it, and what must not."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from portfolio_rag.domain.knowledge import DocumentMetadata, TrustLevel, Visibility
from portfolio_rag.ingestion.provenance import canonical_form, compute_document_fingerprint

BODY = "# Heading\n\nSome content."


def _metadata(**overrides: Any) -> DocumentMetadata:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "id": "alpha",
        "title": "Test Document Alpha",
        "document_type": "reference",
        "language": "en",
        "topics": ["testing"],
        "technologies": ["Python"],
        "source": "fixture",
        "source_type": "authored",
        "version": 1,
        "updated_at": date(2026, 8, 7),
    }
    payload.update(overrides)
    return DocumentMetadata.model_validate(payload)


def test_the_hash_is_a_sha256_hex_digest():
    digest = compute_document_fingerprint(_metadata(), BODY)

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_the_same_document_hashes_the_same_every_time():
    assert compute_document_fingerprint(_metadata(), BODY) == compute_document_fingerprint(
        _metadata(), BODY
    )


def test_a_changed_body_changes_the_hash():
    assert compute_document_fingerprint(_metadata(), BODY) != compute_document_fingerprint(
        _metadata(), f"{BODY}!"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "A Different Title"),
        ("version", 2),
        ("updated_at", date(2026, 8, 8)),
        ("topics", ["testing", "extra"]),
        ("visibility", Visibility.PUBLIC),
        ("trust_level", TrustLevel.AUTHORITATIVE),
        ("license", "CC0-1.0"),
        ("schema_version", 2),
    ],
)
def test_a_changed_metadata_field_changes_the_hash(field: str, value: object):
    """Retrieval reads metadata, so a metadata change is a document change."""
    baseline = compute_document_fingerprint(_metadata(), BODY)

    assert compute_document_fingerprint(_metadata(**{field: value}), BODY) != baseline


def test_field_order_in_the_frontmatter_does_not_change_the_hash():
    """Reordering YAML keys is a cosmetic edit and must not force re-indexing."""
    forwards = DocumentMetadata.model_validate(
        {
            "schema_version": 1,
            "id": "alpha",
            "title": "Test Document Alpha",
            "document_type": "reference",
            "language": "en",
            "topics": ["testing"],
            "technologies": ["Python"],
            "source": "fixture",
            "source_type": "authored",
            "version": 1,
            "updated_at": date(2026, 8, 7),
        }
    )
    backwards = DocumentMetadata.model_validate(
        {
            "updated_at": date(2026, 8, 7),
            "version": 1,
            "source_type": "authored",
            "source": "fixture",
            "technologies": ["Python"],
            "topics": ["testing"],
            "language": "en",
            "document_type": "reference",
            "title": "Test Document Alpha",
            "id": "alpha",
            "schema_version": 1,
        }
    )

    assert compute_document_fingerprint(forwards, BODY) == compute_document_fingerprint(
        backwards, BODY
    )


def test_the_canonical_form_separates_metadata_from_body():
    """No field value can be crafted to look like the start of the body."""
    assert b"\x00" in canonical_form(_metadata(), BODY)


def test_moving_content_between_a_field_and_the_body_changes_the_hash():
    shifted_metadata = _metadata(title="Test Document Alpha extra")
    shifted_body = f"extra{BODY}"

    assert compute_document_fingerprint(shifted_metadata, BODY) != compute_document_fingerprint(
        _metadata(), shifted_body
    )


def test_the_canonical_form_contains_no_volatile_values():
    """Nothing machine-, time- or path-specific may end up in the digest."""
    encoded = canonical_form(_metadata(), BODY).decode("utf-8")

    assert "/Users" not in encoded
    assert "T00:00" not in encoded  # dates only, never timestamps
    assert encoded.count("2026-08-07") == 1
