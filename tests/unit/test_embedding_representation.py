"""What actually gets embedded — pinned to the character.

Golden tests, deliberately. The representation is the input to every vector in
the corpus; changing it silently would invalidate an index without anything
failing. If one of these assertions has to be edited, that edit *is* the
decision to bump the representation version.
"""

from __future__ import annotations

import pytest

from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.ingestion.chunking import chunk_document
from portfolio_rag.ingestion.embedding import (
    EMBEDDING_REPRESENTATION_VERSION,
    build_embedding_text,
    canonical_form,
    compute_embedding_fingerprint,
)
from tests.conftest import DocumentFactory

SPEC = EmbeddingSpec(
    provider="deterministic",
    model="sha256-derived-v1",
    dimensions=8,
    representation_version=EMBEDDING_REPRESENTATION_VERSION,
)


def _texts(document_factory: DocumentFactory, body: str, **kwargs: object) -> list[str]:
    return [
        build_embedding_text(chunk) for chunk in chunk_document(document_factory(body, **kwargs))
    ]


# --- the canonical shape ----------------------------------------------------


def test_the_representation_is_title_then_heading_path_then_content(
    make_document: DocumentFactory,
):
    body = "# Backend\n\n## Cloudflare\n\nIch verwende es für externe Systeme.\n"

    (text,) = _texts(make_document, body, title="API & Systemintegration")

    assert text == (
        "API & Systemintegration\n\nBackend > Cloudflare\n\nIch verwende es für externe Systeme."
    )


def test_content_above_the_first_heading_omits_the_heading_line(
    make_document: DocumentFactory,
):
    """An empty heading path is left out, not rendered as a blank line."""
    (text,) = _texts(make_document, "Just an opening paragraph.\n", title="Profile")

    assert text == "Profile\n\nJust an opening paragraph."


def test_a_document_without_headings_reads_as_title_then_body(
    make_document: DocumentFactory,
):
    (text,) = _texts(make_document, "First.\n\nSecond.\n", title="Notes")

    assert text == "Notes\n\nFirst.\n\nSecond."


def test_a_single_heading_level_is_the_whole_path(make_document: DocumentFactory):
    (text,) = _texts(make_document, "## REST\n\nBody.\n", title="APIs")

    assert text == "APIs\n\nREST\n\nBody."


def test_deep_heading_paths_are_joined_with_a_chevron(make_document: DocumentFactory):
    body = "# A\n\n## B\n\n### C\n\n#### D\n\nBody.\n"

    (text,) = _texts(make_document, body, title="Doc")

    assert text == "Doc\n\nA > B > C > D\n\nBody."


# --- what must not be in there ---------------------------------------------


def test_internal_metadata_is_never_embedded(make_document: DocumentFactory):
    """A provider sees title, headings and content — not the filing system."""
    body = "## Section\n\nThe body text.\n"
    (text,) = _texts(
        make_document,
        body,
        title="Doc",
        document_id="secret-doc",
        source_path="internal/secret-doc.md",
        visibility="internal",
        trust_level="authoritative",
        license="CC0-1.0",
        topics=["confidential"],
        technologies=["Vault"],
    )

    for forbidden in (
        "internal",
        "authoritative",
        "CC0-1.0",
        "secret-doc",
        "confidential",
        "Vault",
        "2026-08-07",
        "schema_version",
        "reference",
    ):
        assert forbidden not in text, f"{forbidden!r} must not reach an embedding provider"


def test_the_chunk_content_is_carried_through_verbatim(make_document: DocumentFactory):
    content = (
        "Text with **bold**, `code`, a [link](https://example.test) and a list:\n\n- one\n- two"
    )

    (text,) = _texts(make_document, f"## S\n\n{content}\n", title="Doc")

    assert text.endswith(content)


@pytest.mark.parametrize(
    "content",
    ["Grüße aus Köln über Straßen.", "これはテストです。", "Ünïcödé ✓ ★ →"],
)
def test_unicode_survives_unchanged(make_document: DocumentFactory, content: str):
    (text,) = _texts(make_document, f"## Ü\n\n{content}\n", title="Überschrift")

    assert text == f"Überschrift\n\nÜ\n\n{content}"


# --- determinism and versioning --------------------------------------------


def test_the_same_chunk_always_produces_the_same_text(make_document: DocumentFactory):
    body = "# A\n\n## B\n\nBody text.\n"

    assert _texts(make_document, body) == _texts(make_document, body)


def test_the_representation_version_is_defined_once_and_is_stable():
    assert EMBEDDING_REPRESENTATION_VERSION == "embedding-text-v1"


def test_the_same_text_and_space_fingerprint_the_same():
    assert compute_embedding_fingerprint("text", SPEC) == compute_embedding_fingerprint(
        "text", SPEC
    )


def test_a_fingerprint_is_a_sha256_hex_digest():
    digest = compute_embedding_fingerprint("text", SPEC)

    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_changed_text_changes_the_fingerprint():
    assert compute_embedding_fingerprint("text", SPEC) != compute_embedding_fingerprint(
        "other", SPEC
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", "mistral"),
        ("model", "other-model"),
        ("dimensions", 16),
        ("representation_version", "embedding-text-v2"),
    ],
)
def test_any_change_to_the_embedding_space_changes_the_fingerprint(field: str, value: object):
    """Including the representation version — even when the text is identical."""
    other = SPEC.model_copy(update={field: value})

    assert compute_embedding_fingerprint("text", other) != compute_embedding_fingerprint(
        "text", SPEC
    )


def test_the_canonical_form_carries_nothing_volatile():
    encoded = canonical_form("body text", SPEC).decode("utf-8")

    assert "/Users" not in encoded
    assert "2026" not in encoded
    assert "chunk_id" not in encoded


def test_a_changed_title_or_heading_changes_what_is_embedded(make_document: DocumentFactory):
    """Both are part of the representation, so both are part of the identity."""
    body = "## Original\n\nBody.\n"
    baseline = _texts(make_document, body, title="Original Title")[0]

    assert _texts(make_document, body, title="New Title")[0] != baseline
    assert _texts(make_document, "## Renamed\n\nBody.\n", title="Original Title")[0] != baseline
