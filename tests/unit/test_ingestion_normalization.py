"""Encoding, line endings and Unicode composition."""

from __future__ import annotations

import unicodedata

import pytest

from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode
from portfolio_rag.ingestion.normalization import (
    BYTE_ORDER_MARK,
    decode_source,
    normalize_body,
    normalize_text,
)


def test_crlf_and_lf_produce_identical_text():
    assert normalize_text("a\r\nb\r\nc") == normalize_text("a\nb\nc") == "a\nb\nc"


def test_lone_carriage_returns_are_normalized_too():
    assert normalize_text("a\rb") == "a\nb"


def test_a_byte_order_mark_is_removed():
    assert normalize_text(f"{BYTE_ORDER_MARK}---\nid: x") == "---\nid: x"


def test_only_a_leading_byte_order_mark_is_removed():
    """A BOM character mid-text is content, not a file artifact."""
    assert normalize_text(f"text{BYTE_ORDER_MARK}more").endswith(f"{BYTE_ORDER_MARK}more")


def test_decomposed_and_composed_umlauts_become_the_same_text():
    composed = "über"
    decomposed = unicodedata.normalize("NFD", composed)

    assert decomposed != composed
    assert normalize_text(decomposed) == normalize_text(composed) == composed


def test_non_utf8_bytes_are_rejected_rather_than_guessed():
    with pytest.raises(DocumentIngestionError) as caught:
        decode_source("Grüße".encode("latin-1"))

    assert caught.value.code is IngestionErrorCode.INVALID_ENCODING


def test_decoding_applies_the_full_normalization():
    decoded = decode_source(f"{BYTE_ORDER_MARK}a\r\nb".encode())

    assert decoded == "a\nb"


def test_body_normalization_trims_only_the_edges():
    assert normalize_body("\n\n# Heading\n\nText\n\n\n") == "# Heading\n\nText"


def test_body_normalization_keeps_indentation_of_the_first_line():
    """Four leading spaces start a Markdown code block — that is content."""
    assert normalize_body("\n    indented code\n") == "    indented code"


def test_body_normalization_keeps_hard_line_breaks_inside_the_text():
    """Two trailing spaces are a Markdown hard line break, not whitespace noise."""
    assert normalize_body("first  \nsecond") == "first  \nsecond"


def test_a_whitespace_only_body_normalizes_to_nothing():
    assert normalize_body("\n   \n\t\n") == ""
