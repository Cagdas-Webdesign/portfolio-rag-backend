"""Splitting and parsing the YAML frontmatter block."""

from __future__ import annotations

import pytest
import yaml

from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode
from portfolio_rag.ingestion.frontmatter import (
    StrictSafeLoader,
    parse_frontmatter,
    split_frontmatter,
)


def test_a_document_splits_into_frontmatter_and_body():
    frontmatter, body = split_frontmatter("---\nid: alpha\n---\n\n# Heading\n")

    assert frontmatter == "id: alpha"
    assert body == "\n# Heading\n"


def test_delimiters_inside_the_body_are_left_alone():
    """A horizontal rule further down must not truncate the document."""
    _, body = split_frontmatter("---\nid: alpha\n---\nintro\n\n---\n\noutro\n")

    assert "outro" in body


def test_a_document_without_frontmatter_is_rejected():
    with pytest.raises(DocumentIngestionError) as caught:
        split_frontmatter("# Just Markdown\n")

    assert caught.value.code is IngestionErrorCode.MISSING_FRONTMATTER


def test_an_unterminated_frontmatter_block_is_rejected():
    """Without this check the entire document would be swallowed as YAML."""
    with pytest.raises(DocumentIngestionError) as caught:
        split_frontmatter("---\nid: alpha\ntitle: No closing delimiter\n")

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER


def test_a_body_may_be_absent_from_the_split():
    frontmatter, body = split_frontmatter("---\nid: alpha\n---")

    assert frontmatter == "id: alpha"
    assert body == ""


def test_valid_yaml_becomes_a_mapping():
    parsed = parse_frontmatter("id: alpha\nversion: 2\ntopics:\n  - one\n  - two")

    assert parsed == {"id": "alpha", "version": 2, "topics": ["one", "two"]}


def test_broken_yaml_is_reported_with_a_compact_reason():
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("topics: [unclosed, list")

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER
    assert caught.value.reason is not None
    assert "\n" not in caught.value.reason


def test_an_empty_frontmatter_block_is_rejected():
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("\n   \n")

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER


def test_frontmatter_that_is_not_a_mapping_is_rejected():
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("- just\n- a\n- list")

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER


def test_non_string_field_names_are_rejected():
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("1: numeric key")

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER


# --- YAML must never be able to run code ------------------------------------


@pytest.mark.parametrize(
    ("payload", "description"),
    [
        ("id: !!python/object/apply:os.system ['echo pwned']", "arbitrary callable"),
        ("id: !!python/object/apply:subprocess.check_output [['id']]", "subprocess"),
        ("id: !!python/name:os.system ''", "name reference"),
        ("id: !!python/object:http.client.HTTPConnection {host: x}", "object construction"),
    ],
)
def test_python_specific_yaml_tags_are_refused(payload: str, description: str):
    """`safe_load` must reject these outright — no construction, no execution."""
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter(payload)

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER, description


def test_a_refused_tag_does_not_leave_a_constructed_object_behind():
    """Belt and braces: the failure is a parse error, not a partial result."""
    with pytest.raises(DocumentIngestionError):
        parse_frontmatter("evil: !!python/object/apply:builtins.eval ['1+1']")


def test_the_loader_is_a_safe_loader():
    """Guards the whole security posture of this module in one assertion."""
    assert issubclass(StrictSafeLoader, yaml.SafeLoader)


# --- duplicate keys ---------------------------------------------------------


def test_a_field_declared_twice_is_rejected_instead_of_last_one_winning():
    """YAML would silently keep `other-profile`; for authored metadata that is
    a data-loss bug, not a convenience."""
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("id: profile\nid: other-profile")

    assert caught.value.code is IngestionErrorCode.INVALID_FRONTMATTER
    assert caught.value.field == "id"
    assert caught.value.reason == "duplicate key: id"


def test_the_duplicate_is_reported_even_when_both_values_are_equal():
    with pytest.raises(DocumentIngestionError):
        parse_frontmatter("language: en\nlanguage: en")


def test_duplicates_are_rejected_in_nested_mappings_too():
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("outer:\n  inner: 1\n  inner: 2")

    assert caught.value.reason == "duplicate key: inner"


def test_duplicates_are_rejected_inside_list_items():
    with pytest.raises(DocumentIngestionError) as caught:
        parse_frontmatter("items:\n  - key: 1\n    key: 2")

    assert caught.value.reason == "duplicate key: key"


def test_repeating_a_value_in_a_list_is_still_fine():
    """Lists may legitimately contain the same entry twice — only keys are unique."""
    assert parse_frontmatter("topics:\n  - backend\n  - backend") == {
        "topics": ["backend", "backend"]
    }


def test_the_same_key_in_two_different_mappings_is_fine():
    parsed = parse_frontmatter("first:\n  name: a\nsecond:\n  name: b")

    assert parsed == {"first": {"name": "a"}, "second": {"name": "b"}}


def test_ordinary_frontmatter_still_parses_unchanged():
    """The duplicate-key check must not disturb the normal path."""
    parsed = parse_frontmatter(
        "schema_version: 1\nid: profile\ntopics:\n  - a\n  - b\nnested:\n  x: 1\n  y: 2"
    )

    assert parsed == {
        "schema_version": 1,
        "id": "profile",
        "topics": ["a", "b"],
        "nested": {"x": 1, "y": 2},
    }
