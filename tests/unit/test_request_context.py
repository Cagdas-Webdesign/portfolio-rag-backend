"""Correlation id handling, including the log-injection guard."""

from __future__ import annotations

import pytest

from portfolio_rag.core.request_context import (
    get_request_id,
    new_request_id,
    reset_request_id,
    sanitize_request_id,
    set_request_id,
)


def test_generated_ids_are_unique_and_well_formed():
    first, second = new_request_id(), new_request_id()

    assert first != second
    assert len(first) == 32
    assert sanitize_request_id(first) == first


@pytest.mark.parametrize(
    "supplied",
    ["client-request-42", "0123456789abcdef", "trace.id:1-2-3"],
)
def test_well_formed_client_ids_are_reused(supplied: str):
    assert sanitize_request_id(supplied) == supplied


@pytest.mark.parametrize(
    ("supplied", "reason"),
    [
        (None, "no header"),
        ("", "empty"),
        ("short", "below the minimum length"),
        ("x" * 65, "above the maximum length"),
        ("injected\nlevel=CRITICAL fake", "newline would forge a log line"),
        ('{"json": "payload"}', "characters outside the safe set"),
        ("../../etc/passwd", "path traversal characters"),
    ],
)
def test_unsafe_client_ids_are_replaced(supplied: str | None, reason: str):
    sanitized = sanitize_request_id(supplied)

    assert sanitized != supplied, reason
    assert len(sanitized) == 32


def test_the_id_is_scoped_to_its_context():
    assert get_request_id() is None

    token = set_request_id("request-under-test")
    try:
        assert get_request_id() == "request-under-test"
    finally:
        reset_request_id(token)

    assert get_request_id() is None
