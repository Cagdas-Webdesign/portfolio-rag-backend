"""Structured log formatting."""

from __future__ import annotations

import json
import logging

from portfolio_rag.core.logging import JsonFormatter, TextFormatter
from portfolio_rag.core.request_context import reset_request_id, set_request_id


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="portfolio_rag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request completed",
        args=None,
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def _formatter() -> JsonFormatter:
    return JsonFormatter(service="portfolio-rag-assistant", environment="local")


def test_json_records_are_one_parsable_object_per_line():
    output = _formatter().format(_record())
    payload = json.loads(output)

    assert "\n" not in output
    assert payload["level"] == "INFO"
    assert payload["logger"] == "portfolio_rag.test"
    assert payload["message"] == "request completed"
    assert payload["service"] == "portfolio-rag-assistant"
    assert payload["timestamp"].endswith("+00:00")


def test_extra_fields_are_promoted_to_top_level_keys():
    payload = json.loads(_formatter().format(_record(http_status=501, duration_ms=1.5)))

    assert payload["http_status"] == 501
    assert payload["duration_ms"] == 1.5


def test_the_request_id_is_attached_exactly_once():
    record = _record()
    record.request_id = "abc123"

    payload = json.loads(_formatter().format(record))

    assert payload["request_id"] == "abc123"
    assert list(payload).count("request_id") == 1


def test_records_without_a_request_id_simply_omit_it():
    assert "request_id" not in json.loads(_formatter().format(_record()))


def test_exceptions_are_rendered_into_the_log_never_dropped():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    payload = json.loads(_formatter().format(record))

    assert "RuntimeError: boom" in payload["exception"]


def test_text_format_stays_on_one_line_and_shows_the_correlation_id():
    token = set_request_id("0123456789abcdef")
    try:
        record = _record(http_status=200)
        record.request_id = "0123456789abcdef"
        line = TextFormatter().format(record)
    finally:
        reset_request_id(token)

    assert "\n" not in line
    assert "[01234567]" in line
    assert "http_status=200" in line
