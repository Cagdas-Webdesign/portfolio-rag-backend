"""The error taxonomy and its mapping to HTTP."""

from __future__ import annotations

from portfolio_rag.api.error_handlers import _STATUS_BY_CODE
from portfolio_rag.core.errors import AppError, ErrorCode, UpstreamUnavailableError


def test_an_upstream_failure_carries_its_code_and_a_default_message():
    error = UpstreamUnavailableError()

    assert error.code is ErrorCode.INTERNAL_ERROR
    assert "temporarily unavailable" in error.message
    assert isinstance(error, AppError)


def test_a_custom_message_replaces_the_default():
    error = UpstreamUnavailableError("The knowledge search is temporarily unavailable.")

    assert error.message == "The knowledge search is temporarily unavailable."
    assert str(error) == "The knowledge search is temporarily unavailable."


def test_every_error_code_maps_to_an_http_status():
    """Guard: a new code must not be able to reach a client without a status."""
    assert set(_STATUS_BY_CODE) == set(ErrorCode)


def test_error_codes_are_stable_strings():
    """The wire format uses the code's value; renaming one is a breaking change."""
    assert ErrorCode.VALIDATION_ERROR.value == "VALIDATION_ERROR"
    assert ErrorCode.RETRIEVAL_UNAVAILABLE.value == "RETRIEVAL_UNAVAILABLE"
    assert ErrorCode.GENERATION_UNAVAILABLE.value == "GENERATION_UNAVAILABLE"
    assert ErrorCode.INTERNAL_ERROR.value == "INTERNAL_ERROR"
