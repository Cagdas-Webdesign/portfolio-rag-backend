"""Every failure leaves the application in the same envelope — and leaks nothing."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from portfolio_rag.api.schemas.errors import ErrorResponse
from portfolio_rag.core.request_context import REQUEST_ID_HEADER

INTERNAL_CONNECTION_STRING = "postgresql://user:hunter2@internal-db:5432/app"


def test_unknown_routes_use_the_error_envelope(client: TestClient):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "NOT_FOUND"
    assert body.error.message == "The requested resource does not exist."


def test_an_unhandled_exception_is_reported_without_any_internal_detail(app: FastAPI):
    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError(f"connection to {INTERNAL_CONNECTION_STRING} failed")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom")

    assert response.status_code == 500
    body = ErrorResponse.model_validate(response.json())
    assert body.error.code == "INTERNAL_ERROR"
    assert body.error.message == "An unexpected error occurred."
    assert body.error.request_id is not None

    # Neither the exception text nor a traceback may reach the client.
    assert INTERNAL_CONNECTION_STRING not in response.text
    assert "Traceback" not in response.text
    assert "RuntimeError" not in response.text


def test_even_a_500_stays_correlated(app: FastAPI):
    @app.get("/boom-correlated")
    async def boom() -> None:
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/boom-correlated")

    assert response.headers[REQUEST_ID_HEADER] == response.json()["error"]["request_id"]


def test_error_bodies_never_contain_a_details_key_unless_it_is_a_validation_error(
    client: TestClient,
):
    payload = client.get("/does-not-exist").json()

    assert "details" not in payload["error"]
