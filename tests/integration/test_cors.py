"""CORS is configured, not wide open."""

from __future__ import annotations

from fastapi.testclient import TestClient

from portfolio_rag.api.middleware.gate import EDGE_AUTH_HEADER
from portfolio_rag.core.config import Settings
from portfolio_rag.core.request_context import REQUEST_ID_HEADER


def test_a_configured_origin_is_allowed(client: TestClient, settings: Settings):
    origin = settings.allowed_origins[0]
    response = client.get("/health", headers={"Origin": origin})

    assert response.headers["access-control-allow-origin"] == origin


def test_an_unconfigured_origin_gets_no_cors_grant(client: TestClient):
    response = client.get("/health", headers={"Origin": "https://not-configured.example"})

    assert "access-control-allow-origin" not in response.headers


def test_the_response_never_grants_a_wildcard(client: TestClient, settings: Settings):
    response = client.get("/health", headers={"Origin": settings.allowed_origins[0]})

    assert response.headers.get("access-control-allow-origin") != "*"


def test_credentials_are_not_allowed_cross_origin(client: TestClient, settings: Settings):
    """No cookie- or credential-based auth exists, so none is granted."""
    response = client.get("/health", headers={"Origin": settings.allowed_origins[0]})

    assert "access-control-allow-credentials" not in response.headers


def test_the_preflight_permits_the_correlation_header(client: TestClient, settings: Settings):
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": settings.allowed_origins[0],
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": f"content-type,{REQUEST_ID_HEADER}",
        },
    )

    assert response.status_code == 200
    allowed = response.headers["access-control-allow-headers"].lower()
    assert "content-type" in allowed
    assert REQUEST_ID_HEADER.lower() in allowed


def test_the_correlation_header_is_readable_by_the_browser(client: TestClient, settings: Settings):
    response = client.get("/health", headers={"Origin": settings.allowed_origins[0]})

    assert REQUEST_ID_HEADER.lower() in response.headers["access-control-expose-headers"].lower()


def test_the_preflight_does_not_permit_the_edge_header(client: TestClient, settings: Settings):
    """A browser must never be able to send the gateway's header.

    It is set server-side by the edge worker and nowhere else. Allowing it
    through CORS would invite a page to try supplying one, and would make the
    guard look like something a client participates in.
    """
    response = client.options(
        "/api/v1/chat",
        headers={
            "Origin": settings.allowed_origins[0],
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": EDGE_AUTH_HEADER,
        },
    )

    allowed = response.headers.get("access-control-allow-headers", "")
    assert EDGE_AUTH_HEADER.lower() not in allowed.lower()
