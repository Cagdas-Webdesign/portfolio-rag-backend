"""`GET /health` behaviour against the real application."""

from __future__ import annotations

from fastapi.testclient import TestClient

from portfolio_rag.api.schemas.health import HealthResponse
from portfolio_rag.core.config import Settings


def test_health_reports_ok(client: TestClient, settings: Settings):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "portfolio-rag-assistant",
        "version": settings.version,
    }


def test_health_matches_its_response_schema(client: TestClient):
    payload = HealthResponse.model_validate(client.get("/health").json())

    assert payload.status == "ok"


def test_health_is_not_versioned(client: TestClient):
    """Uptime checks must not have to follow API version changes."""
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/health").status_code == 404
