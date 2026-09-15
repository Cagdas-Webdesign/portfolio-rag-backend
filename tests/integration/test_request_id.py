"""Request correlation across the whole request lifecycle."""

from __future__ import annotations

from fastapi.testclient import TestClient

from portfolio_rag.core.request_context import REQUEST_ID_HEADER

CHAT_URL = "/api/v1/chat"


def test_every_response_carries_a_request_id(client: TestClient):
    response = client.get("/health")

    assert response.headers[REQUEST_ID_HEADER]


def test_each_request_gets_its_own_id(client: TestClient):
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    assert first != second


def test_a_well_formed_client_id_is_reused_so_traces_join_up(client: TestClient):
    supplied = "client-trace-0001"

    response = client.get("/health", headers={REQUEST_ID_HEADER: supplied})

    assert response.headers[REQUEST_ID_HEADER] == supplied


def test_an_unsafe_client_id_is_replaced(client: TestClient):
    response = client.get("/health", headers={REQUEST_ID_HEADER: "bad id with spaces"})

    assert response.headers[REQUEST_ID_HEADER] != "bad id with spaces"


def test_the_error_body_references_the_same_id_as_the_header(client: TestClient):
    response = client.post(CHAT_URL, json={"message": "hi", "conversation_id": "not valid"})

    assert response.status_code == 422
    assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


def test_a_successful_answer_is_correlated_too(client: TestClient):
    """Correlation is not an error-path feature."""
    response = client.post(CHAT_URL, json={"message": "hi"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]


def test_validation_failures_are_correlated_too(client: TestClient):
    response = client.post(CHAT_URL, json={})

    assert response.status_code == 422
    assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]
