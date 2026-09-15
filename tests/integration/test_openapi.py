"""The generated OpenAPI document must describe what exists — no more, no less.

These tests are the guard against documentation drifting away from the
implementation in either direction: promising a response the code cannot
produce, and hiding one it does.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from portfolio_rag.core.config import Settings


@pytest.fixture
def openapi(client: TestClient) -> dict[str, Any]:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    document: dict[str, Any] = response.json()
    return document


def test_the_document_is_labelled_with_the_running_build(
    openapi: dict[str, Any], settings: Settings
):
    assert openapi["info"]["title"] == "portfolio-rag-assistant"
    assert openapi["info"]["version"] == settings.version


def test_the_description_states_behaviour_rather_than_a_phase(openapi: dict[str, Any]):
    """A phase number in a public API description is a note to ourselves that
    goes stale the moment the phase ends. Describe what it does instead."""
    description = openapi["info"]["description"]

    assert "citation" in description
    assert not any(f"Phase {n}" in description for n in range(1, 10))


def test_only_the_implemented_routes_are_documented(openapi: dict[str, Any]):
    assert set(openapi["paths"]) == {"/health", "/api/v1/chat"}


def test_endpoints_are_tagged(openapi: dict[str, Any]):
    assert openapi["paths"]["/health"]["get"]["tags"] == ["System"]
    assert openapi["paths"]["/api/v1/chat"]["post"]["tags"] == ["Chat"]


def test_chat_documents_exactly_the_responses_it_can_produce(openapi: dict[str, Any]):
    responses = openapi["paths"]["/api/v1/chat"]["post"]["responses"]

    assert set(responses) == {"200", "403", "413", "422", "500", "503"}
    assert "501" not in responses, "the endpoint is implemented; the stub status is gone"
    # 403 and 413 come from the guards in `api/middleware/gate.py`; 403 only
    # where the gateway is configured, 413 always.


def test_the_answer_contract_is_now_the_documented_success_response(openapi: dict[str, Any]):
    schema = openapi["paths"]["/api/v1/chat"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]

    assert schema["$ref"].endswith("/ChatResponse")


def test_the_answer_schema_promises_only_what_the_route_returns(openapi: dict[str, Any]):
    assert set(openapi["components"]["schemas"]["ChatResponse"]["properties"]) == {
        "answer",
        "citations",
        "conversation_id",
    }


def test_the_citation_schema_exposes_only_public_fields(openapi: dict[str, Any]):
    properties = set(openapi["components"]["schemas"]["SourceCitation"]["properties"])

    assert properties == {"document_id", "title", "source", "section"}


def test_every_documented_failure_uses_the_shared_error_schema(openapi: dict[str, Any]):
    responses = openapi["paths"]["/api/v1/chat"]["post"]["responses"]

    for status_code in ("422", "500", "503"):
        schema = responses[status_code]["content"]["application/json"]["schema"]
        assert schema["$ref"].endswith("/ErrorResponse"), status_code


def test_the_error_codes_clients_may_branch_on_are_documented(openapi: dict[str, Any]):
    codes = set(openapi["components"]["schemas"]["ErrorCode"]["enum"])

    assert {"VALIDATION_ERROR", "RETRIEVAL_UNAVAILABLE", "GENERATION_UNAVAILABLE"} <= codes
    assert "NOT_IMPLEMENTED" not in codes, "nothing produces it any more"


def test_the_insufficient_answer_is_documented_as_a_success_not_an_error(openapi: dict[str, Any]):
    description = openapi["paths"]["/api/v1/chat"]["post"]["description"]

    assert "200" in description
    assert "citations" in description


def test_the_health_response_schema_is_a_real_model(openapi: dict[str, Any]):
    schema = openapi["paths"]["/health"]["get"]["responses"]["200"]["content"]["application/json"][
        "schema"
    ]

    assert schema["$ref"].endswith("/HealthResponse")
    assert set(openapi["components"]["schemas"]["HealthResponse"]["properties"]) == {
        "status",
        "service",
        "version",
    }


def test_no_internal_model_is_exposed_in_the_schema(openapi: dict[str, Any]):
    """Retrieval, context and prompt types are not part of the public contract."""
    exposed = set(openapi["components"]["schemas"])

    for internal in ("RetrievedChunk", "GroundedContext", "GenerationRequest", "VectorRecord"):
        assert internal not in exposed


def test_the_interactive_docs_are_served(client: TestClient):
    assert client.get("/docs").status_code == 200
