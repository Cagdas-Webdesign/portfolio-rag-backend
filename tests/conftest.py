"""Shared fixtures.

Tests build the application through :func:`create_app` with explicit settings —
never through the module-level singleton — so nothing depends on the ambient
environment of the machine running them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from portfolio_rag.core.config import Environment, LogLevel, Settings
from portfolio_rag.domain.knowledge import (
    DocumentMetadata,
    DocumentProvenance,
    KnowledgeDocument,
)
from portfolio_rag.main import create_app

TEST_ORIGIN = "http://localhost:5173"

#: The neutral query-side corpus: two public documents and one internal one.
RAG_FIXTURE_ROOT = Path("tests/fixtures/knowledge/rag")

DocumentFactory = Callable[..., KnowledgeDocument]


@pytest.fixture
def make_document() -> DocumentFactory:
    """Build a valid `KnowledgeDocument` in memory.

    Chunking never touches a filesystem, so its tests should not either: the
    body is the interesting part and everything else gets a plausible default.
    """

    def _make(
        content: str,
        *,
        document_id: str = "sample",
        document_fingerprint: str | None = None,
        source_path: str | None = None,
        **metadata: Any,
    ) -> KnowledgeDocument:
        fields: dict[str, Any] = {
            "schema_version": 1,
            "id": document_id,
            "title": "Sample Document",
            "document_type": "reference",
            "language": "en",
            "source": "fixture",
            "source_type": "authored",
            "version": 1,
            "updated_at": date(2026, 8, 7),
        }
        fields.update(metadata)
        return KnowledgeDocument(
            metadata=DocumentMetadata.model_validate(fields),
            provenance=DocumentProvenance(
                source_path=source_path or f"{document_id}.md",
                document_fingerprint=document_fingerprint or "a" * 64,
            ),
            content=content,
        )

    return _make


@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment=Environment.LOCAL,
        log_level=LogLevel.WARNING,
        allowed_origins=[TEST_ORIGIN],
    )


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    # No dependency override needed: `create_app` binds these settings to the
    # instance, and everything in it reads them from there.
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    # The context manager runs the lifespan, so startup is exercised too.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def rag_settings(settings: Settings) -> Settings:
    """Settings for an application that can actually answer something.

    The corpus is the neutral fixture one, and the similarity threshold is
    dropped to zero because the shipped offline embedding provider derives
    vectors from SHA-256 and has no semantics to threshold — every score it
    produces is noise around zero. Retrieval quality is not what an HTTP test
    is measuring; the contract is.
    """
    return settings.model_copy(
        update={"knowledge_root": RAG_FIXTURE_ROOT, "retrieval_min_similarity": -1.0}
    )


@pytest.fixture
def rag_client(rag_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(rag_settings)) as test_client:
        yield test_client


@pytest.fixture
def failing_generation_client(rag_settings: Settings) -> Iterator[TestClient]:
    """A started application whose generation provider is not there.

    The swap happens after startup because startup is what builds the stack:
    the alternative is a second composition path that exists only for tests,
    and then the thing under test is no longer the thing that runs.
    """
    from tests.doubles import FailingLLMProvider

    app = create_app(rag_settings)
    with TestClient(app) as test_client:
        app.state.answer_service._llm = FailingLLMProvider()
        yield test_client


@pytest.fixture
def failing_store_client(rag_settings: Settings) -> Iterator[TestClient]:
    """A started application whose vector store cannot be searched."""
    from tests.doubles import FailingVectorStore

    app = create_app(rag_settings)
    with TestClient(app) as test_client:
        retrieval = app.state.answer_service._retrieval
        retrieval._store = FailingVectorStore(retrieval._store.index_spec)
        yield test_client
