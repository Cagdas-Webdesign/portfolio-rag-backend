"""ASGI application factory — the one place where the pieces are wired together.

Run locally::

    uv run uvicorn --app-dir src portfolio_rag.main:app --reload

Everything above this module is infrastructure (uvicorn, Docker, a future
Cloudflare deployment); everything below it is the portable application. That
boundary is what ADR 0003 protects.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from portfolio_rag.api.error_handlers import register_error_handlers
from portfolio_rag.api.middleware.gate import (
    DEFAULT_MAX_BODY_BYTES,
    BodySizeLimitMiddleware,
    EdgeAuthMiddleware,
)
from portfolio_rag.api.middleware.request_context import RequestContextMiddleware
from portfolio_rag.api.router import API_V1_PREFIX, api_v1_router, operational_router
from portfolio_rag.composition import ConfigurationError, build_query_components
from portfolio_rag.core.config import Settings, get_settings
from portfolio_rag.core.logging import configure_logging, get_logger
from portfolio_rag.core.request_context import REQUEST_ID_HEADER

#: What the guards protect: the versioned product API, which is where the
#: provider calls are. `/health` is deliberately outside it.
GUARDED_PREFIXES: Final = (API_V1_PREFIX,)

#: Ceiling on a chat request body. A question is a sentence, not an upload.
MAX_CHAT_BODY_BYTES: Final = DEFAULT_MAX_BODY_BYTES

_logger = get_logger(__name__)

API_DESCRIPTION = """
Backend for a retrieval-augmented knowledge assistant.

An answer is generated only from passages retrieved for the question, only from
documents marked public, and every citation refers to a document that was
actually retrieved — the backend maps citations itself and never publishes a
source a model merely named. When the knowledge base does not cover a question,
the response says so, cites nothing, and is still a `200`.

The surface is deliberately small: this endpoint and a health probe. Indexing,
retrieval inspection and evaluation are developer workflows reached through the
CLI, not over HTTP, and there is no conversation state — `conversation_id` is
echoed back and nothing more.

Every error response uses the same envelope and carries the `X-Request-ID`
correlation id that is also returned as a response header.
"""

OPENAPI_TAGS = [
    {"name": "System", "description": "Operational endpoints. Not versioned."},
    {"name": "Chat", "description": "Conversational access to the knowledge base."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the query stack once, and take it down cleanly.

    Reads the settings bound to *this* application rather than the process
    singleton, so an app built with explicit settings starts up under those
    settings and not under whatever the environment happens to say.

    Startup is where a misconfiguration is supposed to surface: an unreadable
    corpus, a missing credential or a development stand-in selected in
    production stops the process here, loudly, rather than turning into a
    strange answer for a stranger later.
    """
    settings: Settings = app.state.settings
    components = build_query_components(settings)
    await components.prepare()
    app.state.answer_service = components.answers

    _logger.info(
        "application started",
        extra={
            "environment": settings.environment.value,
            "version": settings.version,
            "cors_origins": len(settings.allowed_origins),
            "embedding_space": components.embeddings.spec.identity,
            "generation_model": components.llm.model,
            "corpus_chunks": len(components.chunks),
        },
    )
    try:
        yield
    finally:
        await components.aclose()
        _logger.info("application stopped")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    A factory rather than a module-level singleton so tests can build an app
    against explicit settings instead of mutating global state.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.service_name,
        version=settings.version,
        description=API_DESCRIPTION,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # Order matters, and `add_middleware` adds to the *outside*, so this reads
    # inside-out. What runs, from the outside in:
    #
    #   CORS            → so even a refusal carries the headers a browser needs
    #   request context → so every refusal below it has a correlation id
    #   edge auth       → a request that skipped the gateway stops here
    #   body size       → an oversized body stops here, unread by the route
    #   the routes
    #
    # Both guards are scoped to the expensive prefix; `/health` is outside it
    # and stays reachable for a platform probe that carries no secret.
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=MAX_CHAT_BODY_BYTES,
        protected_prefixes=GUARDED_PREFIXES,
    )
    if settings.edge_auth_required:
        # Not installed at all when the guard is off, rather than installed and
        # permissive: an absent middleware cannot be misconfigured into
        # accepting everything.
        if settings.edge_shared_secret is None:
            raise ConfigurationError(
                "Edge authentication is required in this environment but "
                "PORTFOLIO_RAG_EDGE_SHARED_SECRET is not set. Set it, or set "
                "PORTFOLIO_RAG_REQUIRE_EDGE_AUTH=false to serve without a gateway."
            )
        app.add_middleware(
            EdgeAuthMiddleware,
            secret=settings.edge_shared_secret.get_secret_value(),
            protected_prefixes=GUARDED_PREFIXES,
        )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
        max_age=600,
    )

    # The single source of truth for this instance. Routes reach it through
    # `get_app_settings`, the lifespan reads it directly.
    app.state.settings = settings

    register_error_handlers(app)

    app.include_router(operational_router)
    app.include_router(api_v1_router)

    return app


app = create_app()
