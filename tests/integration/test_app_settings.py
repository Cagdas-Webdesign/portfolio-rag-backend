"""One application instance, one source of settings.

An app built with explicit settings must use *those* settings everywhere — in
its routes and in its lifespan. Reading the process-wide singleton in one place
and the injected settings in another is the kind of split-brain configuration
that only shows up in production.
"""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from portfolio_rag.core.config import Environment, LogLevel, Settings, get_settings
from portfolio_rag.main import create_app


class _Collector(logging.Handler):
    """Keeps the records themselves, so structured `extra` fields survive."""

    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__()
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


def _explicit_settings() -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.STAGING,
        log_level=LogLevel.WARNING,
        version="9.9.9-explicit",
        allowed_origins=["https://explicit.example"],
    )


def test_routes_answer_from_the_settings_the_app_was_built_with():
    settings = _explicit_settings()

    with TestClient(create_app(settings)) as client:
        payload = client.get("/health").json()

    assert payload["version"] == "9.9.9-explicit"
    assert payload["version"] != get_settings().version


def test_the_bound_settings_are_reachable_on_the_instance():
    settings = _explicit_settings()
    app = create_app(settings)

    assert app.state.settings is settings


def test_the_lifespan_reads_the_same_settings_as_the_routes():
    """Startup logs the environment; it must be the injected one.

    The handler is attached after `create_app`, because configuring logging is
    part of building the app and replaces the root handlers.
    """
    settings = _explicit_settings()
    app = create_app(settings)

    records: list[logging.LogRecord] = []
    handler = _Collector(records)
    logger = logging.getLogger("portfolio_rag.main")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with TestClient(app) as client:
            client.get("/health")
    finally:
        logger.removeHandler(handler)

    started = [record for record in records if record.getMessage() == "application started"]
    assert started
    assert started[0].environment == Environment.STAGING.value  # type: ignore[attr-defined]
    assert started[0].version == "9.9.9-explicit"  # type: ignore[attr-defined]


def test_two_apps_with_different_settings_do_not_interfere():
    first_settings = _explicit_settings()
    second_settings = Settings(
        _env_file=None,
        environment=Environment.LOCAL,
        version="1.1.1-other",
        allowed_origins=["https://other.example"],
    )

    with (
        TestClient(create_app(first_settings)) as first,
        TestClient(create_app(second_settings)) as second,
    ):
        assert first.get("/health").json()["version"] == "9.9.9-explicit"
        assert second.get("/health").json()["version"] == "1.1.1-other"


def test_an_app_built_without_settings_uses_the_process_defaults():
    get_settings.cache_clear()
    try:
        app = create_app()
        assert app.state.settings == get_settings()
    finally:
        get_settings.cache_clear()
