"""Structured logging on top of the standard library — no external dependency.

Two formatters are available: JSON lines for deployed environments and a
readable single-line format for local development. Both automatically attach
the current request id, so every log line emitted while handling a request is
correlatable.

Privacy rule (see ``AGENTS.md``): chat messages, prompts, retrieved passages and
raw request bodies are **never** logged. Log identifiers, counts and sizes
instead. ``extra={...}`` fields are serialized as-is, so only put
non-sensitive values there.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

from portfolio_rag.core.config import Settings
from portfolio_rag.core.request_context import get_request_id

#: Attributes present on every ``LogRecord``; anything else was passed by the
#: caller via ``extra=`` and is treated as a structured field.
_STANDARD_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName", "request_id"}


def _structured_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value for key, value in record.__dict__.items() if key not in _STANDARD_RECORD_ATTRS
    }


class _RequestIdFilter(logging.Filter):
    """Attach the current correlation id to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects."""

    def __init__(self, *, service: str, environment: str) -> None:
        super().__init__()
        self._service = service
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self._service,
            "environment": self._environment,
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            payload["request_id"] = request_id

        payload.update(_structured_fields(record))

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Readable single-line output for local development."""

    def __init__(self) -> None:
        super().__init__(datefmt="%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None)
        prefix = f"[{request_id[:8]}] " if isinstance(request_id, str) else ""
        fields = " ".join(f"{k}={v}" for k, v in _structured_fields(record).items())
        line = (
            f"{self.formatTime(record, self.datefmt)} "
            f"{record.levelname:<8} {record.name} {prefix}{record.getMessage()}"
        )
        if fields:
            line = f"{line} | {fields}"
        if record.exc_info is not None:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def configure_logging(settings: Settings) -> None:
    """Install the root log handler. Safe to call more than once."""
    formatter: logging.Formatter = (
        JsonFormatter(service=settings.service_name, environment=settings.environment.value)
        if settings.use_json_logs
        else TextFormatter()
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level.value)

    # The application emits its own access log (with a request id) from
    # ``RequestContextMiddleware``; uvicorn's would only duplicate it.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn.error").propagate = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)
