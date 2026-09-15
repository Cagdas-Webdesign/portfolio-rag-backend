"""Validating a raw frontmatter mapping into :class:`DocumentMetadata`.

Two gates, in this order:

1. **Schema version.** Which frontmatter versions ingestion understands is a
   property of the *reader*, not of the model, so it lives here. A document
   declaring an unknown version is rejected rather than interpreted
   optimistically — silently ignoring a field that version 2 gave a meaning to
   is exactly the failure versioning exists to prevent.
2. **Field validation**, by the Pydantic model itself: types, enums, patterns,
   bounds, and no unknown keys.

There is no migration engine and no version 2 scaffolding. When a second
version genuinely exists, this is the one place that has to learn about it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import ValidationError

from portfolio_rag.domain.knowledge import DocumentMetadata
from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode

#: Version new documents should declare.
CURRENT_SCHEMA_VERSION: Final = 1

#: Versions this build can read. Kept as a set so that supporting two versions
#: side by side later is a one-line change rather than a redesign.
SUPPORTED_SCHEMA_VERSIONS: Final[frozenset[int]] = frozenset({CURRENT_SCHEMA_VERSION})

_SCHEMA_VERSION_FIELD: Final = "schema_version"


def validate_metadata(raw: Mapping[str, Any]) -> DocumentMetadata:
    """Validate authored frontmatter, or raise :class:`DocumentIngestionError`."""
    _require_supported_schema_version(raw)

    try:
        return DocumentMetadata.model_validate(dict(raw))
    except ValidationError as exc:
        raise _as_metadata_error(exc) from exc


def _require_supported_schema_version(raw: Mapping[str, Any]) -> None:
    if _SCHEMA_VERSION_FIELD not in raw:
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_METADATA,
            "Frontmatter is missing the schema version.",
            field=_SCHEMA_VERSION_FIELD,
            reason=f"required; the current version is {CURRENT_SCHEMA_VERSION}",
        )

    declared = raw[_SCHEMA_VERSION_FIELD]
    # `bool` is an `int` in Python; `schema_version: true` is a mistake, not a 1.
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_METADATA,
            "Schema version must be a whole number.",
            field=_SCHEMA_VERSION_FIELD,
            reason=f"must be a whole number, got {type(declared).__name__}",
        )

    if declared not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(str(version) for version in sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise DocumentIngestionError(
            IngestionErrorCode.UNSUPPORTED_SCHEMA_VERSION,
            f"Schema version {declared} is not supported by this build.",
            field=_SCHEMA_VERSION_FIELD,
            reason=f"{declared} is not supported; this build reads version {supported}",
        )


def _as_metadata_error(exc: ValidationError) -> DocumentIngestionError:
    """Report the first invalid field, and say how many others there are.

    One issue per document keeps batch output readable, and the author fixes
    the file in front of them either way. Pydantic reports errors in field
    order, so which one comes first is deterministic rather than incidental.
    """
    problems = exc.errors()
    first = problems[0]
    field = ".".join(str(part) for part in first["loc"]) or None

    reason = str(first["msg"])
    others = len(problems) - 1
    if others:
        reason = f"{reason} (and {others} more invalid field{'s' if others > 1 else ''})"

    return DocumentIngestionError(
        IngestionErrorCode.INVALID_METADATA,
        "Frontmatter failed validation.",
        field=field,
        reason=reason,
    )
