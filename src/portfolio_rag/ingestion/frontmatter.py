"""Separating YAML frontmatter from the Markdown body.

The split is done here, by hand, because it is a five-line rule and a library
for it would be a dependency that mostly does something else. The YAML itself
is parsed by PyYAML — writing a YAML parser would be the opposite trade.

**Security.** Parsing goes through :class:`StrictSafeLoader`, which derives
from ``yaml.SafeLoader``. It constructs plain scalars, lists and mappings and
refuses tags such as ``!!python/object/apply``, so a knowledge file cannot cause
code to be executed by being ingested. Nothing in this package may parse YAML
with a loader that is not a ``SafeLoader`` subclass — a regression test asserts
that relationship.
"""

from __future__ import annotations

from typing import Any, Final

import yaml

from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode

#: A frontmatter block is opened and closed by this marker on its own line.
DELIMITER: Final = "---"

#: YAML parser messages can be long and multi-line; keep reports to one line.
_MAX_REASON_LENGTH: Final = 140


class DuplicateFrontmatterKeyError(yaml.YAMLError):
    """A mapping declares the same key twice.

    YAML itself permits this and lets the last value win, which is precisely
    the wrong behaviour for authored metadata: writing ``id`` twice is a
    mistake, and silently keeping one of the two values hides it.

    Derives from ``yaml.YAMLError`` so that any caller handling parse failures
    generically still handles this one.
    """

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate key: {key}")


class StrictSafeLoader(yaml.SafeLoader):
    """``SafeLoader`` that additionally rejects duplicate mapping keys.

    ``construct_mapping`` is called for every mapping node the parser builds,
    so nested mappings are covered as well as the top level.

    Public because it is the *only* YAML loader in this project: anything that
    parses YAML — frontmatter, the evaluation dataset — uses this one. A second
    loader defined elsewhere is how a non-safe one eventually gets used.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                if key in seen:
                    raise DuplicateFrontmatterKeyError(str(key))
                seen.add(key)
            except TypeError:
                # Unhashable key (a list or mapping used as a key). Leave the
                # complaint to PyYAML, which reports it precisely.
                continue
        return super().construct_mapping(node, deep=deep)


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split normalized document text into ``(frontmatter, body)``.

    The document must open with a delimiter line and close it later. Anything
    else is an error: a Markdown file without frontmatter is not a knowledge
    document, and one whose block is never closed is a typo that would
    otherwise silently swallow the whole document as YAML.
    """
    lines = text.split("\n")

    if not lines or lines[0].rstrip() != DELIMITER:
        raise DocumentIngestionError(
            IngestionErrorCode.MISSING_FRONTMATTER,
            "Document does not start with a `---` frontmatter block.",
        )

    for index in range(1, len(lines)):
        if lines[index].rstrip() == DELIMITER:
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])

    raise DocumentIngestionError(
        IngestionErrorCode.INVALID_FRONTMATTER,
        "Frontmatter block is never closed by a `---` line.",
    )


def parse_frontmatter(block: str) -> dict[str, Any]:
    """Parse a frontmatter block into a plain mapping."""
    try:
        parsed = yaml.load(block, Loader=StrictSafeLoader)  # noqa: S506 - SafeLoader subclass
    except DuplicateFrontmatterKeyError as exc:
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_FRONTMATTER,
            "Frontmatter declares the same field twice.",
            field=exc.key,
            reason=f"duplicate key: {exc.key}",
        ) from exc
    except yaml.YAMLError as exc:
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_FRONTMATTER,
            "Frontmatter is not valid YAML.",
            reason=_compact(exc),
        ) from exc

    if parsed is None:
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_FRONTMATTER,
            "Frontmatter block is empty.",
        )

    if not isinstance(parsed, dict):
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_FRONTMATTER,
            "Frontmatter must be a mapping of fields.",
            reason=f"parsed as {type(parsed).__name__}",
        )

    non_string_keys = sorted(str(key) for key in parsed if not isinstance(key, str))
    if non_string_keys:
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_FRONTMATTER,
            "Frontmatter field names must be strings.",
            reason=f"offending keys: {', '.join(non_string_keys)}",
        )

    return parsed


def _compact(exc: yaml.YAMLError) -> str:
    """Flatten a parser error into a single short line."""
    flattened = " ".join(str(exc).split())
    if len(flattened) > _MAX_REASON_LENGTH:
        return f"{flattened[:_MAX_REASON_LENGTH]}…"
    return flattened
