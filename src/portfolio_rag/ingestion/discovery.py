"""Finding the files that make up a knowledge base.

Deterministic by construction: the result is sorted by the document's relative
path, so two runs on the same tree produce the same order on any filesystem,
whatever order the OS happens to return directory entries in.

The exclusion rules are deliberately a short, fixed list rather than an ignore
file. A knowledge base is a curated directory of a few dozen documents; a
configurable ignore system would be more machinery than the problem has.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from portfolio_rag.core.logging import get_logger
from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode

_logger = get_logger(__name__)

#: The only file type accepted as an authored knowledge document.
MARKDOWN_SUFFIX: Final = ".md"

#: Documentation about the knowledge base, not knowledge itself.
_RESERVED_STEMS: Final[frozenset[str]] = frozenset({"readme"})


@dataclass(frozen=True, slots=True)
class DiscoveredDocument:
    """A candidate file, with the identity it will be known by."""

    path: Path
    """Absolute path, used to read the file."""

    relative_path: str
    """Path relative to the knowledge root, with forward slashes on every
    platform. This is what provenance records and what results sort by."""


def discover_documents(root: Path) -> list[DiscoveredDocument]:
    """Return every knowledge document under *root*, in a stable order.

    Raises :class:`DocumentIngestionError` when the root itself is unusable —
    a missing directory is a configuration problem, not a document problem, and
    reporting it as "0 documents, all valid" would be a lie.
    """
    if not root.exists():
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_KNOWLEDGE_ROOT,
            f"Knowledge root does not exist: {root}",
        )
    if not root.is_dir():
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_KNOWLEDGE_ROOT,
            f"Knowledge root is not a directory: {root}",
        )

    resolved_root = root.resolve()
    found = list(_walk(root, resolved_root))
    found.sort(key=lambda document: document.relative_path)
    return found


def _walk(root: Path, resolved_root: Path) -> Iterator[DiscoveredDocument]:
    # `Path.glob` does not descend into symlinked directories, so a symlink
    # loop cannot hang discovery and a linked directory cannot smuggle in a
    # whole tree from outside the root.
    for path in root.glob(f"**/*{MARKDOWN_SUFFIX}"):
        relative = path.relative_to(root)
        if _is_excluded(relative):
            continue
        if not path.is_file():
            continue
        if not _is_contained(path, resolved_root):
            _logger.warning(
                "skipping knowledge file that resolves outside the knowledge root",
                extra={"source_path": relative.as_posix()},
            )
            continue
        yield DiscoveredDocument(path=path, relative_path=relative.as_posix())


def _is_excluded(relative: Path) -> bool:
    """Apply the discovery rules to a path relative to the knowledge root."""
    for part in relative.parts:
        # Hidden files and directories: `.git`, `.obsidian`, editor state.
        if part.startswith("."):
            return True
        # Format artifacts: `_template.md`, `_drafts/`. Underscore means
        # "this is about the knowledge base, not part of it".
        if part.startswith("_"):
            return True
        # Editor backup files: `notes.md~`.
        if part.endswith("~"):
            return True
    return relative.stem.lower() in _RESERVED_STEMS


def _is_contained(path: Path, resolved_root: Path) -> bool:
    """Reject a symlinked file whose target lives outside the knowledge base."""
    try:
        return path.resolve().is_relative_to(resolved_root)
    except OSError:
        # Broken symlink, or a loop the OS refused to resolve.
        return False
