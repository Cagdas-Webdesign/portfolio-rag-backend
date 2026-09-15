"""Which files count as knowledge documents, and in what order."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from portfolio_rag.ingestion.discovery import discover_documents
from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode


def _touch(root: Path, relative: str, text: str = "content") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _relative_paths(root: Path) -> list[str]:
    return [document.relative_path for document in discover_documents(root)]


def test_discovery_recurses_into_subdirectories(tmp_path: Path):
    _touch(tmp_path, "alpha.md")
    _touch(tmp_path, "topics/beta.md")
    _touch(tmp_path, "topics/nested/gamma.md")

    assert _relative_paths(tmp_path) == ["alpha.md", "topics/beta.md", "topics/nested/gamma.md"]


def test_results_are_sorted_by_relative_path(tmp_path: Path):
    for name in ["zeta.md", "alpha.md", "middle/beta.md", "middle/alpha.md"]:
        _touch(tmp_path, name)

    assert _relative_paths(tmp_path) == [
        "alpha.md",
        "middle/alpha.md",
        "middle/beta.md",
        "zeta.md",
    ]


def test_ordering_does_not_depend_on_filesystem_order(tmp_path: Path):
    for name in ["c.md", "a.md", "b.md"]:
        _touch(tmp_path, name)

    assert _relative_paths(tmp_path) == _relative_paths(tmp_path) == ["a.md", "b.md", "c.md"]


def test_relative_paths_use_forward_slashes(tmp_path: Path):
    _touch(tmp_path, "topics/beta.md")

    assert "\\" not in _relative_paths(tmp_path)[0]


@pytest.mark.parametrize(
    ("relative", "why"),
    [
        ("README.md", "documentation about the knowledge base"),
        ("readme.md", "same, lower case"),
        ("_template.md", "format artifact"),
        ("_drafts/wip.md", "everything under an underscore directory"),
        (".hidden.md", "hidden file"),
        (".obsidian/config.md", "hidden directory"),
        ("notes.md~", "editor backup"),
        ("notes.txt", "not Markdown"),
        ("notes.markdown", "only `.md` in Phase 2"),
    ],
)
def test_non_knowledge_files_are_skipped(tmp_path: Path, relative: str, why: str):
    _touch(tmp_path, "real.md")
    _touch(tmp_path, relative)

    assert _relative_paths(tmp_path) == ["real.md"], why


def test_an_empty_knowledge_base_is_not_an_error(tmp_path: Path):
    assert _relative_paths(tmp_path) == []


def test_a_missing_root_is_reported_not_treated_as_empty(tmp_path: Path):
    with pytest.raises(DocumentIngestionError) as caught:
        discover_documents(tmp_path / "nope")

    assert caught.value.code is IngestionErrorCode.INVALID_KNOWLEDGE_ROOT


def test_a_root_that_is_a_file_is_rejected(tmp_path: Path):
    root = _touch(tmp_path, "not-a-directory.md")

    with pytest.raises(DocumentIngestionError) as caught:
        discover_documents(root)

    assert caught.value.code is IngestionErrorCode.INVALID_KNOWLEDGE_ROOT


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_a_symlink_pointing_outside_the_root_is_not_followed(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("must not be ingested", encoding="utf-8")

    root = tmp_path / "knowledge"
    root.mkdir()
    _touch(root, "real.md")
    (root / "escaped.md").symlink_to(outside / "secret.md")

    assert _relative_paths(root) == ["real.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_a_symlinked_directory_outside_the_root_is_not_traversed(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("must not be ingested", encoding="utf-8")

    root = tmp_path / "knowledge"
    root.mkdir()
    _touch(root, "real.md")
    (root / "linked").symlink_to(outside, target_is_directory=True)

    assert _relative_paths(root) == ["real.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_a_symlink_staying_inside_the_root_is_still_ingested(tmp_path: Path):
    root = tmp_path / "knowledge"
    root.mkdir()
    _touch(root, "sources/real.md")
    (root / "alias.md").symlink_to(root / "sources/real.md")

    assert _relative_paths(root) == ["alias.md", "sources/real.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_a_broken_symlink_is_skipped_not_crashed_on(tmp_path: Path):
    root = tmp_path / "knowledge"
    root.mkdir()
    _touch(root, "real.md")
    (root / "dangling.md").symlink_to(root / "does-not-exist.md")

    assert _relative_paths(root) == ["real.md"]
