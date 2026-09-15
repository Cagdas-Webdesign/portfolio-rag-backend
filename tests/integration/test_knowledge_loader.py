"""The loader, exercised against real files on disk.

These are integration tests on purpose: encoding, line endings, directory
traversal and duplicate detection are exactly the things that behave
differently once a real filesystem is involved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.domain.knowledge import DocumentType, TrustLevel, Visibility
from portfolio_rag.ingestion import (
    IngestionErrorCode,
    KnowledgeBaseError,
    collect_knowledge_base,
    load_knowledge_base,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "knowledge"
VALID_ROOT = FIXTURES / "valid"
INVALID_ROOT = FIXTURES / "invalid"

VALID_DOCUMENT = """---
schema_version: 1
id: {id}
title: Fixture
document_type: reference
language: en
source: fixture
source_type: authored
version: 1
updated_at: 2026-08-07
---

Body text.
"""


def _write(root: Path, relative: str, text: str, *, encoding: str = "utf-8") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


# --- happy path -------------------------------------------------------------


def test_the_valid_fixture_base_loads_completely():
    documents = load_knowledge_base(VALID_ROOT)

    assert [document.metadata.id for document in documents] == ["alpha", "gamma", "beta"]


def test_documents_are_returned_in_discovery_order():
    """Sorted by path — `alpha.md`, `guides/gamma.md`, `topics/beta.md`."""
    documents = load_knowledge_base(VALID_ROOT)

    assert [document.provenance.source_path for document in documents] == [
        "alpha.md",
        "guides/gamma.md",
        "topics/beta.md",
    ]


def test_metadata_is_fully_parsed():
    alpha = load_knowledge_base(VALID_ROOT)[0]

    assert alpha.metadata.schema_version == 1
    assert alpha.metadata.title == "Test Document Alpha"
    assert alpha.metadata.document_type is DocumentType.REFERENCE
    assert alpha.metadata.language == "en"
    assert alpha.metadata.topics == ("testing", "ingestion")
    assert alpha.metadata.technologies == ("Python", "Markdown")
    assert alpha.metadata.visibility is Visibility.PUBLIC
    assert alpha.metadata.trust_level is TrustLevel.VERIFIED
    assert alpha.metadata.updated_at.isoformat() == "2026-08-07"


def test_the_body_is_the_markdown_without_the_frontmatter():
    alpha = load_knowledge_base(VALID_ROOT)[0]

    assert alpha.content.startswith("# Test Document Alpha")
    assert alpha.content.endswith("later phases have structure to work with.")
    assert "schema_version" not in alpha.content
    assert "---" not in alpha.content


def test_provenance_records_where_each_document_came_from():
    alpha = load_knowledge_base(VALID_ROOT)[0]

    assert alpha.provenance.source_path == "alpha.md"
    assert len(alpha.provenance.document_fingerprint) == 64


def test_umlauts_and_non_latin_characters_survive_unchanged():
    beta = next(d for d in load_knowledge_base(VALID_ROOT) if d.metadata.id == "beta")

    assert "äöü ÄÖÜ ß" in beta.content
    assert "漢字" in beta.content


def test_an_optional_field_is_preserved_when_present():
    beta = next(d for d in load_knowledge_base(VALID_ROOT) if d.metadata.id == "beta")

    assert beta.metadata.license == "CC0-1.0"


def test_documentation_template_and_hidden_files_are_not_ingested():
    ids = {document.metadata.id for document in load_knowledge_base(VALID_ROOT)}

    assert ids == {"alpha", "beta", "gamma"}
    assert "replace-me" not in ids
    assert "hidden-document" not in ids


# --- determinism ------------------------------------------------------------


def test_loading_twice_produces_identical_documents():
    first = load_knowledge_base(VALID_ROOT)
    second = load_knowledge_base(VALID_ROOT)

    assert first == second


def test_hashes_are_stable_across_runs():
    first = [d.provenance.document_fingerprint for d in load_knowledge_base(VALID_ROOT)]
    second = [d.provenance.document_fingerprint for d in load_knowledge_base(VALID_ROOT)]

    assert first == second


def test_crlf_and_lf_versions_of_a_document_are_indistinguishable(tmp_path: Path):
    unix = tmp_path / "unix"
    windows = tmp_path / "windows"
    body = VALID_DOCUMENT.format(id="alpha")
    _write(unix, "doc.md", body)
    _write(windows, "doc.md", body.replace("\n", "\r\n"))

    (loaded_unix,) = load_knowledge_base(unix)
    (loaded_windows,) = load_knowledge_base(windows)

    assert loaded_unix == loaded_windows
    assert "\r" not in loaded_unix.content


def test_a_byte_order_mark_does_not_change_the_document(tmp_path: Path):
    plain = tmp_path / "plain"
    with_bom = tmp_path / "with-bom"
    body = VALID_DOCUMENT.format(id="alpha")
    _write(plain, "doc.md", body)
    _write(with_bom, "doc.md", body, encoding="utf-8-sig")

    assert load_knowledge_base(plain) == load_knowledge_base(with_bom)


def test_trailing_newlines_do_not_change_the_document(tmp_path: Path):
    tight = tmp_path / "tight"
    padded = tmp_path / "padded"
    body = VALID_DOCUMENT.format(id="alpha")
    _write(tight, "doc.md", body)
    _write(padded, "doc.md", f"{body}\n\n\n")

    assert load_knowledge_base(tight) == load_knowledge_base(padded)


# --- error paths ------------------------------------------------------------


def test_the_invalid_fixture_base_reports_every_problem_at_once():
    report = collect_knowledge_base(INVALID_ROOT)

    assert not report.is_valid
    assert {issue.code for issue in report.issues} == {
        IngestionErrorCode.MISSING_FRONTMATTER,
        IngestionErrorCode.INVALID_FRONTMATTER,
        IngestionErrorCode.UNSUPPORTED_SCHEMA_VERSION,
        IngestionErrorCode.INVALID_METADATA,
        IngestionErrorCode.EMPTY_DOCUMENT,
        IngestionErrorCode.DUPLICATE_DOCUMENT_ID,
    }


def test_a_batch_run_does_not_stop_at_the_first_bad_document():
    report = collect_knowledge_base(INVALID_ROOT)

    assert len(report.issues) == 6
    assert report.document_count == 7


def test_every_issue_names_the_file_it_came_from():
    report = collect_knowledge_base(INVALID_ROOT)

    for issue in report.issues:
        assert issue.source_path.endswith(".md")
        assert issue.message


def test_valid_documents_alongside_broken_ones_are_still_collected():
    report = collect_knowledge_base(INVALID_ROOT)

    assert [document.metadata.id for document in report.documents] == ["contested-id"]


def test_the_first_claimant_of_an_id_wins_and_later_ones_are_flagged():
    report = collect_knowledge_base(INVALID_ROOT)
    duplicate = next(
        issue for issue in report.issues if issue.code is IngestionErrorCode.DUPLICATE_DOCUMENT_ID
    )

    assert duplicate.source_path == "06-duplicate.md"
    assert "00-claims-an-id.md" in (duplicate.reason or "")


def test_loading_an_invalid_base_raises_instead_of_returning_a_partial_corpus():
    with pytest.raises(KnowledgeBaseError) as caught:
        load_knowledge_base(INVALID_ROOT)

    assert len(caught.value.issues) == 6
    assert "6 documents could not be ingested" in str(caught.value)


def test_a_missing_knowledge_root_is_reported_as_an_issue(tmp_path: Path):
    report = collect_knowledge_base(tmp_path / "nowhere")

    assert not report.is_valid
    assert report.issues[0].code is IngestionErrorCode.INVALID_KNOWLEDGE_ROOT


def test_an_empty_knowledge_base_is_valid_and_empty(tmp_path: Path):
    report = collect_knowledge_base(tmp_path)

    assert report.is_valid
    assert report.documents == ()


def test_a_file_that_is_not_valid_utf8_is_reported(tmp_path: Path):
    (tmp_path / "broken.md").write_bytes(b"---\nid: alpha\n---\nGr\xfc\xdfe\n")

    report = collect_knowledge_base(tmp_path)

    assert report.issues[0].code is IngestionErrorCode.INVALID_ENCODING


def test_a_duplicate_id_across_directories_is_still_detected(tmp_path: Path):
    _write(tmp_path, "one/doc.md", VALID_DOCUMENT.format(id="shared"))
    _write(tmp_path, "two/doc.md", VALID_DOCUMENT.format(id="shared"))

    report = collect_knowledge_base(tmp_path)

    assert [issue.code for issue in report.issues] == [IngestionErrorCode.DUPLICATE_DOCUMENT_ID]
    assert report.issues[0].source_path == "two/doc.md"


def test_a_document_repeating_a_frontmatter_field_is_rejected(tmp_path: Path):
    """End to end: the duplicate-key rule reaches real files, not just the parser."""
    _write(
        tmp_path,
        "doubled.md",
        VALID_DOCUMENT.format(id="alpha").replace("id: alpha", "id: alpha\nid: beta"),
    )

    report = collect_knowledge_base(tmp_path)

    assert not report.is_valid
    assert report.issues[0].code is IngestionErrorCode.INVALID_FRONTMATTER
    assert report.issues[0].reason == "duplicate key: id"


def test_yaml_in_frontmatter_cannot_execute_code(tmp_path: Path):
    """A hostile knowledge file must fail to parse, not run anything."""
    marker = tmp_path / "pwned.txt"
    payload = (
        "id: !!python/object/apply:pathlib.Path.write_text "
        f"[!!python/object/apply:pathlib.Path ['{marker}'], 'pwned']"
    )
    _write(tmp_path, "hostile.md", f"---\nschema_version: 1\n{payload}\n---\n\nBody.\n")

    report = collect_knowledge_base(tmp_path)

    assert report.issues[0].code is IngestionErrorCode.INVALID_FRONTMATTER
    assert not marker.exists()
