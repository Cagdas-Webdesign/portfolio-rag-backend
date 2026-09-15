"""The developer CLI, driven exactly as a shell would drive it."""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.cli import EXIT_INTERNAL_ERROR, EXIT_INVALID, EXIT_OK, main

FIXTURES = Path(__file__).parent.parent / "fixtures" / "knowledge"
VALID_ROOT = FIXTURES / "valid"
INVALID_ROOT = FIXTURES / "invalid"


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --- validate ---------------------------------------------------------------


def test_validate_succeeds_on_a_valid_knowledge_base(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "validate", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "3 documents" in out
    assert "3 valid" in out
    assert "0 errors" in out


def test_validate_lists_every_document_it_accepted(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "validate", "--root", str(VALID_ROOT))

    assert "✓ alpha.md" in out
    assert "✓ guides/gamma.md" in out
    assert "✓ topics/beta.md" in out


def test_validate_shows_the_knowledge_root_it_used(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "validate", "--root", str(VALID_ROOT))

    assert str(VALID_ROOT) in out


def test_validate_fails_on_a_broken_knowledge_base(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "validate", "--root", str(INVALID_ROOT))

    assert code == EXIT_INVALID
    assert "6 errors" in out
    assert "1 valid" in out


def test_validate_explains_each_failure(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "validate", "--root", str(INVALID_ROOT))

    assert "✗ 01-missing-frontmatter.md" in out
    assert "MISSING_FRONTMATTER" in out
    assert "UNSUPPORTED_SCHEMA_VERSION" in out
    assert "DUPLICATE_DOCUMENT_ID" in out
    assert "language:" in out  # the offending field is named


def test_validate_output_is_ordered_by_path(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "validate", "--root", str(INVALID_ROOT))
    listed = [line[2:] for line in out.splitlines() if line[:1] in {"✓", "✗"}]

    assert listed == sorted(listed)


def test_validate_never_prints_a_traceback(capsys: pytest.CaptureFixture[str]):
    _, out, err = _run(capsys, "knowledge", "validate", "--root", str(INVALID_ROOT))

    assert "Traceback" not in out + err
    assert 'File "' not in out + err


def test_validate_reports_a_missing_root_as_a_failure(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    code, out, _ = _run(capsys, "knowledge", "validate", "--root", str(tmp_path / "nowhere"))

    assert code == EXIT_INVALID
    assert "INVALID_KNOWLEDGE_ROOT" in out


def test_validate_accepts_an_empty_knowledge_base(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    code, out, _ = _run(capsys, "knowledge", "validate", "--root", str(tmp_path))

    assert code == EXIT_OK
    assert "0 documents" in out


def test_singular_and_plural_are_both_readable(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    (tmp_path / "solo.md").write_text("no frontmatter", encoding="utf-8")

    _, out, _ = _run(capsys, "knowledge", "validate", "--root", str(tmp_path))

    assert "1 document\n" in out
    assert "1 error" in out
    assert "1 errors" not in out


# --- inspect ----------------------------------------------------------------


def test_inspect_finds_a_document_by_id(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "inspect", "alpha", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "ID             alpha" in out
    assert "Title          Test Document Alpha" in out
    assert "Schema         1" in out
    assert "Type           reference" in out
    assert "Language       en" in out
    assert "Trust          verified" in out
    assert "Topics         testing, ingestion" in out
    assert "Technologies   Python, Markdown" in out


def test_inspect_finds_a_document_by_path(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "inspect", "topics/beta.md", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "ID             beta" in out


def test_inspect_reports_the_fingerprint_and_size_but_not_the_content(
    capsys: pytest.CaptureFixture[str],
):
    _, out, _ = _run(capsys, "knowledge", "inspect", "alpha", "--root", str(VALID_ROOT))

    assert "Fingerprint" in out
    assert "Characters" in out
    assert "Neutral fixture content" not in out


def test_inspect_shows_an_optional_field_only_when_it_is_set(
    capsys: pytest.CaptureFixture[str],
):
    _, with_license, _ = _run(capsys, "knowledge", "inspect", "beta", "--root", str(VALID_ROOT))
    _, without_license, _ = _run(capsys, "knowledge", "inspect", "alpha", "--root", str(VALID_ROOT))

    assert "License        CC0-1.0" in with_license
    assert "License" not in without_license


def test_inspect_fails_cleanly_for_an_unknown_document(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "knowledge", "inspect", "nope", "--root", str(VALID_ROOT))

    assert code == EXIT_INVALID
    assert "No valid document matches" in err


def test_inspect_explains_why_a_broken_document_cannot_be_shown(
    capsys: pytest.CaptureFixture[str],
):
    code, _, err = _run(
        capsys, "knowledge", "inspect", "04-invalid-metadata.md", "--root", str(INVALID_ROOT)
    )

    assert code == EXIT_INVALID
    assert "INVALID_METADATA" in err


# --- argument handling ------------------------------------------------------


@pytest.mark.parametrize("argv", [[], ["knowledge"], ["nonsense"], ["knowledge", "nonsense"]])
def test_incomplete_invocations_are_usage_errors(argv: list[str]):
    with pytest.raises(SystemExit) as caught:
        main(argv)

    assert caught.value.code == 2  # argparse's own usage exit code


def test_an_unexpected_failure_is_not_reported_as_success(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    def explode(_: Path) -> None:
        raise RuntimeError("something went badly wrong")

    monkeypatch.setattr("portfolio_rag.cli.collect_knowledge_base", explode)

    code, _, err = _run(capsys, "knowledge", "validate", "--root", str(VALID_ROOT))

    assert code == EXIT_INTERNAL_ERROR
    assert "internal error: RuntimeError" in err
    assert "Traceback" not in err
