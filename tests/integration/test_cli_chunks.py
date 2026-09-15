"""`knowledge chunks`, driven exactly as a shell would drive it."""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.cli import EXIT_INVALID, EXIT_OK, main

FIXTURES = Path(__file__).parent.parent / "fixtures" / "knowledge"
VALID_ROOT = FIXTURES / "valid"
INVALID_ROOT = FIXTURES / "invalid"

DOCUMENT = """---
schema_version: 1
id: {id}
title: {title}
document_type: reference
language: en
source: fixture
source_type: authored
version: 1
updated_at: 2026-08-07
---

{body}
"""


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _knowledge_base(root: Path, **documents: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for document_id, body in documents.items():
        slug = document_id.replace("_", "-")
        (root / f"{slug}.md").write_text(
            DOCUMENT.format(id=slug, title=slug.title(), body=body), encoding="utf-8"
        )
    return root


# --- single document --------------------------------------------------------


def test_chunks_shows_policy_and_boundaries_for_one_document(
    capsys: pytest.CaptureFixture[str],
):
    code, out, _ = _run(capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "Document: Test Document Alpha" in out
    assert "ID: alpha" in out
    assert "strategy       markdown-structure-v1" in out
    assert "target chars   1200" in out
    assert "Chunk 0000" in out
    assert "alpha--0000" in out


def test_chunks_reports_the_heading_path_of_every_chunk(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT))

    assert "Test Document Alpha > A second section" in out


def test_chunks_accepts_a_path_as_well_as_an_id(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "chunks", "topics/beta.md", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "ID: beta" in out


def test_chunks_shows_a_fingerprint_and_an_overlap_count(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT))

    assert "Fingerprint" in out
    assert "Overlap" in out


def test_content_is_hidden_unless_it_is_asked_for(capsys: pytest.CaptureFixture[str]):
    _, without, _ = _run(capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT))
    _, with_content, _ = _run(
        capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT), "--show-content"
    )

    assert "Neutral fixture content" not in without
    assert "Neutral fixture content" in with_content


def test_shown_content_is_visually_separated_from_the_metadata(
    capsys: pytest.CaptureFixture[str],
):
    _, out, _ = _run(
        capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT), "--show-content"
    )

    assert "    │ " in out


def test_the_output_makes_no_quality_claims(capsys: pytest.CaptureFixture[str]):
    """Structural statistics must not be presented as retrieval quality."""
    _, out, _ = _run(capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT))

    for forbidden in ("quality", "score", "optimal", "accuracy", "%"):
        assert forbidden not in out.lower()


# --- policy overrides -------------------------------------------------------


def test_a_smaller_budget_produces_more_chunks(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    root = _knowledge_base(
        tmp_path / "kb", long=" ".join(f"Sentence {index} of the body." for index in range(40))
    )

    _, wide, _ = _run(capsys, "knowledge", "chunks", "long", "--root", str(root))
    _, narrow, _ = _run(
        capsys,
        "knowledge",
        "chunks",
        "long",
        "--root",
        str(root),
        "--target-chars",
        "200",
        "--max-chars",
        "300",
        "--overlap-chars",
        "50",
    )

    assert "1 chunk\n" in wide
    assert "chunks\n" in narrow
    assert narrow.count("Chunk 00") > wide.count("Chunk 00")


def test_an_override_is_shown_in_the_policy_block(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    root = _knowledge_base(tmp_path / "kb", short="Body text.")

    _, out, _ = _run(
        capsys, "knowledge", "chunks", "short", "--root", str(root), "--target-chars", "500"
    )

    assert "target chars   500" in out
    assert "max chars      1800" in out, "unspecified settings keep their defaults"


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (["--target-chars", "0"], "target_chars"),
        (["--max-chars", "-5"], "max_chars"),
        (["--target-chars", "500", "--max-chars", "100"], "at least"),
        (["--overlap-chars", "-1"], "overlap_chars"),
        (["--target-chars", "100", "--max-chars", "200"], "smaller than"),
    ],
)
def test_an_incoherent_policy_is_refused_with_an_explanation(
    capsys: pytest.CaptureFixture[str], flags: list[str], expected: str
):
    code, _, err = _run(capsys, "knowledge", "chunks", "alpha", "--root", str(VALID_ROOT), *flags)

    assert code == EXIT_INVALID
    assert "Invalid chunking policy" in err
    assert expected in err
    assert "Traceback" not in err


# --- corpus summary ---------------------------------------------------------


def test_all_summarizes_the_corpus(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "chunks", "--all", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "Documents      3" in out
    assert "Chunks" in out
    assert "Average chars" in out
    assert "Over max       0" in out


def test_all_refuses_to_summarize_a_broken_knowledge_base(
    capsys: pytest.CaptureFixture[str],
):
    """A partial corpus reported as a whole one would be a lie with numbers on it."""
    code, _, err = _run(capsys, "knowledge", "chunks", "--all", "--root", str(INVALID_ROOT))

    assert code == EXIT_INVALID
    assert "unusable document" in err
    assert "knowledge validate" in err


def test_all_on_an_empty_knowledge_base_reports_zeroes(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    code, out, _ = _run(capsys, "knowledge", "chunks", "--all", "--root", str(tmp_path))

    assert code == EXIT_OK
    assert "Documents      0" in out
    assert "Chunks         0" in out


# --- error paths ------------------------------------------------------------


def test_an_unknown_document_fails_cleanly(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "knowledge", "chunks", "nope", "--root", str(VALID_ROOT))

    assert code == EXIT_INVALID
    assert "No valid document matches" in err
    assert "Traceback" not in err


def test_a_document_that_failed_to_ingest_explains_why(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(
        capsys, "knowledge", "chunks", "04-invalid-metadata.md", "--root", str(INVALID_ROOT)
    )

    assert code == EXIT_INVALID
    assert "INVALID_METADATA" in err


def test_a_missing_knowledge_root_fails_cleanly(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    code, _, err = _run(
        capsys, "knowledge", "chunks", "anything", "--root", str(tmp_path / "nowhere")
    )

    assert code == EXIT_INVALID
    assert "Traceback" not in err


def test_omitting_both_target_and_all_is_explained(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "knowledge", "chunks", "--root", str(VALID_ROOT))

    assert code == EXIT_INVALID
    assert "--all" in err


def test_an_unsplittable_block_is_reported_not_crashed_on(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    fence = "```python\n" + "\n".join(f"value_{index} = {index}" for index in range(60)) + "\n```"
    root = _knowledge_base(tmp_path / "kb", huge=f"# Examples\n\n{fence}")

    code, _, err = _run(
        capsys,
        "knowledge",
        "chunks",
        "huge",
        "--root",
        str(root),
        "--target-chars",
        "100",
        "--max-chars",
        "150",
        "--overlap-chars",
        "0",
    )

    assert code == EXIT_INVALID
    assert "UNSPLITTABLE_BLOCK" in err
    assert "huge" in err
    assert "Traceback" not in err


def test_an_unsplittable_block_also_fails_the_corpus_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    fence = "```python\n" + "\n".join(f"value_{index} = {index}" for index in range(60)) + "\n```"
    root = _knowledge_base(tmp_path / "kb", fine="Short body.", huge=f"# Examples\n\n{fence}")

    code, _, err = _run(
        capsys,
        "knowledge",
        "chunks",
        "--all",
        "--root",
        str(root),
        "--target-chars",
        "100",
        "--max-chars",
        "150",
        "--overlap-chars",
        "0",
    )

    assert code == EXIT_INVALID
    assert "UNSPLITTABLE_BLOCK" in err
