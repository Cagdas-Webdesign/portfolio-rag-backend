"""`knowledge embedding` and `knowledge index`, driven as a shell would.

The default configuration — deterministic provider, in-memory store — needs no
credentials and reaches no network, which is exactly what makes these runnable
in CI.

One honest limitation is visible here: an in-memory index does not survive
between processes, so a *second* CLI invocation starts from an empty index.
Idempotency across runs is therefore proven in the application tests, which
hold one store instance, and would only be demonstrable end to end against a
persistent adapter. Building a file-backed store to make the CLI look
idempotent would be inventing infrastructure to satisfy a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.cli import EXIT_INVALID, EXIT_OK, main
from portfolio_rag.core.config import get_settings

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


@pytest.fixture(autouse=True)
def _local_defaults(monkeypatch: pytest.MonkeyPatch):
    """Force the offline stack, whatever the developer's environment says."""
    for name in ("EMBEDDING_PROVIDER", "VECTOR_STORE", "MISTRAL_API_KEY"):
        monkeypatch.delenv(f"PORTFOLIO_RAG_{name}", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


# --- knowledge embedding ----------------------------------------------------


def test_embedding_shows_the_space_and_the_representation(
    capsys: pytest.CaptureFixture[str],
):
    code, out, _ = _run(capsys, "knowledge", "embedding", "alpha", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "provider       deterministic" in out
    assert "representation embedding-text-v1" in out
    assert "Chunk ID       alpha--0000" in out
    assert "Fingerprint" in out
    assert "Dimensions" in out
    assert "Characters" in out


def test_embedding_previews_the_text_and_can_show_all_of_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    body = " ".join(f"Sentence {index} of a long body." for index in range(40))
    root = _knowledge_base(tmp_path / "kb", long=body)

    _, preview, _ = _run(capsys, "knowledge", "embedding", "long", "--root", str(root))
    _, full, _ = _run(capsys, "knowledge", "embedding", "long", "--root", str(root), "--show-text")

    assert "…" in preview
    assert len(full) > len(preview)
    assert "Sentence 39 of a long body." in full


def test_embedding_starts_with_the_title_and_heading_path(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    root = _knowledge_base(tmp_path / "kb", doc="# Backend\n\n## APIs\n\nBody text.")

    _, out, _ = _run(capsys, "knowledge", "embedding", "doc", "--root", str(root), "--show-text")

    assert "    │ Doc" in out
    assert "    │ Backend > APIs" in out
    assert "    │ Body text." in out


def test_embedding_accepts_a_chunk_id(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "embedding", "alpha--0001", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "Chunk ID       alpha--0001" in out
    assert "1 chunk\n" in out


def test_embedding_accepts_a_document_path(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(
        capsys, "knowledge", "embedding", "topics/beta.md", "--root", str(VALID_ROOT)
    )

    assert code == EXIT_OK
    assert "Chunk ID       beta--0000" in out


def test_embedding_needs_no_credentials_and_makes_no_request(
    capsys: pytest.CaptureFixture[str],
):
    """Showing what *would* be sent must never send it."""
    code, out, _ = _run(capsys, "knowledge", "embedding", "alpha", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "api.mistral.ai" not in out


def test_embedding_reports_an_unknown_target_cleanly(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "knowledge", "embedding", "nope", "--root", str(VALID_ROOT))

    assert code == EXIT_INVALID
    assert "No chunk or document matches" in err
    assert "Traceback" not in err


def test_embedding_reports_a_broken_knowledge_base_cleanly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    code, _, err = _run(
        capsys, "knowledge", "embedding", "anything", "--root", str(tmp_path / "nowhere")
    )

    assert code == EXIT_INVALID
    assert "Traceback" not in err


def test_embedding_refuses_an_incoherent_policy(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(
        capsys,
        "knowledge",
        "embedding",
        "alpha",
        "--root",
        str(VALID_ROOT),
        "--target-chars",
        "0",
    )

    assert code == EXIT_INVALID
    assert "Invalid chunking policy" in err


# --- knowledge index --dry-run ----------------------------------------------


def test_a_dry_run_reports_the_plan(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "index", "--dry-run", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "Documents      3" in out
    assert "Chunks         4" in out
    assert "create              4" in out
    assert "unchanged           0" in out
    assert "delete              0" in out
    assert "embeddings required 4" in out
    assert "Dry run: nothing was embedded, written or deleted." in out


def test_a_dry_run_writes_nothing(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    """Asserted by running it twice: a run that wrote would change the second plan."""
    root = _knowledge_base(tmp_path / "kb", doc="Body text.")

    _run(capsys, "knowledge", "index", "--dry-run", "--root", str(root))
    _, second, _ = _run(capsys, "knowledge", "index", "--dry-run", "--root", str(root))

    assert "create              1" in second
    assert "unchanged           0" in second


def test_a_dry_run_makes_no_quality_claims(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "knowledge", "index", "--dry-run", "--root", str(VALID_ROOT))

    for forbidden in ("quality", "score", "optimal", "accuracy"):
        assert forbidden not in out.lower()


def test_a_dry_run_shows_the_embedding_space_it_planned_against(
    capsys: pytest.CaptureFixture[str],
):
    _, out, _ = _run(capsys, "knowledge", "index", "--dry-run", "--root", str(VALID_ROOT))

    assert "provider       deterministic" in out
    assert "representation embedding-text-v1" in out


# --- knowledge index --------------------------------------------------------


def test_indexing_reports_what_it_did(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "index", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "created              4" in out
    assert "embeddings generated 4" in out
    assert "embeddings reused    0" in out
    assert "records written      4" in out
    assert "Index synchronized." in out


def test_indexing_an_empty_knowledge_base_is_a_clean_no_op(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    tmp_path.mkdir(parents=True, exist_ok=True)

    code, out, _ = _run(capsys, "knowledge", "index", "--root", str(tmp_path))

    assert code == EXIT_OK
    assert "Chunks         0" in out
    assert "Index already up to date." in out


def test_indexing_refuses_a_broken_knowledge_base(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "knowledge", "index", "--root", str(INVALID_ROOT))

    assert code == EXIT_INVALID
    assert "unusable document" in err
    assert "knowledge validate" in err


def test_indexing_reports_a_chunking_failure_rather_than_crashing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    fence = "```python\n" + "\n".join(f"value_{index} = {index}" for index in range(60)) + "\n```"
    root = _knowledge_base(tmp_path / "kb", huge=f"# Examples\n\n{fence}")

    code, _, err = _run(
        capsys,
        "knowledge",
        "index",
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
    assert "Traceback" not in err


def test_indexing_refuses_an_incoherent_policy(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "knowledge", "index", "--root", str(VALID_ROOT), "--max-chars", "0")

    assert code == EXIT_INVALID
    assert "Invalid chunking policy" in err


def test_a_missing_credential_is_reported_as_configuration_not_a_crash(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PORTFOLIO_RAG_EMBEDDING_PROVIDER", "mistral")
    get_settings.cache_clear()

    code, _, err = _run(capsys, "knowledge", "index", "--dry-run", "--root", str(VALID_ROOT))

    assert code == EXIT_INVALID
    assert "MISTRAL_API_KEY" in err
    assert "Traceback" not in err


def test_rebuild_is_accepted_and_reported_as_creates(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "knowledge", "index", "--rebuild", "--root", str(VALID_ROOT))

    assert code == EXIT_OK
    assert "created              4" in out
