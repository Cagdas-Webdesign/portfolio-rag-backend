"""`query retrieve` and `query answer`, driven as a shell would drive them.

The default configuration — deterministic embeddings, in-memory store, the
development generation stub — needs no credentials and reaches no network,
which is what makes these runnable in CI and on a fresh checkout.

The similarity threshold is dropped to zero in most of these runs, and that is
not a trick: the shipped offline embedding provider derives vectors from
SHA-256 and has no semantics, so every score it produces is noise around zero.
Retrieval *quality* is measured elsewhere, against a provider that has some;
what these tests check is the command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.cli import EXIT_INVALID, EXIT_OK, main
from portfolio_rag.core.config import get_settings

FIXTURES = Path(__file__).parent.parent / "fixtures" / "knowledge"
RAG_ROOT = FIXTURES / "rag"
INVALID_ROOT = FIXTURES / "invalid"
SECRET_MARKER = "INTERNAL-ONLY-SECRET-VALUE"  # noqa: S105 - a fixture marker, not a secret

ALL_MATCHES = ("--min-similarity", "-1.0")


@pytest.fixture(autouse=True)
def _local_defaults(monkeypatch: pytest.MonkeyPatch):
    """Force the offline stack, whatever the developer's environment says."""
    for name in (
        "EMBEDDING_PROVIDER",
        "LLM_PROVIDER",
        "VECTOR_STORE",
        "MISTRAL_API_KEY",
        "RETRIEVAL_TOP_K",
        "RETRIEVAL_MIN_SIMILARITY",
    ):
        monkeypatch.delenv(f"PORTFOLIO_RAG_{name}", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _retrieve(capsys: pytest.CaptureFixture[str], question: str, *extra: str):
    return _run(capsys, "query", "retrieve", question, "--root", str(RAG_ROOT), *extra)


def _answer(capsys: pytest.CaptureFixture[str], question: str, *extra: str):
    return _run(capsys, "query", "answer", question, "--root", str(RAG_ROOT), *extra)


# --- retrieve ----------------------------------------------------------------


def test_a_valid_question_reports_what_it_found(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _retrieve(capsys, "Which HTTP framework is used?", *ALL_MATCHES)

    assert code == EXIT_OK
    assert "Question: Which HTTP framework is used?" in out
    assert "Retrieval" in out


def test_the_embedding_space_and_the_policy_are_shown(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "Which framework?", *ALL_MATCHES)

    assert "Embedding space" in out
    assert "deterministic" in out
    assert "top k" in out
    assert "min similarity" in out


def test_each_match_reports_its_identity_and_provenance(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "Which framework?", *ALL_MATCHES)

    assert "Similarity" in out
    assert "Chunk ID" in out
    assert "Document" in out
    assert "Heading" in out
    assert "Source" in out
    assert "Visibility" in out


def test_a_content_preview_is_printed(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "Which framework?", *ALL_MATCHES)

    assert "    │ " in out


def test_full_content_can_be_requested(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "Which framework?", *ALL_MATCHES, "--show-content")

    assert "FastAPI" in out or "Markdown" in out


def test_several_matches_are_numbered(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "framework storage", *ALL_MATCHES, "--top-k", "3")

    assert "Match 1" in out
    assert "Match 2" in out


def test_top_k_bounds_how_many_are_shown(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "framework storage", *ALL_MATCHES, "--top-k", "1")

    assert "Match 1" in out
    assert "Match 2" not in out


def test_a_threshold_nothing_can_meet_reports_no_passages(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _retrieve(capsys, "Which framework?", "--min-similarity", "1.0")

    assert code == EXIT_OK
    assert "No passage met the retrieval criteria." in out


def test_similarity_is_never_presented_as_a_confidence(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _retrieve(capsys, "Which framework?", *ALL_MATCHES)

    assert "Similarity" in out
    assert "confidence" not in out.lower()
    assert "%" not in out


def test_only_public_passages_are_ever_listed(capsys: pytest.CaptureFixture[str]):
    _, out, err = _retrieve(capsys, "What is the confidential codename?", *ALL_MATCHES)

    assert SECRET_MARKER not in out
    assert SECRET_MARKER not in err
    assert "internal" not in out.replace("internal-notes", "")


# --- answer ------------------------------------------------------------------


def test_a_question_is_answered_end_to_end(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _answer(capsys, "Which HTTP framework is used?", *ALL_MATCHES)

    assert code == EXIT_OK
    assert "Answer" in out
    assert "outcome" in out


def test_the_answer_reports_its_sources(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _answer(capsys, "Which HTTP framework is used?", *ALL_MATCHES)

    assert "Sources" in out
    assert "HTTP Stack" in out or "Storage Layer" in out


def test_an_unsupported_question_says_so_without_sources(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _answer(capsys, "Which framework?", "--min-similarity", "1.0")

    assert code == EXIT_OK
    assert "not grounded in the knowledge base" in out


def test_retrieval_can_be_shown_alongside_the_answer(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _answer(capsys, "Which framework?", *ALL_MATCHES, "--show-retrieval")

    assert "Retrieval" in out
    assert "Similarity" in out


def test_the_context_the_model_received_can_be_shown(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _answer(capsys, "Which framework?", *ALL_MATCHES, "--show-context")

    assert "[SOURCE S1]" in out
    assert "estimated tokens" in out


def test_the_generation_model_is_named(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _answer(capsys, "Which framework?", *ALL_MATCHES)

    assert "generation model" in out
    assert "context-echo-v1" in out


def test_internal_knowledge_is_absent_from_the_answer(capsys: pytest.CaptureFixture[str]):
    _, out, err = _answer(capsys, "What is the confidential codename?", *ALL_MATCHES)

    assert SECRET_MARKER not in out
    assert SECRET_MARKER not in err


def test_no_credential_or_header_is_ever_printed(capsys: pytest.CaptureFixture[str]):
    _, out, err = _answer(capsys, "Which framework?", *ALL_MATCHES, "--show-context")

    for leak in ("Authorization", "Bearer", "api_key", "API_KEY"):
        assert leak not in out
        assert leak not in err


# --- invalid input -----------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_an_empty_question_is_a_usage_failure(capsys: pytest.CaptureFixture[str], question: str):
    code, _, err = _retrieve(capsys, question)

    assert code == EXIT_INVALID
    assert "Invalid question" in err


def test_a_question_over_the_limit_is_refused(capsys: pytest.CaptureFixture[str]):
    code, _, err = _retrieve(capsys, "y" * 4001)

    assert code == EXIT_INVALID
    assert "Invalid question" in err


@pytest.mark.parametrize("override", [("--top-k", "0"), ("--min-similarity", "5")])
def test_an_impossible_policy_is_refused_before_anything_runs(
    capsys: pytest.CaptureFixture[str], override: tuple[str, str]
):
    code, _, err = _retrieve(capsys, "Which framework?", *override)

    assert code == EXIT_INVALID
    assert "Invalid retrieval policy" in err


def test_a_broken_knowledge_base_is_reported_rather_than_half_searched(
    capsys: pytest.CaptureFixture[str],
):
    code, _, err = _run(capsys, "query", "retrieve", "anything", "--root", str(INVALID_ROOT))

    assert code == EXIT_INVALID
    assert "Configuration error" in err


def test_a_missing_knowledge_root_is_reported(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    code, _, err = _run(
        capsys, "query", "retrieve", "anything", "--root", str(tmp_path / "nowhere")
    )

    assert code == EXIT_INVALID
    assert "Configuration error" in err


def test_no_traceback_is_ever_printed(capsys: pytest.CaptureFixture[str]):
    _, out, err = _run(capsys, "query", "retrieve", "", "--root", str(RAG_ROOT))

    assert "Traceback" not in out + err
    assert ".py" not in err


def test_an_empty_corpus_answers_without_failing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    empty = tmp_path / "empty"
    empty.mkdir()

    code, out, _ = _run(capsys, "query", "answer", "anything", "--root", str(empty))

    assert code == EXIT_OK
    assert "not grounded in the knowledge base" in out
