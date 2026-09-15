"""`eval run`, driven as a shell would drive it — and what pacing does to it.

The offline stack answers nothing: the shipped deterministic embedding provider
derives vectors from SHA-256, so at the default threshold every question
short-circuits before generation. That makes this a test of the *command* —
the flag, its validation, its header, and the fact that an unpaced run is
unchanged — and not of retrieval quality, which is measured in
`tests/evaluation/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from portfolio_rag import cli, composition
from portfolio_rag.cli import EXIT_INVALID, EXIT_OK, main
from portfolio_rag.core.config import get_settings

DATASET = Path("evaluation/questions.yaml")


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
    code = main(["eval", "run", "--dataset", str(DATASET), *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_run_reports_what_it_measured(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run(capsys, "--retrieval-only")

    assert code == EXIT_OK
    assert "Retrieval" in out
    assert "no generation provider was called" in out


def test_an_unpaced_run_says_nothing_about_pacing(capsys: pytest.CaptureFixture[str]):
    """The default is the run that existed before the flag did."""
    _, out, _ = _run(capsys)

    assert "Generation pacing" not in out


def test_a_paced_run_prints_the_interval_it_will_keep(capsys: pytest.CaptureFixture[str]):
    """A report that does not say how it was produced is harder to trust."""
    code, out, _ = _run(capsys, "--generation-delay-seconds", "8")

    assert code == EXIT_OK
    assert "Generation pacing" in out
    assert "8.0s" in out


def test_pacing_is_not_claimed_when_no_model_is_reached(capsys: pytest.CaptureFixture[str]):
    _, out, _ = _run(capsys, "--retrieval-only", "--generation-delay-seconds", "8")

    assert "Generation pacing" not in out


def _record_wiring(monkeypatch: pytest.MonkeyPatch) -> list[int | None]:
    """Capture the transport budget the CLI asks the composition root for."""
    asked: list[int | None] = []
    build = composition.build_query_components

    def record(settings: Any, *, generation_attempts: int | None = None) -> Any:
        asked.append(generation_attempts)
        return build(settings, generation_attempts=generation_attempts)

    monkeypatch.setattr(cli, "build_query_components", record)
    return asked


def test_a_paced_run_builds_a_provider_that_does_no_retrying_of_its_own(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """The wiring decision that keeps a question to three requests.

    The pacer retries; the adapter under it must not, or the two budgets
    multiply. Checked at the seam, because the offline stack has no transport
    to count requests on — `tests/integration/test_eval_generation_budget.py`
    is where the requests themselves are counted.
    """
    asked = _record_wiring(monkeypatch)

    _run(capsys, "--generation-delay-seconds", "8")

    assert asked == [1]


def test_an_unpaced_run_leaves_the_adapter_on_its_own_budget(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Without pacing there is no second retrying layer to make room for."""
    asked = _record_wiring(monkeypatch)

    _run(capsys)

    assert asked == [None]


def test_a_negative_delay_is_refused(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "--generation-delay-seconds", "-1")

    assert code == EXIT_INVALID
    assert "Invalid generation pacing" in err
