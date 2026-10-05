"""`eval run`, driven as a shell would drive it — and what pacing does to it.

The offline stack answers nothing: the shipped deterministic embedding provider
derives vectors from SHA-256, so at the default threshold every question
short-circuits before generation. That makes this a test of the *command* —
the flag, its validation, its header, and the fact that an unpaced run is
unchanged — and not of retrieval quality, which is measured in
`tests/evaluation/`.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag import cli, composition
from portfolio_rag.cli import (
    EXIT_ACCEPTANCE_FAILED,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_OK,
    main,
)
from portfolio_rag.core.config import EmbeddingProviderName, LLMProviderName, get_settings
from portfolio_rag.core.request_context import get_request_id
from portfolio_rag.infrastructure.llm import DeterministicLLMProvider
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.rag.policy import DEFAULT_TOP_K
from portfolio_rag.rag.service import GroundedAnswerService

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
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_WORKERS_AI_TOKEN",
        "CLOUDFLARE_VECTORIZE_INDEX",
    ):
        monkeypatch.delenv(f"PORTFOLIO_RAG_{name}", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["eval", "run", "--dataset", str(DATASET), *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _run_acceptance(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """An acceptance-tier run: the tier asks the release-acceptance suite, which
    exists for the portfolio dataset only."""
    code = main(
        [
            "eval",
            "run",
            "--dataset",
            "evaluation/portfolio-questions.yaml",
            "--e2e",
            "--tier",
            "acceptance",
            *argv,
        ]
    )
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


# --- suites and the retrieval export ------------------------------------------
#
# Still the offline stack: deterministic vectors, the in-memory store and no
# transport anywhere, so none of these can reach a provider.

PORTFOLIO = Path("evaluation/portfolio-questions.yaml")


def _run_portfolio(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["eval", "run", "--dataset", str(PORTFOLIO), "--retrieval-only", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _without_run_id(report: str) -> str:
    return "\n".join(line for line in report.splitlines() if not line.startswith("Run:"))


def test_without_a_suite_the_run_is_the_full_run_it_always_was(
    capsys: pytest.CaptureFixture[str],
):
    """Case O: no flag and `--suite full` print the same report, with no suite line."""
    _, default, _ = _run_portfolio(capsys)
    _, full, _ = _run_portfolio(capsys, "--suite", "full")

    # Every run has an id of its own; everything else is the same report.
    assert _without_run_id(default) == _without_run_id(full)
    assert "Suite:" not in default
    assert "questions            49" in default


def test_the_smoke_suite_runs_twelve_questions_and_says_so(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run_portfolio(capsys, "--suite", "smoke")

    assert code == EXIT_OK
    assert "Suite:   smoke  (12 of 49 questions)" in out
    assert "questions            12" in out


def test_a_dataset_without_a_smoke_file_is_refused(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "--retrieval-only", "--suite", "smoke")

    assert code == EXIT_INVALID
    assert "Evaluation dataset error" in err


def test_without_output_no_file_is_written_and_nothing_is_said(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Case N."""
    monkeypatch.chdir(tmp_path)
    Path("evaluation").symlink_to(Path(__file__).parents[2] / "evaluation")
    Path("knowledge").symlink_to(Path(__file__).parents[2] / "knowledge")

    code, out, _ = _run_portfolio(capsys)

    assert code == EXIT_OK
    assert "Retrieval export" not in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evaluation", "knowledge"]


def test_the_export_is_written_beside_the_normal_report(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Cases F and G, through the command."""
    target = tmp_path / "smoke.json"

    code, out, _ = _run_portfolio(capsys, "--suite", "smoke", "--output", str(target))

    assert code == EXIT_OK
    assert "Retrieval\n" in out, "the stdout report is still printed"
    assert f"Retrieval export: {target}" in out
    data = _read_json(target)
    assert data["run"]["suite"] == "smoke"
    assert data["run"]["question_count"] == 12
    assert data["run"]["dataset"]["question_count"] == 49
    assert data["run"]["dataset"]["path"] == PORTFOLIO.as_posix()
    assert data["run"]["embedding"]["provider"] == "deterministic"
    assert data["run"]["vector_store"] == "memory"
    assert len(data["questions"]) == 12


def test_a_raw_run_exports_every_match_the_store_returned(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """What the threshold analysis will read: `-1` keeps all `top_k` matches.

    The deterministic vectors score far below the production threshold, so at
    the default these questions would export nothing. At `-1` every question
    carries its `top_k` matches, scores included, and a higher threshold can be
    applied afterwards by filtering the file.
    """
    target = tmp_path / "raw.json"

    _run_portfolio(capsys, "--min-similarity", "-1", "--output", str(target))

    data = _read_json(target)
    assert data["run"]["retrieval_policy"] == {
        "top_k": DEFAULT_TOP_K,
        "min_similarity": -1.0,
        "visibility": "public",
    }
    for question in data["questions"]:
        assert len(question["hits"]) == DEFAULT_TOP_K
        assert question["retrieval"]["below_threshold"] == 0
        scores = [hit["similarity"] for hit in question["hits"]]
        assert scores == sorted(scores, reverse=True)
    assert any(hit["similarity"] < 0.25 for q in data["questions"] for hit in q["hits"])


def test_at_the_default_threshold_dropped_matches_are_counted(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    """Case L through the command: what the threshold removed is still accounted for."""
    target = tmp_path / "default.json"

    _run_portfolio(capsys, "--output", str(target))

    for question in _read_json(target)["questions"]:
        counts = question["retrieval"]
        assert (
            counts["matches_returned"]
            == (counts["below_threshold"] + counts["unresolved"] + counts["withheld"])
            + counts["retrieved"]
        )
        assert len(question["hits"]) == counts["retrieved"]


def test_no_secret_reaches_the_export(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Case P: credentials in the environment stay out of the file."""
    canaries = {
        "MISTRAL_API_KEY": "canary-mistral-key-7f3a",
        "CLOUDFLARE_ACCOUNT_ID": "canary-account-id-7f3a",
        "CLOUDFLARE_API_TOKEN": "canary-vectorize-token-7f3a",
        "CLOUDFLARE_WORKERS_AI_TOKEN": "canary-workers-token-7f3a",
        "CLOUDFLARE_VECTORIZE_INDEX": "canary-index-name-7f3a",
    }
    for name, value in canaries.items():
        monkeypatch.setenv(f"PORTFOLIO_RAG_{name}", value)
    get_settings.cache_clear()
    target = tmp_path / "run.json"

    code, _, _ = _run_portfolio(capsys, "--suite", "smoke", "--output", str(target))

    assert code == EXIT_OK
    text = target.read_text(encoding="utf-8")
    assert "7f3a" not in text
    assert "canary" not in text


def test_retrieval_only_with_an_export_never_calls_a_model(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Case Q."""

    async def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("a retrieval-only run called the generation provider")

    monkeypatch.setattr(DeterministicLLMProvider, "generate", refuse)
    target = tmp_path / "run.json"

    code, out, _ = _run_portfolio(capsys, "--min-similarity", "-1", "--output", str(target))

    assert code == EXIT_OK
    assert "no generation provider was called" in out
    assert target.exists()


def test_an_unwritable_export_fails_the_run(capsys: pytest.CaptureFixture[str], tmp_path: Path):
    code, _, err = _run_portfolio(capsys, "--output", str(tmp_path / "missing" / "run.json"))

    assert code == EXIT_INVALID
    assert "Retrieval export could not be written" in err


# --- pacing between questions --------------------------------------------------
#
# The runner's sleeper is replaced at the CLI seam, so these record every wait
# the command asked for without spending a second on it.


def _record_question_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Capture the waits a run makes, and make none of them."""
    waited: list[float] = []

    async def no_wait(seconds: float) -> None:
        waited.append(seconds)

    for name in ("run_retrieval_evaluation", "run_grounding_evaluation"):
        real = getattr(cli, name)

        def paced(*args: Any, _real: Any = real, **kwargs: Any) -> Any:
            return _real(*args, sleeper=no_wait, **kwargs)

        monkeypatch.setattr(cli, name, paced)
    return waited


def test_by_default_a_run_waits_between_no_questions(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Case A."""
    waited = _record_question_delays(monkeypatch)

    code, out, _ = _run_portfolio(capsys)

    assert code == EXIT_OK
    assert waited == []
    assert "Retrieval pacing" not in out, "the default header is the one it always was"


@pytest.mark.parametrize("value", ["-1", "-0.5", "nan", "inf"])
def test_an_unusable_retrieval_delay_is_refused(capsys: pytest.CaptureFixture[str], value: str):
    """Case B."""
    code, _, err = _run_portfolio(capsys, "--retrieval-delay-seconds", value)

    assert code == EXIT_INVALID
    assert "Invalid retrieval pacing" in err


def test_a_delay_of_three_seconds_is_parsed_and_kept(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Case C."""
    waited = _record_question_delays(monkeypatch)

    code, _, _ = _run_portfolio(capsys, "--suite", "smoke", "--retrieval-delay-seconds", "3")

    assert code == EXIT_OK
    assert set(waited) == {3.0}


@pytest.mark.parametrize(("suite", "questions"), [("smoke", 12), ("full", 49)])
def test_both_suites_pause_once_between_each_pair_of_questions(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    suite: str,
    questions: int,
):
    """Cases D and G: 11 pauses for smoke, 48 for full."""
    waited = _record_question_delays(monkeypatch)

    code, out, _ = _run_portfolio(capsys, "--suite", suite, "--retrieval-delay-seconds", "3")

    assert code == EXIT_OK
    assert len(waited) == questions - 1
    assert f"questions      {questions}" in out
    assert f"pauses         {questions - 1} per pass" in out
    assert "delay          3.0s between questions" in out
    assert "generation     off (--retrieval-only)" in out


def test_the_pacing_header_comes_before_any_question_is_asked(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """A live run shows what it is about to do before its first request."""
    _record_question_delays(monkeypatch)

    _, out, _ = _run_portfolio(capsys, "--suite", "smoke", "--retrieval-delay-seconds", "3")

    assert out.index("Retrieval pacing") < out.index("Retrieval\n")


def test_a_paced_retrieval_only_run_never_calls_a_model(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Case H."""

    async def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("a retrieval-only run called the generation provider")

    monkeypatch.setattr(DeterministicLLMProvider, "generate", refuse)
    _record_question_delays(monkeypatch)

    code, _, _ = _run_portfolio(capsys, "--min-similarity", "-1", "--retrieval-delay-seconds", "3")

    assert code == EXIT_OK


def test_the_export_records_the_delay_the_run_kept(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Case J, through the command."""
    _record_question_delays(monkeypatch)
    target = tmp_path / "run.json"

    _run_portfolio(
        capsys, "--suite", "smoke", "--retrieval-delay-seconds", "3", "--output", str(target)
    )

    assert _read_json(target)["run"]["retrieval_delay_seconds"] == 3.0


def test_a_full_run_paces_its_answering_pass_as_well(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    """Without --retrieval-only both passes embed every question, and both are paced."""
    waited = _record_question_delays(monkeypatch)

    code, out, _ = _run(capsys, "--retrieval-delay-seconds", "2")

    assert code == EXIT_OK
    assert len(waited) == 2 * (24 - 1)
    assert "generation     on" in out


# --- the end-to-end run --------------------------------------------------------
#
# The offline stack again, so every question short-circuits before generation.
# That is enough to check the command: which flags it accepts, what it writes
# and when it exits non-zero. The scoring itself is in
# `tests/unit/test_evaluation_e2e.py`.


def test_an_end_to_end_run_without_an_output_is_refused_before_anything_runs(
    capsys: pytest.CaptureFixture[str],
):
    code, out, err = _run(capsys, "--e2e")

    assert code == EXIT_INVALID
    assert "--e2e needs --output" in err
    assert out == ""


def test_an_end_to_end_run_cannot_also_be_retrieval_only(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    code, _, err = _run(capsys, "--e2e", "--retrieval-only", "--output", str(tmp_path / "x.json"))

    assert code == EXIT_INVALID
    assert "--retrieval-only" in err
    assert not (tmp_path / "x.json").exists()


def test_the_end_to_end_flags_mean_nothing_without_the_run(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "--summary", "summary.md")

    assert code == EXIT_INVALID
    assert "belong to an --e2e run" in err


@pytest.fixture
def real_provider_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Settings that name real providers, over a stack that is still offline.

    An end-to-end run refuses development stand-ins, so the tests of the command
    itself have to get past that check the way a real run does: by configuring
    real providers. What is built for those settings is swapped for the offline
    stack at the composition seam, and the credential is a placeholder that no
    adapter is ever handed.
    """
    monkeypatch.setenv("PORTFOLIO_RAG_EMBEDDING_PROVIDER", "mistral")
    monkeypatch.setenv("PORTFOLIO_RAG_LLM_PROVIDER", "mistral")
    monkeypatch.setenv("PORTFOLIO_RAG_MISTRAL_API_KEY", "test-value-not-a-real-credential")
    get_settings.cache_clear()
    build = composition.build_query_components

    def offline(settings: Any, *, generation_attempts: int | None = None) -> Any:
        local = settings.model_copy(
            update={
                "embedding_provider": EmbeddingProviderName.DETERMINISTIC,
                "llm_provider": LLMProviderName.DETERMINISTIC,
            }
        )
        return build(local, generation_attempts=generation_attempts)

    monkeypatch.setattr(cli, "build_query_components", offline)
    # A known, clean release candidate: whether this checkout happens to be
    # dirty is not what these tests are about.
    monkeypatch.setattr(
        cli, "_git_state", lambda: cli.GitState("c0ffee" + "0" * 34, False, None, "5" * 64)
    )
    # A test run is not a provider run: its ledger and history stay in the test.
    monkeypatch.setattr(cli, "DEFAULT_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(cli, "DEFAULT_HISTORY", tmp_path / "history")


@pytest.mark.parametrize(
    ("configured", "named"),
    [
        ({}, "embedding and generation provider"),
        ({"EMBEDDING_PROVIDER": "mistral"}, "configured generation provider"),
        ({"LLM_PROVIDER": "cloudflare_workers_ai"}, "configured embedding provider"),
    ],
)
def test_an_end_to_end_run_refuses_a_development_stand_in(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: dict[str, str],
    named: str,
):
    """Before anything is built: a stub behind an end-to-end run answers everything."""
    for name, value in configured.items():
        monkeypatch.setenv(f"PORTFOLIO_RAG_{name}", value)
    get_settings.cache_clear()

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may be built for a refused run")

    monkeypatch.setattr(cli, "build_query_components", refuse)
    target = tmp_path / "e2e.json"

    code, out, err = _run_acceptance(capsys, "--output", str(target))

    assert code == EXIT_INVALID
    assert out == ""
    assert "development stand-in" in err
    assert named in err
    assert f"embedding provider: `{configured.get('EMBEDDING_PROVIDER', 'deterministic')}`" in err
    assert f"generation provider: `{configured.get('LLM_PROVIDER', 'deterministic')}`" in err
    assert not target.exists()


def test_the_stand_ins_are_still_welcome_in_every_other_run(capsys: pytest.CaptureFixture[str]):
    """The refusal belongs to --e2e. The offline evaluation is unchanged."""
    code, out, err = _run(capsys)

    assert code == EXIT_OK
    assert "Grounding" in out
    assert err == ""


@pytest.mark.usefixtures("real_provider_names")
def test_an_end_to_end_run_writes_the_export_and_the_summary(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    target, summary = tmp_path / "e2e.json", tmp_path / "e2e.md"

    code, out, _ = _run_acceptance(
        capsys,
        "--output",
        str(target),
        "--summary",
        str(summary),
        "--note",
        "offline stack",
    )

    data = _read_json(target)
    assert code == EXIT_OK
    assert "End to end" in out
    assert f"End-to-end export: {target}" in out
    assert data["format"] == "e2e-eval-v3"
    assert data["passed"] is True
    assert data["run"]["generation"]["provider"] == "mistral"
    assert data["run"]["notes"] == ["offline stack"]
    assert len(data["questions"]) == data["run"]["question_count"]
    assert {question["outcome"] for question in data["questions"]} == {"no_knowledge"}
    assert "**Gates: PASS**" in summary.read_text(encoding="utf-8")
    assert "offline stack" in summary.read_text(encoding="utf-8")


@pytest.mark.usefixtures("real_provider_names")
def test_an_end_to_end_run_makes_one_pass_not_two(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The retrieval pass is not run in front of it: that would double every request."""

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an end-to-end run must not make a separate retrieval pass")

    monkeypatch.setattr(cli, "run_retrieval_evaluation", refuse)
    monkeypatch.setattr(cli, "run_grounding_evaluation", refuse)

    code, _, _ = _run_acceptance(capsys, "--output", str(tmp_path / "e2e.json"))

    assert code == EXIT_OK


@pytest.mark.usefixtures("real_provider_names")
def test_an_end_to_end_export_holds_no_setting_that_was_not_named(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID", "account-that-must-not-appear")
    monkeypatch.setenv("PORTFOLIO_RAG_CLOUDFLARE_VECTORIZE_INDEX", "index-that-must-not-appear")
    get_settings.cache_clear()
    target = tmp_path / "e2e.json"

    _run_acceptance(capsys, "--output", str(target))

    text = target.read_text(encoding="utf-8")
    assert "must-not-appear" not in text
    assert "test-value-not-a-real-credential" not in text


# --- running a selection of questions ------------------------------------------


def test_a_run_can_be_narrowed_to_named_questions(capsys: pytest.CaptureFixture[str]):
    code, out, _ = _run_portfolio(
        capsys, "--question-id", "section-degree-claim", "--question-id", "direct-languages"
    )

    assert code == EXIT_OK
    assert "Selection: 2 of 49 questions, by id" in out
    assert re.search(r"^\s*questions\s+2$", out, flags=re.MULTILINE)


def test_a_question_that_is_not_in_the_suite_is_refused(capsys: pytest.CaptureFixture[str]):
    code, out, err = _run_portfolio(capsys, "--question-id", "no-such-question")

    assert code == EXIT_INVALID
    assert "no such question in this suite: no-such-question" in err
    assert out == ""


@pytest.mark.usefixtures("real_provider_names")
def test_an_end_to_end_export_says_which_questions_were_selected(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    target, summary = tmp_path / "e2e.json", tmp_path / "e2e.md"

    main(
        [
            "eval",
            "run",
            "--dataset",
            str(PORTFOLIO),
            "--e2e",
            "--tier",
            "smoke",
            "--output",
            str(target),
            "--summary",
            str(summary),
            # Given out of order; the dataset's order is what is kept.
            "--question-id",
            "ambiguous-react-role",
            "--question-id",
            "direct-wordpress-experience",
        ]
    )
    capsys.readouterr()

    data = _read_json(target)
    assert data["run"]["selected_question_ids"] == [
        "direct-wordpress-experience",
        "ambiguous-react-role",
    ]
    assert [question["id"] for question in data["questions"]] == data["run"][
        "selected_question_ids"
    ]
    assert data["run"]["question_count"] == 2
    assert data["run"]["dataset"]["question_count"] == 49
    assert "2 questions (selected by id)" in summary.read_text(encoding="utf-8")


# --- the run id ------------------------------------------------------------------


@pytest.mark.usefixtures("real_provider_names")
def test_one_run_id_names_the_header_the_export_and_every_provider_call(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The id a log line carries during the run is the id the export names."""
    seen_during_calls: list[str | None] = []
    generate = DeterministicLLMProvider.generate

    async def observed(self: DeterministicLLMProvider, request: Any) -> Any:
        seen_during_calls.append(get_request_id())
        return await generate(self, request)

    monkeypatch.setattr(DeterministicLLMProvider, "generate", observed)
    target, summary = tmp_path / "e2e.json", tmp_path / "e2e.md"

    # Below every offline similarity, so that questions reach the provider.
    _, out, _ = _run_acceptance(
        capsys,
        "--min-similarity",
        "-1",
        "--output",
        str(target),
        "--summary",
        str(summary),
    )
    data = _read_json(target)

    run_id = data["run"]["run_id"]
    assert re.fullmatch(r"[0-9a-f]{32}", run_id)
    assert seen_during_calls, "the run made provider calls"
    assert set(seen_during_calls) == {run_id}
    assert get_request_id() is None, "the binding ends with the run"
    assert f"| Run | `{run_id}` |" in summary.read_text(encoding="utf-8")
    assert f"Run:     {run_id}" in out
    calls = [call for question in data["questions"] for call in question["provider_calls"]]
    assert (
        len(calls) == data["metrics"]["provider_calls"]["total"]["calls"] == len(seen_during_calls)
    )


def test_two_runs_have_two_ids(capsys: pytest.CaptureFixture[str]):
    _, first, _ = _run(capsys, "--retrieval-only")
    _, second, _ = _run(capsys, "--retrieval-only")

    ids = [re.search(r"^Run:\s+([0-9a-f]{32})$", out, re.MULTILINE) for out in (first, second)]
    assert all(ids)
    assert ids[0].group(1) != ids[1].group(1)  # type: ignore[union-attr]


# --- the provider experiment (Paket 3) -----------------------------------------------


def _experiment(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["eval", "experiment", "--dataset", str(DATASET), *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_an_experiment_refuses_a_development_stand_in(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    target = tmp_path / "experiment.json"

    code, _, err = _experiment(
        capsys,
        "--question-id",
        "direct-http-framework",
        "--max-calls",
        "4",
        "--output",
        str(target),
    )

    assert code == EXIT_INVALID
    assert "development stand-in" in err
    assert not target.exists(), "nothing was built, so nothing was written"


@pytest.mark.usefixtures("real_provider_names")
def test_an_experiment_stopped_by_a_rate_limit_still_writes_what_it_measured(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Below every offline similarity, so the question has a context to send.
    monkeypatch.setenv("PORTFOLIO_RAG_RETRIEVAL_MIN_SIMILARITY", "-1")
    get_settings.cache_clear()
    generate = DeterministicLLMProvider.generate
    calls: list[int] = []

    async def limited(self: DeterministicLLMProvider, request: Any) -> Any:
        calls.append(1)
        if len(calls) == 2:
            raise LLMProviderError(
                "scripted rate limit",
                retryable=True,
                kind=ProviderFailureKind.RATE_LIMITED,
                status_code=429,
            )
        return await generate(self, request)

    monkeypatch.setattr(DeterministicLLMProvider, "generate", limited)
    target = tmp_path / "experiment.json"

    code, out, _ = _experiment(
        capsys,
        "--question-id",
        "direct-http-framework",
        "--repetitions",
        "3",
        "--max-calls",
        "6",
        "--output",
        str(target),
    )

    assert code == EXIT_INVALID, "the provider stopped the run"
    assert len(calls) == 2, "nothing after the rate limit, and no retry of it"
    assert "Stopped:    rate_limited after 2 calls" in out
    data = _read_json(target)
    assert data["experiment"]["stop_reason"] == "rate_limited"
    assert data["experiment"]["planned_calls"] == 6
    assert [call["variant"] for call in data["calls"]] == ["json_object", "text"]
    assert data["calls"][1]["result"] == "provider_error"
    assert data["experiment"]["experiment_id"] in out
    # Its own tier in the shared ledger: the same allocation, never the acceptance budget.
    (entry,) = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert (entry["tier"], entry["status"], entry["calls"]) == ("experiment", "aborted", 2)


@pytest.mark.parametrize(
    "argv",
    [
        ("--max-calls", "0"),
        ("--max-calls", "4", "--neuron-budget", "100"),
        ("--max-calls", "4", "--repetitions", "0"),
        ("--max-calls", "4", "--variant", "text", "--variant", "text"),
    ],
)
@pytest.mark.usefixtures("real_provider_names")
def test_an_experiment_with_unenforceable_limits_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: tuple[str, ...]
):
    target = tmp_path / "experiment.json"

    code, _, err = _experiment(
        capsys, "--question-id", "direct-http-framework", *argv, "--output", str(target)
    )

    assert code == EXIT_INVALID
    assert "Invalid experiment" in err
    assert not target.exists()


@pytest.mark.usefixtures("real_provider_names")
def test_an_internal_defect_aborts_the_run_and_still_writes_the_partial_export(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A defect in this code is not dressed up as a provider failure and not
    run past — but what was measured before it is kept, and marked partial."""
    answer = GroundedAnswerService.answer
    asked: list[int] = []

    async def defective(self: GroundedAnswerService, message: str, *args: Any) -> Any:
        asked.append(1)
        if len(asked) == 2:
            raise TypeError("a defect in this code")
        return await answer(self, message, *args)

    monkeypatch.setattr(GroundedAnswerService, "answer", defective)
    target = tmp_path / "e2e.json"

    code, _, err = _run_acceptance(capsys, "--output", str(target))

    assert code == EXIT_INTERNAL_ERROR
    assert len(asked) == 2, "nothing is asked after the defect"
    assert "Run aborted" in err and "TypeError" in err
    data = _read_json(target)
    assert data["run"]["complete"] is False
    assert data["run"]["aborted"]["error_type"] == "TypeError"
    assert len(data["questions"]) == 1
    assert data["passed"] is False


# --- Paket 4: tiers, preflight, budget -------------------------------------------------


@pytest.fixture
def workers_ai_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Settings naming Workers AI — so the real cost profile applies — and a
    composition root that must never be reached: these runs stop at preflight."""
    for name, value in (
        ("EMBEDDING_PROVIDER", "mistral"),
        ("LLM_PROVIDER", "cloudflare_workers_ai"),
        ("MISTRAL_API_KEY", "test-value-not-a-real-credential"),
        ("CLOUDFLARE_ACCOUNT_ID", "test-account"),
        ("CLOUDFLARE_WORKERS_AI_TOKEN", "test-value-not-a-real-credential"),
    ):
        monkeypatch.setenv(f"PORTFOLIO_RAG_{name}", value)
    get_settings.cache_clear()

    def never(*_: Any, **__: Any) -> Any:
        raise AssertionError("a run stopped at preflight must build nothing")

    monkeypatch.setattr(cli, "build_query_components", never)
    history = tmp_path / "history"
    history.mkdir()
    monkeypatch.setattr(cli, "DEFAULT_LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(cli, "DEFAULT_HISTORY", history)
    return history


def _measured_history(directory: Path, *, generation_tokens: tuple[int, int]) -> None:
    """One earlier v3 export of five questions with measured usage."""

    def call(call_type: str, tokens: tuple[int, int]) -> dict[str, Any]:
        return {
            "call_type": call_type,
            "attempt": 1,
            "regeneration": False,
            "model": "@cf/openai/gpt-oss-120b",
            "response_format": "json_object",
            "elapsed_seconds": 2.0,
            "result": "parsed",
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
        }

    payload = {
        "format": "e2e-eval-v3",
        "run": {
            "generation": {
                "provider": "cloudflare_workers_ai",
                "model": "@cf/openai/gpt-oss-120b",
            }
        },
        "questions": [
            {
                "id": f"q-{n}",
                "provider_calls": [
                    call("generation", generation_tokens),
                    call("grounding_check", (800, 10)),
                ],
            }
            for n in range(5)
        ],
    }
    (directory / "earlier.json").write_text(json.dumps(payload), encoding="utf-8")


def test_an_end_to_end_run_against_a_real_provider_names_its_tier(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, workers_ai_names: Path
):
    code, _, err = _run(capsys, "--e2e", "--output", str(tmp_path / "e2e.json"))

    assert code == EXIT_INVALID
    assert "needs --tier smoke or --tier acceptance" in err


def test_an_answering_pass_against_a_real_provider_is_only_a_tiered_e2e_run(
    capsys: pytest.CaptureFixture[str], workers_ai_names: Path
):
    code, _, err = _run(capsys)

    assert code == EXIT_INVALID
    assert "--e2e --tier smoke|acceptance" in err


def test_acceptance_without_measured_usage_is_red_and_builds_nothing(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, workers_ai_names: Path
):
    code, out, err = _run_acceptance(capsys, "--output", str(tmp_path / "e2e.json"))

    assert code == EXIT_INVALID
    assert "ACCEPTANCE PREFLIGHT" in out
    assert re.search(r"budget zone\s+RED", out)
    assert re.search(r"actual remaining\s+unknown", out)
    assert "budget zone RED" in err
    assert not (tmp_path / "e2e.json").exists()


def test_yellow_starts_only_when_confirmed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, workers_ai_names: Path
):
    # 24 questions at this history's cost land above half the budget, below the reserve.
    _measured_history(workers_ai_names, generation_tokens=(4000, 450))
    target = str(tmp_path / "e2e.json")

    refused, out, err = _run_acceptance(capsys, "--output", target)
    confirmed, out_confirmed, _ = _run_acceptance(
        capsys,
        "--allow-yellow",
        "--preflight-only",
        "--output",
        target,
    )

    assert re.search(r"budget zone\s+YELLOW", out)
    assert refused == EXIT_INVALID and "--allow-yellow" in err
    assert confirmed == EXIT_OK
    assert "Preflight only: no provider was called." in out_confirmed


def _spent_today(ledger: Path, neurons: float) -> None:
    """A smoke run recorded in the ledger today, which the preflight charges against."""
    line = {
        "date_utc": datetime.now(UTC).date().isoformat(),
        "run_id": "s" * 32,
        "tier": "smoke",
        "status": "complete",
        "calls": 4,
        "input_tokens": 1,
        "output_tokens": 1,
        "estimated_neurons": neurons,
        "usage_coverage": 1.0,
        "artifact": None,
        "smoke_passed": True,
    }
    ledger.write_text(json.dumps(line) + "\n", encoding="utf-8")


def test_the_reserve_override_belongs_to_a_tiered_run(capsys: pytest.CaptureFixture[str]):
    code, _, err = _run(capsys, "--retrieval-only", "--override-budget-reserve")

    assert code == EXIT_INVALID
    assert "--override-budget-reserve belongs to an --e2e --tier run" in err


def test_the_reserve_override_starts_a_run_inside_the_nominal_budget_and_says_so(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, workers_ai_names: Path
):
    # The same history as above: 24 questions above half the budget, inside it.
    _measured_history(workers_ai_names, generation_tokens=(4000, 450))
    _spent_today(tmp_path / "ledger.jsonl", 2_000.0)
    target = str(tmp_path / "e2e.json")

    refused, out, err = _run_acceptance(capsys, "--allow-yellow", "--output", target)
    started, out_override, err_override = _run_acceptance(
        capsys,
        "--allow-yellow",
        "--override-budget-reserve",
        "--preflight-only",
        "--output",
        target,
    )

    assert refused == EXIT_INVALID
    assert "would not leave the minimum reserve" in err
    assert "WARNING" not in out
    assert started == EXIT_OK
    assert "WARNING: minimum reserve manually overridden for this run" in out_override
    assert "WARNING: minimum reserve manually overridden for this run" in err_override
    assert re.search(r"budget override\s+minimum_reserve", out_override)
    assert "Preflight only: no provider was called." in out_override


def test_the_reserve_override_never_passes_the_nominal_daily_budget(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, workers_ai_names: Path
):
    _measured_history(workers_ai_names, generation_tokens=(4000, 450))
    _spent_today(tmp_path / "ledger.jsonl", 9_000.0)

    code, _, err = _run_acceptance(
        capsys,
        "--allow-yellow",
        "--override-budget-reserve",
        "--preflight-only",
        "--output",
        str(tmp_path / "e2e.json"),
    )

    assert code == EXIT_INVALID
    assert "exceed the nominal daily budget" in err


def test_an_acceptance_run_never_takes_question_ids(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, workers_ai_names: Path
):
    code, _, err = _run_acceptance(
        capsys,
        "--question-id",
        "direct-http-framework",
        "--output",
        str(tmp_path / "e2e.json"),
    )

    assert code == EXIT_INVALID
    assert "--question-id is for a smoke run" in err


@pytest.mark.usefixtures("real_provider_names")
def test_a_smoke_run_asks_the_provider_smoke_questions_and_records_itself(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    target = tmp_path / "smoke.json"

    code, out, _ = _run_portfolio_e2e(capsys, "--tier", "smoke", "--output", str(target))

    data = _read_json(target)
    assert code in (EXIT_OK, EXIT_INVALID)
    assert "SMOKE PREFLIGHT" in out
    assert {q["id"] for q in data["questions"]} == {
        "broad-deployment",
        "multi-api-backend-evidence",
        "section-lead-flow",
        "ambiguous-frontend-technologies",
        "unknown-kubernetes-production",
    }
    assert data["operations"]["tier"] == "smoke"
    assert data["operations"]["status"] == "complete"
    assert data["operations"]["smoke"] is not None
    ledger = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 1
    assert json.loads(ledger[0])["tier"] == "smoke"


@pytest.mark.usefixtures("real_provider_names")
def test_a_rate_limit_stops_the_run_writes_what_it_measured_and_never_restarts(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[int] = []

    async def limited(self: DeterministicLLMProvider, request: Any) -> Any:
        calls.append(1)
        raise LLMProviderError(
            "scripted", retryable=True, kind=ProviderFailureKind.RATE_LIMITED, status_code=429
        )

    monkeypatch.setattr(DeterministicLLMProvider, "generate", limited)
    target = tmp_path / "e2e.json"

    # Below every offline similarity, so that the first question reaches the provider.
    code, _, err = _run_acceptance(
        capsys,
        "--min-similarity",
        "-1",
        "--output",
        str(target),
    )

    # An acceptance run that stopped is not a release acceptance.
    assert code == EXIT_ACCEPTANCE_FAILED
    assert len(calls) == 1, "one call, then nothing — no next question, no rerun"
    assert "rate_limited" in err and "not restarted automatically" in err
    data = _read_json(target)
    assert data["run"]["complete"] is False
    assert data["run"]["aborted"]["reason"] == "rate_limited"
    assert len(data["questions"]) == 1
    entry = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert (entry["status"], entry["abort_reason"]) == ("aborted", "rate_limited")


def _run_portfolio_e2e(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["eval", "run", "--dataset", str(PORTFOLIO), "--e2e", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err
