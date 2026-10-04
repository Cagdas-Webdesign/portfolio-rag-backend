"""`eval run --tier acceptance` and `eval validate-acceptance`, driven as a shell would.

The offline stack stands in for the providers (see `real_provider_names`), so
an acceptance run here spends nothing and calls nothing. What is checked is
the governance around it: the provenance the artifact carries, the verdict,
the validator, and the rule that a commit is not rerun until it is green.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag import __version__, cli
from portfolio_rag.cli import (
    EXIT_ACCEPTANCE_FAILED,
    EXIT_INTERNAL_ERROR,
    EXIT_INVALID,
    EXIT_OK,
    GitState,
    main,
)
from portfolio_rag.evaluation import E2EFailure, E2EReport, run_e2e_evaluation
from portfolio_rag.evaluation.e2e import AbortReason, RunAbort
from tests.integration.test_cli_eval import (  # noqa: F401 - fixtures, used by name
    _local_defaults,
    real_provider_names,
)

DATASET = Path("evaluation/portfolio-questions.yaml")
COMMIT = "c0ffee" + "0" * 34
SOURCE = "5" * 64


def _tree(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dirty: bool | None,
    commit: str | None = COMMIT,
    source: str | None = SOURCE,
    tag: str | None = None,
) -> None:
    monkeypatch.setattr(cli, "_git_state", lambda: GitState(commit, dirty, tag, source))


def _accept(capsys: pytest.CaptureFixture[str], output: Path, *argv: str) -> tuple[int, str, str]:
    code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(DATASET),
            "--e2e",
            "--tier",
            "acceptance",
            "--output",
            str(output),
            *argv,
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _validate(capsys: pytest.CaptureFixture[str], artifact: Path, *argv: str) -> tuple[int, str]:
    code = main(["eval", "validate-acceptance", str(artifact), *argv])
    return code, capsys.readouterr().out


def _read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.mark.usefixtures("real_provider_names")
def test_a_clean_acceptance_run_writes_a_release_acceptance_the_validator_accepts(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    target, summary = tmp_path / "acceptance.json", tmp_path / "acceptance.md"

    code, out, _ = _accept(capsys, target, "--summary", str(summary))

    data = _read(target)
    release = data["release_acceptance"]
    assert code == EXIT_OK
    assert "Release acceptance: PASS" in out
    assert release["verdict"] == "PASS"
    assert release["commit"] == COMMIT
    assert data["run"]["tier"] == "acceptance"
    suite_file = Path("evaluation/portfolio-questions.release-acceptance.yaml")
    assert data["run"]["suite"] == "release-acceptance"
    assert data["run"]["suite_version"] == 1
    assert data["run"]["suite_path"] == suite_file.as_posix()
    assert data["run"]["suite_sha256"] == hashlib.sha256(suite_file.read_bytes()).hexdigest()
    assert data["run"]["question_count"] == 24
    assert data["run"]["dataset"]["question_count"] == 49
    assert len(data["questions"]) == 24
    assert "**Release acceptance: PASS**" in summary.read_text(encoding="utf-8")

    code, report = _validate(capsys, target)
    assert code == EXIT_OK
    assert "Accepted: a passing release acceptance that may be published." in report


@pytest.mark.usefixtures("real_provider_names")
def test_a_dirty_tree_run_is_readable_for_development_and_refused_for_publication(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=True)
    target = tmp_path / "dev.json"

    code, out, _ = _accept(capsys, target)

    # The gates pass; the release verdict does not, and that is what the exit reports.
    assert code == EXIT_ACCEPTANCE_FAILED
    assert "Release acceptance: FAIL" in out
    assert _read(target)["run"]["git_dirty"] is True

    code, report = _validate(capsys, target)
    assert code == EXIT_INVALID
    assert "Rejected for publication" in report
    assert "the working tree was not clean" in report

    code, report = _validate(capsys, target, "--development")
    assert code == EXIT_OK
    assert "Accepted as a development artifact" in report


@pytest.mark.usefixtures("real_provider_names")
def test_a_smoke_run_records_the_suite_it_actually_ran(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The tier picks the suite; the artifact must name that one, not --suite's default."""
    _tree(monkeypatch, dirty=False)
    target = tmp_path / "smoke.json"
    dataset = Path("evaluation/portfolio-questions.yaml")

    main(
        [
            "eval",
            "run",
            "--dataset",
            str(dataset),
            "--e2e",
            "--tier",
            "smoke",
            "--output",
            str(target),
        ]
    )
    capsys.readouterr()

    run_ = _read(target)["run"]
    assert run_["suite"] == "provider-smoke"
    assert run_["suite_path"] == "evaluation/portfolio-questions.provider-smoke.yaml"
    assert _read(target)["release_acceptance"]["verdict"] == "FAIL"


@pytest.mark.usefixtures("real_provider_names")
def test_a_commit_that_already_has_an_acceptance_run_is_not_run_again(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    first = tmp_path / "first.json"
    assert _accept(capsys, first)[0] == EXIT_OK
    first_id = _read(first)["run"]["run_id"]

    code, _, err = _accept(capsys, tmp_path / "second.json")
    assert code == EXIT_INVALID
    assert f"already has an acceptance run ({first_id})" in err
    assert not (tmp_path / "second.json").exists()

    code, _, err = _accept(
        capsys, tmp_path / "rerun.json", "--rerun-of", first_id, "--note", "provider 503 burst"
    )
    assert code == EXIT_INVALID
    assert "a passing run is not rerun" in err


@pytest.mark.usefixtures("real_provider_names")
def test_the_ledger_records_the_commit_and_the_gates(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    _accept(capsys, tmp_path / "run.json")

    (line,) = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(line)
    assert entry["commit_sha"] == COMMIT
    assert entry["git_dirty"] is False
    assert entry["failed_gates"] == []
    assert entry["rerun_of"] is None


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--rerun-of", "run-1"], "--rerun-of needs a --note"),
        (["--retrieval-only", "--rerun-of", "run-1", "--note", "x"], "--rerun-of belongs to"),
    ],
)
def test_a_rerun_must_be_an_acceptance_run_with_a_documented_reason(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, argv: list[str], message: str
):
    if "--retrieval-only" in argv:
        code = main(["eval", "run", "--dataset", str(DATASET), *argv])
    else:
        code = main(
            [
                "eval",
                "run",
                "--dataset",
                str(DATASET),
                "--e2e",
                "--tier",
                "acceptance",
                "--output",
                str(tmp_path / "x.json"),
                *argv,
            ]
        )

    assert code == EXIT_INVALID
    assert message in capsys.readouterr().err


def test_the_validator_rejects_a_file_that_is_not_json(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    code = main(["eval", "validate-acceptance", str(broken)])

    assert code == EXIT_INVALID
    assert "could not be read as JSON" in capsys.readouterr().err


@pytest.mark.usefixtures("real_provider_names")
def test_the_validator_rejects_a_manipulated_artifact_in_either_mode(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=True)
    target = tmp_path / "dev.json"
    _accept(capsys, target)
    data = _read(target)
    data["run"]["git_dirty"] = False  # a dirty run relabelled clean after the fact
    target.write_text(json.dumps(data), encoding="utf-8")

    for mode in ((), ("--development",)):
        code, report = _validate(capsys, target, *mode)
        assert code == EXIT_INVALID
        assert "does not match what the artifact's own data says" in report


@pytest.mark.usefixtures("real_provider_names")
def test_the_validator_finds_a_configured_credential_without_printing_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    target = tmp_path / "run.json"
    _accept(capsys, target)
    data = _read(target)
    # The fixture's placeholder credential, written into a free-text field.
    data["run"]["notes"] = ["test-value-not-a-real-credential"]
    target.write_text(json.dumps(data), encoding="utf-8")

    code, report = _validate(capsys, target)

    assert code == EXIT_INVALID
    assert "a configured credential appears at run.notes.0" in report
    assert "test-value-not-a-real-credential" not in report


# --- the acceptance exit code ---------------------------------------------------


def _alter_report(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[E2EReport], E2EReport]
) -> None:
    """Let the offline run happen, then change its report the way a real run
    could have ended."""

    async def altered(*args: Any, **kwargs: Any) -> E2EReport:
        return change(await run_e2e_evaluation(*args, **kwargs))

    monkeypatch.setattr(cli, "run_e2e_evaluation", altered)


def _with_failure(failure: E2EFailure) -> Callable[[E2EReport], E2EReport]:
    def change(report: E2EReport) -> E2EReport:
        first, *rest = report.records
        return replace(report, records=(replace(first, failures=(failure,)), *rest))

    return change


@pytest.mark.usefixtures("real_provider_names")
@pytest.mark.parametrize(
    "change",
    [
        pytest.param(_with_failure(E2EFailure.PIPELINE_ERROR), id="pipeline-gate"),
        pytest.param(_with_failure(E2EFailure.UNVERIFIED_CITATION), id="citation-gate"),
        pytest.param(_with_failure(E2EFailure.ANSWERED_UNKNOWN), id="refusal-gate"),
        pytest.param(_with_failure(E2EFailure.LABEL_LEAK), id="internal-leak"),
        pytest.param(lambda report: replace(report, records=report.records[:1]), id="incomplete"),
        pytest.param(
            lambda report: replace(
                report, aborted=RunAbort(question_id="x", reason=AbortReason.BUDGET_GUARD)
            ),
            id="aborted",
        ),
    ],
)
def test_a_failed_acceptance_exits_with_its_own_code(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Any,
):
    _tree(monkeypatch, dirty=False)
    _alter_report(monkeypatch, change)

    code, out, _ = _accept(capsys, tmp_path / "run.json")

    assert code == EXIT_ACCEPTANCE_FAILED
    assert "Release acceptance: FAIL" in out


@pytest.mark.usefixtures("real_provider_names")
@pytest.mark.parametrize(
    ("tree", "reason"),
    [
        ({"dirty": None, "commit": None, "source": None}, "missing provenance"),
        ({"dirty": False, "source": None}, "release_source_identity"),
        ({"dirty": False, "tag": "v1.0.1"}, "does not match project_version"),
    ],
)
def test_missing_or_contradictory_provenance_fails_the_acceptance_exit(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree: dict[str, Any],
    reason: str,
):
    _tree(monkeypatch, **tree)

    code, out, _ = _accept(capsys, tmp_path / "run.json")

    assert code == EXIT_ACCEPTANCE_FAILED
    assert reason in out


@pytest.mark.usefixtures("real_provider_names")
def test_an_internal_defect_is_still_reported_as_a_crash(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    _alter_report(
        monkeypatch,
        lambda report: replace(
            report,
            aborted=RunAbort(
                question_id="x", reason=AbortReason.INTERNAL_DEFECT, error_type="KeyError"
            ),
        ),
    )

    code, _, _ = _accept(capsys, tmp_path / "run.json")

    assert code == EXIT_INTERNAL_ERROR


@pytest.mark.usefixtures("real_provider_names")
def test_a_smoke_run_on_a_dirty_tree_keeps_its_exit_code(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Only release acceptance has the hard exit: a smoke run is a health check."""
    _tree(monkeypatch, dirty=True)

    code = main(
        [
            "eval",
            "run",
            "--dataset",
            "evaluation/portfolio-questions.yaml",
            "--e2e",
            "--tier",
            "smoke",
            "--output",
            str(tmp_path / "smoke.json"),
        ]
    )
    capsys.readouterr()

    assert code == EXIT_OK


@pytest.mark.usefixtures("real_provider_names")
def test_the_artifact_records_the_project_version_and_the_source_identity(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False, tag=f"v{__version__}")
    target = tmp_path / "run.json"

    code, _, _ = _accept(capsys, target)

    data = _read(target)
    assert code == EXIT_OK
    assert data["run"]["project_version"] == __version__
    assert data["run"]["release_source_identity"] == SOURCE
    assert data["run"]["git_tag"] == f"v{__version__}"
    assert data["release_acceptance"]["project_version"] == __version__
    assert data["release_acceptance"]["release_source_identity"] == SOURCE


# --- the release candidate is its source, not its sha ---------------------------


@pytest.mark.usefixtures("real_provider_names")
def test_a_results_only_commit_is_not_a_new_candidate(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    first = tmp_path / "first.json"
    _accept(capsys, first)

    # A new commit — the artifact committed — over the same source.
    _tree(monkeypatch, dirty=False, commit="b" * 40)
    code, _, err = _accept(capsys, tmp_path / "second.json")

    assert code == EXIT_INVALID
    assert f"source {SOURCE[:12]} already has an acceptance run" in err


@pytest.mark.usefixtures("real_provider_names")
def test_a_changed_source_is_a_new_candidate(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    _accept(capsys, tmp_path / "first.json")

    _tree(monkeypatch, dirty=False, commit="b" * 40, source="6" * 64)
    code, _, _ = _accept(capsys, tmp_path / "second.json")

    assert code == EXIT_OK


@pytest.mark.usefixtures("real_provider_names")
def test_the_ledger_records_the_source_identity(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _tree(monkeypatch, dirty=False)
    _accept(capsys, tmp_path / "run.json")

    (line,) = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["release_source_identity"] == SOURCE
