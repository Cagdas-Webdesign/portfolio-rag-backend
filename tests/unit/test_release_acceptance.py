"""Release acceptance: one export, one verdict, decided by frozen rules.

A real (scripted) end-to-end run is exported and judged. The failing cases are
made the way a real run would make them — a pipeline error, an answered
unknown — or, for the states the pipeline is built to prevent, by editing a
real record, because the point is that the verdict would notice them. The
manipulation cases edit the JSON after the fact, because that is what the
validator exists to catch.
"""

from __future__ import annotations

import ast
import copy
import re
import subprocess
import tomllib
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag import __version__
from portfolio_rag.cli import _git_state
from portfolio_rag.evaluation import (
    GATES,
    E2EFailure,
    E2EReport,
    EvaluationQuestion,
    QuestionCategory,
    SuiteIdentity,
    export_e2e,
    render_summary,
    run_e2e_evaluation,
)
from portfolio_rag.evaluation.acceptance import (
    ACCEPTANCE_PROTOCOL_VERSION,
    INTERNAL_LEAKS_GATE,
    RELEASE_GATES,
    SAFETY_INVARIANTS,
    SOURCE_IDENTITY_EXCLUDES,
    SOURCE_IDENTITY_INCLUDES,
    AcceptanceAttempt,
    assess,
    rerun_blockers,
    source_identity,
    validate_artifact,
    with_release_acceptance,
)
from portfolio_rag.evaluation.budget import (
    BudgetPolicy,
    Ledger,
    LedgerEntry,
    LedgerReading,
    UsageHistory,
)
from portfolio_rag.evaluation.checkpoint import identity_digest, run_identity
from portfolio_rag.evaluation.e2e import AbortReason, RunAbort
from portfolio_rag.evaluation.operations import Tier, acceptance_attempts, preflight
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from tests.doubles import FailingLLMProvider, ScriptedLLMProvider, ScriptedReply, grounded
from tests.e2e_stack import ANSWERABLE, dataset_of, e2e_service, label_of, run_metadata
from tests.support import run

COMMIT = "c0ffee" + "0" * 34
SOURCE = "5" * 64
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

UNKNOWN = EvaluationQuestion(
    id="q-salary",
    category=QuestionCategory.UNKNOWN,
    question="What is the salary of the backend API author?",
)
DECLINED = ScriptedReply(answer="The passages do not cover this.", sources=())


def _report(*replies: ScriptedReply) -> E2EReport:
    label = label_of("stack", ANSWERABLE)
    script = replies or (grounded(f"FastAPI [{label}].", label), DECLINED)
    llm = ScriptedLLMProvider(*script)
    return run(run_e2e_evaluation(dataset_of(ANSWERABLE, UNKNOWN), e2e_service(llm)))


def _artifact(
    report: E2EReport | None = None, *, dirty: bool | None = False, **changes: Any
) -> dict[str, Any]:
    """A release-acceptance artifact as the CLI writes one: clean and passing,
    unless told otherwise."""
    report = report if report is not None else _report()
    dataset = dataset_of(*(record.question for record in report.records))
    metadata = run_metadata(dataset)
    metadata = replace(
        metadata,
        retrieval=replace(
            metadata.retrieval,
            run_id="run-0001",
            git_revision=COMMIT,
            git_dirty=dirty,
            source_identity=SOURCE,
            suite_identity=SuiteIdentity(version=1, sha256="e" * 64, path="suite.yaml"),
        ),
        **{"tier": Tier.ACCEPTANCE.value, **changes},
    )
    return with_release_acceptance(_executed(export_e2e(report, metadata)))


def _executed(payload: dict[str, Any]) -> dict[str, Any]:
    """*payload* with the one execution segment an uninterrupted CLI run records."""
    run_ = payload["run"]
    ids = [question["id"] for question in payload["questions"]]
    calls = sum(len(question["provider_calls"]) for question in payload["questions"])
    run_.update(
        {
            "logical_run_id": run_["run_id"],
            "original_started_at": run_["generated_at"],
            "completed_at": run_["generated_at"] if run_["complete"] else None,
            "resume_count": 0,
            "execution_segments": [
                {
                    "segment": 1,
                    "run_id": run_["run_id"],
                    "identity_sha256": identity_digest(run_identity(run_, ids)),
                    "status": "complete" if run_["complete"] else "paused",
                    "stop_reason": None,
                    "completed_question_ids": ids,
                    "interrupted_question_ids": [],
                    "calls": calls,
                    "interrupted_calls": 0,
                    "estimated_neurons": 0.0,
                }
            ],
            "rate_limit_events": [],
        }
    )
    return payload


# --- the verdict ----------------------------------------------------------------


def test_a_clean_complete_passing_run_is_a_release_acceptance():
    payload = _artifact()
    release = payload["release_acceptance"]

    assert release["verdict"] == "PASS"
    assert release["reasons"] == []
    assert all(release["conditions"].values())
    assert release["gates"] == {
        "no_pipeline_errors": "PASS",
        "citations_verified": "PASS",
        "unanswerable_refused": "PASS",
        "nothing_internal_published": "PASS",
        "internal_leaks": 0,
    }
    assert release["commit"] == COMMIT
    assert release["git_dirty"] is False
    assert release["protocol"] == ACCEPTANCE_PROTOCOL_VERSION
    assert release["scope"]["deployed_image_tested"] is False
    assert release["safety_invariants"] == sorted(SAFETY_INVARIANTS)


def test_the_artifact_carries_the_provenance_a_release_needs():
    run_ = _artifact()["run"]

    assert run_["run_id"] == "run-0001"
    assert run_["git_revision"] == COMMIT
    assert run_["git_dirty"] is False
    assert run_["git_dirty_excludes"] == ["evaluation/results"]
    assert run_["tier"] == "acceptance"
    assert (run_["suite_version"], run_["suite_sha256"]) == (1, "e" * 64)
    assert run_["retrieval_policy"]["visibility"] == "public"
    generation = run_["generation"]
    assert generation["max_output_tokens"] == generation["output_reserve_tokens"]
    assert generation["generation_attempt_limit"] == 2
    for name in ("provider", "model", "prompt_version", "grounding_check_version"):
        assert generation[name]


def test_a_broken_refusal_gate_fails_the_release():
    label = label_of("stack", ANSWERABLE)
    report = _report(grounded(f"FastAPI [{label}].", label), grounded("It says so [S1].", "S1"))

    release = _artifact(report)["release_acceptance"]

    assert release["verdict"] == "FAIL"
    assert release["gates"]["unanswerable_refused"] == "FAIL"
    assert "gate unanswerable_refused FAIL" in release["reasons"]


def test_a_broken_citation_gate_fails_the_release():
    report = _report()
    first, second = report.records
    unverified = replace(first, verified_citations=0, failures=(E2EFailure.UNVERIFIED_CITATION,))

    release = _artifact(replace(report, records=(unverified, second)))["release_acceptance"]

    assert release["verdict"] == "FAIL"
    assert release["gates"]["citations_verified"] == "FAIL"


def test_one_pipeline_error_fails_the_release():
    """A provider outlier is still a pipeline error: the run stays red."""
    failing = FailingLLMProvider(LLMProviderError(ProviderFailureKind.UNREACHABLE))
    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(failing)))

    release = _artifact(report)["release_acceptance"]

    assert release["verdict"] == "FAIL"
    assert release["gates"]["no_pipeline_errors"] == "FAIL"


def test_an_internal_leak_fails_the_release_and_is_counted():
    report = _report()
    first, second = report.records
    leaked = replace(first, answer="FastAPI [S1].", failures=(E2EFailure.LABEL_LEAK,))

    payload = _artifact(replace(report, records=(leaked, second)))
    release = payload["release_acceptance"]

    assert release["verdict"] == "FAIL"
    assert release["gates"][INTERNAL_LEAKS_GATE] == 1
    assert release["gates"]["nothing_internal_published"] == "FAIL"
    assert release["conditions"]["no_internal_leaks"] is False
    # The publication check sees the marker in the answer as well.
    assert any("internal context marker" in p for p in validate_artifact(payload).problems)


def test_an_aborted_run_is_never_a_release_acceptance():
    report = replace(
        _report(), aborted=RunAbort(question_id="q-api", reason=AbortReason.RATE_LIMITED)
    )

    release = _artifact(report)["release_acceptance"]

    assert release["verdict"] == "FAIL"
    assert release["conditions"]["run_complete"] is False
    assert release["conditions"]["run_not_aborted"] is False


def test_a_run_that_did_not_record_every_question_is_incomplete():
    payload = _artifact()
    payload["questions"] = payload["questions"][:1]

    assessed = assess(payload)

    assert assessed["verdict"] == "FAIL"
    assert assessed["conditions"]["every_question_recorded"] is False


def test_a_dirty_tree_fails_the_release_but_stays_a_readable_development_artifact():
    payload = _artifact(dirty=True)
    check = validate_artifact(payload)

    assert payload["release_acceptance"]["verdict"] == "FAIL"
    assert payload["release_acceptance"]["git_dirty"] is True
    assert "the working tree was not clean (or its state is unknown)" in check.reasons
    assert check.consistent
    assert check.accepted(publication=True) is False
    assert check.accepted(publication=False) is True


def test_an_unknown_tree_state_is_not_clean():
    assert _artifact(dirty=None)["release_acceptance"]["conditions"]["clean_tree"] is False


@pytest.mark.parametrize(
    ("path", "named"),
    [
        (("git_revision",), "git_revision"),
        (("suite_version",), "suite_version"),
        (("generation", "model"), "generation.model"),
        (("retrieval_policy", "min_similarity"), "retrieval_policy.min_similarity"),
        (("embedding", "model"), "embedding.model"),
        (("generation", "prompt_version"), "generation.prompt_version"),
    ],
)
def test_missing_provenance_fails_the_release(path: tuple[str, ...], named: str):
    payload = _artifact()
    target = payload["run"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = None

    assessed = assess(payload)

    assert assessed["verdict"] == "FAIL"
    assert assessed["conditions"]["provenance_complete"] is False
    assert any(named in reason for reason in assessed["reasons"])


def test_a_short_revision_is_not_a_commit():
    payload = _artifact()
    payload["run"]["git_revision"] = "c0ffee"

    assert assess(payload)["conditions"]["provenance_complete"] is False


def test_a_smoke_run_or_a_selection_is_not_a_release_acceptance():
    assert _artifact(tier="smoke")["release_acceptance"]["verdict"] == "FAIL"
    narrowed = _artifact(selected_question_ids=("q-api",))["release_acceptance"]
    assert narrowed["conditions"]["whole_suite"] is False


# --- the frozen rules -----------------------------------------------------------


def test_the_release_gates_are_frozen():
    """Changing a gate is a new protocol version, decided before a run. v2 added
    one condition (execution segments) and changed no gate."""
    assert ACCEPTANCE_PROTOCOL_VERSION == "release-acceptance-v2"
    assert tuple(GATES) == RELEASE_GATES
    assert {name: sorted(f.value for f in failures) for name, failures in GATES.items()} == {
        "no_pipeline_errors": ["empty_answer", "pipeline_error"],
        "citations_verified": [
            "answered_without_citation",
            "dangling_citation_mark",
            "invented_source_label",
            "unverified_citation",
        ],
        "unanswerable_refused": ["answered_unknown", "citation_on_refusal"],
        "nothing_internal_published": ["internal_label_leak", "internal_passage_retrieved"],
    }
    assert INTERNAL_LEAKS_GATE == "internal_leaks"


def test_every_safety_invariant_names_tests_that_exist():
    """The freeze references tests; a renamed or deleted one breaks this."""
    assert set(SAFETY_INVARIANTS) == {
        "strict_generation_parser",
        "citation_validation",
        "grounding_validation",
        "fail_closed_on_technical_failure",
        "controlled_refusal",
        "public_only_retrieval",
        "conversation_is_not_evidence",
        "recovery_passes_safety_again",
        "telemetry_stays_internal",
    }
    for invariant, nodes in SAFETY_INVARIANTS.items():
        assert nodes, invariant
        for node in nodes:
            file, function = node.split("::")
            tree = ast.parse((REPOSITORY_ROOT / file).read_text(encoding="utf-8"))
            defined = {
                item.name
                for item in tree.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
            }
            assert function in defined, node


# --- the validator --------------------------------------------------------------


def test_the_validator_accepts_a_good_artifact():
    check = validate_artifact(_artifact())

    assert check.verdict == "PASS"
    assert check.problems == ()
    assert check.accepted(publication=True)


def test_a_verdict_flipped_to_pass_is_caught():
    payload = _artifact(dirty=True)
    payload["release_acceptance"]["verdict"] = "PASS"
    payload["release_acceptance"]["conditions"]["clean_tree"] = True

    check = validate_artifact(payload)

    assert check.verdict == "FAIL"
    assert "release_acceptance.verdict does not match what the artifact's own data says" in (
        check.problems
    )
    assert not check.accepted(publication=True)
    assert not check.accepted(publication=False)


def test_gates_edited_green_over_recorded_failures_are_caught():
    label = label_of("stack", ANSWERABLE)
    payload = _artifact(
        _report(grounded(f"FastAPI [{label}].", label), grounded("It says so [S1].", "S1"))
    )
    payload["gates"]["unanswerable_refused"] = True
    payload["passed"] = True

    problems = validate_artifact(payload).problems

    assert "the stored gates do not match the questions' recorded failures" in problems
    assert "`passed` does not match the gates and the run status" in problems


def test_a_failure_erased_from_a_question_is_caught_by_the_failure_index():
    label = label_of("stack", ANSWERABLE)
    payload = _artifact(
        _report(grounded(f"FastAPI [{label}].", label), grounded("It says so [S1].", "S1"))
    )
    payload["questions"][1]["failures"] = []

    problems = validate_artifact(payload).problems

    assert "the failure index does not match the questions' recorded failures" in problems


def test_a_hidden_pipeline_error_is_caught_by_the_counts():
    payload = _artifact()
    payload["metrics"]["robustness"]["pipeline_errors"] = 1

    problems = validate_artifact(payload).problems

    assert "the pipeline error count does not match the recorded questions" in problems


def test_a_changed_gate_set_is_caught():
    payload = _artifact()
    del payload["gates"]["unanswerable_refused"]

    assert (
        "the gate set differs from the frozen release gates" in validate_artifact(payload).problems
    )


def test_an_artifact_from_another_protocol_is_not_judged_by_this_one():
    payload = _artifact()
    payload["release_acceptance"]["protocol"] = "release-acceptance-v0"

    assert not validate_artifact(payload).consistent


@pytest.mark.parametrize("section", ["run", "gates", "questions", "release_acceptance"])
def test_an_incomplete_artifact_is_rejected(section: str):
    payload = _artifact()
    del payload[section]

    check = validate_artifact(payload)

    assert check.verdict is None
    assert f"missing section `{section}`" in check.problems


def test_something_that_is_not_an_export_is_rejected():
    assert validate_artifact(["not", "an", "export"]).verdict is None
    other = _artifact()
    other["format"] = "retrieval-eval-v1"
    assert not validate_artifact(other).consistent


# --- publication safety ---------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        "Authorization: Bearer abcdefghijklmnop",
        "key sk-abcdefghijklmnopqrstuvwx",
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_a_credential_shaped_value_blocks_publication(note: str):
    payload = _artifact(notes=(note,))

    check = validate_artifact(payload)

    assert not check.accepted(publication=True)
    assert "credential-shaped value at run.notes.0" in check.problems


def test_a_configured_credential_anywhere_blocks_publication_without_being_echoed():
    secret = "configured-credential-value"  # noqa: S105 - a placeholder, not a credential
    payload = _artifact(notes=(f"ran with {secret}",))

    check = validate_artifact(payload, secrets=[secret])

    assert "a configured credential appears at run.notes.0" in check.problems
    assert all(secret not in problem for problem in check.problems)


@pytest.mark.parametrize("field", ["prompt", "messages", "conversation", "authorization"])
def test_a_field_that_holds_internals_blocks_publication(field: str):
    payload = _artifact()
    payload["questions"][0][field] = "anything"

    assert (
        f"forbidden field `{field}` at questions.0.{field}" in validate_artifact(payload).problems
    )


def test_a_clean_artifact_has_no_publication_problem():
    payload = _artifact()

    assert validate_artifact(copy.deepcopy(payload), secrets=["not-in-the-file"]).problems == ()


# --- the summary ----------------------------------------------------------------


def test_the_summary_carries_the_release_section():
    summary = render_summary(_artifact())

    assert "## Release acceptance" in summary
    assert "**Release acceptance: PASS** (protocol `release-acceptance-v2`)" in summary
    assert f"| Commit | `{COMMIT}` |" in summary
    assert f"| Release source identity | `{SOURCE}` |" in summary
    assert f"| Project version | {__version__} |" in summary
    assert "| Working tree | clean |" in summary
    assert "| internal_leaks | 0 |" in summary
    assert "deployed image tested: no" in summary


def test_the_summary_says_why_a_release_failed():
    summary = render_summary(_artifact(dirty=True))

    assert "**Release acceptance: FAIL**" in summary
    assert "| Working tree | dirty |" in summary
    assert "- the working tree was not clean (or its state is unknown)" in summary


# --- the rerun rule -------------------------------------------------------------


def _attempt(
    run_id: str,
    *,
    failed: tuple[str, ...] | None = ("no_pipeline_errors",),
    status: str = "complete",
    abort: str | None = None,
    commit: str = COMMIT,
    dirty: bool | None = False,
) -> AcceptanceAttempt:
    return AcceptanceAttempt(
        run_id=run_id,
        commit_sha=commit,
        git_dirty=dirty,
        status=status,
        abort_reason=abort,
        failed_gates=failed,
    )


def _blockers(*attempts: AcceptanceAttempt, rerun_of: str | None = None, dirty: bool = False):
    return rerun_blockers(attempts, commit=COMMIT, dirty=dirty, rerun_of=rerun_of)


def test_the_first_acceptance_run_of_a_commit_may_start():
    assert _blockers() == []


def test_a_second_run_of_the_same_commit_is_refused_without_a_documented_rerun():
    (blocker,) = _blockers(_attempt("run-1"))

    assert "already has an acceptance run (run-1)" in blocker
    assert "a commit of run outputs alone is not" in blocker


def test_a_run_that_failed_on_provider_availability_alone_may_be_rerun_once():
    assert _blockers(_attempt("run-1"), rerun_of="run-1") == []
    assert (
        _blockers(
            _attempt("run-1", status="aborted", abort="rate_limited", failed=()), rerun_of="run-1"
        )
        == []
    )


def test_the_one_rerun_is_used_once():
    (blocker,) = _blockers(_attempt("run-1"), _attempt("run-2"), rerun_of="run-1")

    assert "used its one permitted rerun" in blocker


@pytest.mark.parametrize(
    "attempt",
    [
        _attempt("run-1", failed=("citations_verified",)),
        _attempt("run-1", failed=("no_pipeline_errors", "unanswerable_refused")),
        _attempt("run-1", status="aborted", abort="internal_defect", failed=()),
        _attempt("run-1", status="aborted", abort="budget_guard", failed=()),
        _attempt("run-1", failed=None),
    ],
)
def test_a_run_that_failed_on_anything_but_availability_is_not_rerun(attempt: AcceptanceAttempt):
    (blocker,) = _blockers(attempt, rerun_of="run-1")

    assert "did not fail on provider availability alone" in blocker


def test_a_passing_run_is_not_rerun():
    (blocker,) = _blockers(_attempt("run-1", failed=()), rerun_of="run-1")

    assert "a passing run is not rerun" in blocker


def test_a_rerun_must_name_a_run_of_the_same_commit():
    other = _attempt("run-1", commit="b" * 40)

    (blocker,) = _blockers(other, rerun_of="run-1")

    assert "no clean acceptance run of release candidate commit" in blocker


def test_dirty_development_runs_neither_count_nor_may_claim_a_rerun():
    assert _blockers(_attempt("dev-1", dirty=True), _attempt("dev-2", dirty=True)) == []
    assert _blockers(dirty=True) == []
    (blocker,) = _blockers(dirty=True, rerun_of="run-1")
    assert "dirty or unknown" in blocker


def test_the_preflight_applies_the_rerun_rule_from_the_ledger():
    entry = LedgerEntry(
        date_utc="2026-10-01",
        run_id="run-1",
        tier="acceptance",
        status="complete",
        calls=1,
        input_tokens=0,
        output_tokens=0,
        estimated_neurons=0.0,
        usage_coverage=None,
        artifact=None,
        commit_sha=COMMIT,
        git_dirty=False,
        failed_gates=["citations_verified"],
    )

    def plan(rerun_of: str | None = None) -> Any:
        return preflight(
            tier=Tier.ACCEPTANCE,
            questions=2,
            provider="mistral",
            model="mistral-small",
            profile=None,
            history=UsageHistory(calls=(), questions=0, sources=(), newest=None, notes=()),
            ledger=LedgerReading(entries=(entry,), unreadable_lines=0),
            today=date(2026, 10, 4),
            policy=BudgetPolicy(),
            commit=COMMIT,
            dirty=False,
            rerun_of=rerun_of,
        )

    assert any("already has an acceptance run" in b for b in plan().blockers)
    assert any("did not fail on provider availability" in b for b in plan("run-1").blockers)
    assert not plan().can_start(yellow_confirmed=True)


def test_an_old_ledger_line_without_provenance_is_still_read(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        '{"date_utc": "2026-10-01", "run_id": "old", "tier": "acceptance", "status": '
        '"complete", "calls": 1, "input_tokens": 0, "output_tokens": 0, '
        '"estimated_neurons": 0.0, "usage_coverage": null, "artifact": null}\n',
        encoding="utf-8",
    )

    reading = Ledger(path).read()
    (attempt,) = acceptance_attempts(reading)

    assert reading.unreadable_lines == 0
    assert attempt.counts is False


# --- git provenance -------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)  # noqa: S603, S607


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    try:
        _git(tmp_path, "init", "-q")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    _git(
        tmp_path,
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "base",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_a_clean_checkout_is_clean_and_names_its_commit(repository: Path):
    state = _git_state()

    assert state.dirty is False
    assert state.revision is not None and len(state.revision) == 40


def test_a_run_output_does_not_make_the_tree_dirty_but_anything_else_does(repository: Path):
    (repository / "evaluation" / "results").mkdir(parents=True)
    (repository / "evaluation" / "results" / "run.json").write_text("{}", encoding="utf-8")
    assert _git_state().dirty is False

    (repository / "evaluation" / "questions.yaml").write_text("x", encoding="utf-8")
    assert _git_state().dirty is True


def test_a_tag_on_the_commit_is_recorded(repository: Path):
    assert _git_state().tag is None
    _git(repository, "tag", "v9.9.9")
    assert _git_state().tag == "v9.9.9"


def test_outside_a_checkout_nothing_is_guessed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    state = _git_state()

    assert (state.revision, state.dirty, state.tag, state.source_identity) == (
        None,
        None,
        None,
        None,
    )


# --- the source identity --------------------------------------------------------


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "user.name=t",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        message,
    )


@pytest.fixture
def project(repository: Path) -> Path:
    """A committed tree with one file in every included area."""
    for path, text in {
        "src/portfolio_rag/rag/prompt.py": "PROMPT = 1\n",
        "knowledge/profile.md": "# Profile\n",
        "evaluation/portfolio-questions.yaml": "version: 1\n",
        "evaluation/portfolio-questions.provider-smoke.yaml": "version: 1\n",
        "pyproject.toml": "[project]\n",
        "uv.lock": "lock\n",
        "docs/RELEASE_ACCEPTANCE.md": "# Doc\n",
        "tests/unit/test_x.py": "def test_x(): pass\n",
    }.items():
        (repository / path).parent.mkdir(parents=True, exist_ok=True)
        (repository / path).write_text(text, encoding="utf-8")
    _commit(repository, "candidate")
    return repository


def test_a_results_only_commit_keeps_the_source_identity(project: Path):
    before = _git_state()
    (project / "evaluation" / "results").mkdir()
    (project / "evaluation" / "results" / "run.json").write_text("{}", encoding="utf-8")
    (project / "evaluation" / "results" / "provider-ledger.jsonl").write_text("", encoding="utf-8")

    untracked = _git_state()
    _commit(project, "results")
    after = _git_state()

    assert untracked.source_identity == before.source_identity
    assert after.revision != before.revision
    assert after.source_identity == before.source_identity
    assert after.dirty is False


def test_documentation_and_tests_alone_are_not_a_new_candidate(project: Path):
    before = _git_state().source_identity
    (project / "docs" / "RELEASE_ACCEPTANCE.md").write_text("# Changed\n", encoding="utf-8")
    (project / "tests" / "unit" / "test_x.py").write_text("# changed\n", encoding="utf-8")

    assert _git_state().source_identity == before


@pytest.mark.parametrize(
    "path",
    [
        "src/portfolio_rag/rag/prompt.py",
        "evaluation/portfolio-questions.yaml",
        "evaluation/portfolio-questions.provider-smoke.yaml",
        "knowledge/profile.md",
        "pyproject.toml",
        "uv.lock",
    ],
)
def test_a_release_relevant_change_is_a_new_identity(project: Path, path: str):
    before = _git_state().source_identity
    (project / path).write_text("changed\n", encoding="utf-8")

    assert _git_state().source_identity != before


def test_a_new_release_relevant_file_is_a_new_identity_even_untracked(project: Path):
    before = _git_state().source_identity
    (project / "evaluation" / "portfolio-questions.release-acceptance.yaml").write_text(
        "version: 1\n", encoding="utf-8"
    )

    assert _git_state().source_identity != before


def test_the_source_identity_is_deterministic_and_content_based(project: Path):
    first = _git_state().source_identity
    (project / "knowledge" / "profile.md").write_text("changed\n", encoding="utf-8")
    (project / "knowledge" / "profile.md").write_text("# Profile\n", encoding="utf-8")

    assert _git_state().source_identity == first
    assert source_identity([("b", b"2"), ("a", b"1")]) == source_identity(
        [("a", b"1"), ("b", b"2")]
    )
    assert source_identity([("a", b"1")]) != source_identity([("a", b"2")])
    assert source_identity([("a", b"1")]) != source_identity([("b", b"1")])


def test_every_tracked_top_level_entry_is_classified():
    """A new top-level file or directory must be placed deliberately: inside
    the release candidate, or outside it with a reason."""
    outside = {
        ".claude",  # assistant tooling
        ".dockerignore",  # image build, not the acceptance run
        ".env.example",  # documentation of the environment
        ".github",  # CI
        ".gitignore",
        "AGENTS.md",
        "CLAUDE.md",
        "Dockerfile",  # image build, not the acceptance run
        "README.md",
        "docs",
        "edge",  # the edge worker; never on the acceptance path
        "tests",  # the local gate, not the run
    }
    listed = subprocess.run(
        ["git", "ls-files"],  # noqa: S607 - git from PATH, as the fixtures use it
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    top_level = {path.split("/")[0] for path in listed}

    assert top_level <= set(SOURCE_IDENTITY_INCLUDES) | outside, top_level - outside
    assert not set(SOURCE_IDENTITY_INCLUDES) & outside
    assert SOURCE_IDENTITY_EXCLUDES == ("evaluation/results",)


# --- the rerun rule, by source identity -----------------------------------------


def test_a_new_commit_with_the_same_source_is_the_same_candidate():
    earlier = replace(_attempt("run-1"), source_identity="5" * 64)

    (blocker,) = rerun_blockers(
        [earlier], commit="b" * 40, dirty=False, rerun_of=None, source_identity="5" * 64
    )

    assert "already has an acceptance run (run-1)" in blocker


def test_a_new_commit_with_a_new_source_is_a_new_candidate():
    earlier = replace(_attempt("run-1"), source_identity="5" * 64)

    assert (
        rerun_blockers(
            [earlier], commit="b" * 40, dirty=False, rerun_of=None, source_identity="6" * 64
        )
        == []
    )


def test_the_same_commit_is_the_same_candidate_for_an_entry_without_identity():
    earlier = _attempt("run-1")  # written before the identity was recorded

    assert rerun_blockers(
        [earlier], commit=COMMIT, dirty=False, rerun_of=None, source_identity="5" * 64
    )


# --- the version ----------------------------------------------------------------


def test_the_version_has_one_source():
    pyproject = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    init = (REPOSITORY_ROOT / "src/portfolio_rag/__init__.py").read_text(encoding="utf-8")

    assert "version" not in pyproject["project"]
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/portfolio_rag/__init__.py"
    # What the package build reads is what runtime reports.
    (static,) = re.findall(r'^__version__ = "([^"]+)"$', init, flags=re.MULTILINE)
    assert static == __version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def test_the_artifact_records_the_project_version():
    payload = _artifact()

    assert payload["run"]["project_version"] == __version__
    assert payload["release_acceptance"]["project_version"] == __version__


def test_a_tag_that_names_another_version_is_a_provenance_failure():
    payload = _artifact()
    payload["run"]["git_tag"] = "v1.0.1"

    assessed = assess(payload)

    assert assessed["verdict"] == "FAIL"
    assert any("does not match project_version" in reason for reason in assessed["reasons"])
    # A tag set after the run is also an identity its segment did not run
    # under; the run as the CLI would have recorded it with the right tag passes.
    payload["run"]["git_tag"] = f"v{__version__}"
    assert assess(payload)["verdict"] == "FAIL"
    assert assess(_executed(payload))["verdict"] == "PASS"


def test_a_release_without_a_tag_is_valid_and_the_tag_is_kept_apart():
    payload = _artifact()

    assert payload["run"]["git_tag"] is None
    assert payload["release_acceptance"]["verdict"] == "PASS"
    assert payload["run"]["project_version"] == __version__


@pytest.mark.parametrize("version", [None, "1.1", "v1.1.0", "latest"])
def test_the_validator_requires_a_well_formed_project_version(version: Any):
    payload = _artifact()
    payload["run"]["project_version"] = version

    assessed = assess(payload)

    assert assessed["verdict"] == "FAIL"
    assert any("project_version" in reason for reason in assessed["reasons"])


def test_the_rerun_guard_blocks_a_results_only_commit_of_a_real_tree(project: Path):
    """End to end over git: the identity of a results-only commit, fed to the guard."""
    first = _git_state()
    earlier = replace(
        _attempt("run-1", commit=first.revision or ""), source_identity=first.source_identity
    )
    (project / "evaluation" / "results").mkdir()
    (project / "evaluation" / "results" / "run-1.json").write_text("{}", encoding="utf-8")
    _commit(project, "results of run-1")
    second = _git_state()

    blocked = rerun_blockers(
        [earlier],
        commit=second.revision,
        dirty=second.dirty,
        rerun_of=None,
        source_identity=second.source_identity,
    )
    (project / "src" / "portfolio_rag" / "rag" / "prompt.py").write_text("PROMPT = 2\n", "utf-8")
    _commit(project, "a real change")
    third = _git_state()
    allowed = rerun_blockers(
        [earlier],
        commit=third.revision,
        dirty=third.dirty,
        rerun_of=None,
        source_identity=third.source_identity,
    )

    assert second.revision != first.revision
    assert len(blocked) == 1 and "already has an acceptance run (run-1)" in blocked[0]
    assert allowed == []
