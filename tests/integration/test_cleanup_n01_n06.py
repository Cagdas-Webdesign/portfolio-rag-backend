"""N-01 to N-06 of the adversarial pre-release audit, and their combination.

Offline throughout: the CLI's acceptance path over the offline provider of
`test_cli_resume`, and the real Workers AI adapter and answering service over
the simulated network of `test_recovery_timeout_and_rate_limit`.
"""

# ruff: noqa: F811 - pytest fixtures are parameters named like their imports

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.cli import EXIT_ACCEPTANCE_FAILED, EXIT_INVALID, EXIT_OK, main
from portfolio_rag.evaluation import run_e2e_evaluation
from portfolio_rag.evaluation.acceptance import validate_artifact
from portfolio_rag.evaluation.checkpoint import (
    AcceptanceCheckpoint,
    CheckpointLockedError,
    acquire_lock,
    lock_path,
    stale_checkpoint_problem,
)
from portfolio_rag.evaluation.e2e import recovery_evidence
from portfolio_rag.rag.errors import GenerationUnavailableError
from portfolio_rag.rag.generation import cut_off_json
from portfolio_rag.rag.service import AnswerOutcome
from tests.e2e_stack import dataset_of
from tests.integration.test_cli_eval import (  # noqa: F401 - fixtures, used by name
    _local_defaults,
    real_provider_names,
)
from tests.integration.test_cli_resume import (
    DATASET,
    IDS,
    Provider,
    _ledger,
    _read,
    _run,
    _validate,
    provider,  # noqa: F401 - fixture, used by name
)
from tests.integration.test_recovery_timeout_and_rate_limit import (
    DATASET as FULL,
)
from tests.integration.test_recovery_timeout_and_rate_limit import (
    EMPTY_AT_LIMIT,
    QUESTION,
    TRUNCATED,
    Reply,
    retrieval,  # noqa: F401 - fixture, used by name
)
from tests.integration.test_recovery_timeout_and_rate_limit import (
    Provider as Network,
)
from tests.integration.test_recovery_timeout_and_rate_limit import (
    _service as service_over,
)
from tests.support import run

REPOSITORY = Path(__file__).resolve().parents[2]


def _question(qid: str) -> Any:
    return next(question for question in FULL.questions if question.id == qid)


# --- N-01: tiered runs name their dataset ----------------------------------------------


@pytest.mark.parametrize("tier", ["acceptance", "smoke"])
def test_n01_a_tiered_run_without_a_dataset_is_refused_before_anything(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
    tier: str,
):
    code = main(["eval", "run", "--e2e", "--tier", tier, "--output", str(tmp_path / "x.json")])

    err = capsys.readouterr().err
    assert code == EXIT_INVALID
    assert f"--tier {tier} needs an explicit --dataset" in err
    assert "evaluation/portfolio-questions.yaml" in err
    assert not provider.answered and not (tmp_path / "x.json").exists()


def test_n01_every_documented_tier_command_names_the_release_dataset():
    commands = []
    for name in ("CLAUDE.md", "docs/RELEASE_ACCEPTANCE.md", "evaluation/README.md"):
        for line in (REPOSITORY / name).read_text(encoding="utf-8").splitlines():
            if "eval run" in line and "--tier" in line and not line.lstrip().startswith("|"):
                commands.append((name, line))
    assert commands
    for name, line in commands:
        assert "--dataset evaluation/portfolio-questions.yaml" in line, (name, line)


def test_n01_a_smoke_with_its_dataset_runs(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    target = tmp_path / "smoke.json"
    code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(DATASET),
            "--e2e",
            "--tier",
            "smoke",
            "--question-id",
            "multi-marketing-automation-web",
            "--question-id",
            "broad-project-scope",
            "--generation-delay-seconds",
            "2",
            "--min-similarity",
            "-1",
            "--output",
            str(target),
        ]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert _read(target)["run"]["generation"]["delay_seconds"] == 2.0
    assert "RECOVERY_LIVE_VALIDATED = false" in out


def test_n01_a_resume_under_another_dataset_path_is_refused(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    provider.limit_at = {IDS[3]}
    _run(capsys, "--output", str(tmp_path / "run.json"))
    provider.limit_at = set()

    code = main(
        [
            "eval",
            "run",
            "--dataset",
            str((REPOSITORY / DATASET).resolve()),
            "--e2e",
            "--tier",
            "acceptance",
            "--min-similarity",
            "-1",
            "--resume-from",
            str(tmp_path / "run.checkpoint.json"),
            "--output",
            str(tmp_path / "x.json"),
        ]
    )

    assert code == EXIT_INVALID
    assert "dataset.path: checkpoint" in capsys.readouterr().err


# --- N-02: only the latest checkpoint continues ----------------------------------------


def test_n02_an_older_copy_of_the_checkpoint_is_refused_and_the_latest_continues(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    checkpoint = tmp_path / "run.checkpoint.json"
    stale = tmp_path / "copy-after-segment-1.checkpoint.json"
    provider.limit_at = {IDS[5]}
    _run(capsys, "--output", str(tmp_path / "run.json"))
    shutil.copy(checkpoint, stale)

    provider.limit_at = {IDS[12]}
    _run(capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "s2.json"))
    asked = sum(provider.answered.values())

    code, _, err = _run(capsys, "--resume-from", str(stale), "--output", str(tmp_path / "x.json"))
    assert code == EXIT_INVALID
    assert "STALE_CHECKPOINT" in err and "segment 2 in the ledger" in err
    assert sum(provider.answered.values()) == asked, "nothing asked again"

    provider.limit_at = set()
    code, _, _ = _run(
        capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "s3.json")
    )
    assert code == EXIT_OK


def test_n02_a_copy_taken_mid_segment_is_stale_once_that_segment_is_recorded(tmp_path: Path):
    checkpoint = AcceptanceCheckpoint(
        path=tmp_path / "c.json",
        logical_run_id="L",
        started_at="t0",
        identity={"question_ids": ["q1", "q2"]},
    )
    checkpoint.begin_segment(run_id="L", started_at="t0")
    entry = type(
        "Entry", (), {"tier": "acceptance", "logical_run_id": "L", "segment": 1, "run_id": "L"}
    )()

    assert stale_checkpoint_problem(checkpoint, []) is None, "a crash has no ledger line yet"
    problem = stale_checkpoint_problem(checkpoint, [entry])
    assert problem is not None and "STALE_CHECKPOINT" in problem


# --- N-03: the smoke says whether the larger recovery was really exercised ----------------


def _report(retrieval: Any, network: Network, qid: str) -> Any:
    return run(run_e2e_evaluation(dataset_of(_question(qid)), service_over(retrieval, network)))


@pytest.mark.parametrize(
    ("generation", "check", "expected"),
    [
        ([Reply("<cite>")], [Reply('{"verdict": "supported"}')], (False, 0, 0, None, False)),
        (
            [TRUNCATED, Reply("<cite>", seconds=45.0)],
            [Reply('{"verdict": "supported"}')],
            (True, 1, 0, 1500, True),
        ),
        (
            [Reply("<cite>")],
            [EMPTY_AT_LIMIT, Reply('{"verdict": "supported"}', seconds=45.0)],
            (True, 0, 1, 1500, True),
        ),
        (
            [TRUNCATED, Reply("<cite>", seconds=45.0)],
            [EMPTY_AT_LIMIT, Reply('{"verdict": "supported"}', seconds=45.0)],
            (True, 1, 1, 1500, True),
        ),
    ],
    ids=["no-recovery", "generation", "grounding", "both"],
)
def test_n03_recovery_evidence(
    retrieval: Any,
    generation: list[Reply],
    check: list[Reply],
    expected: tuple[Any, ...],
):
    report = _report(retrieval, Network(generation=generation, check=check), "broad-project-scope")
    evidence = recovery_evidence(report, recovery_cap=1500)

    assert report.passed
    assert (
        evidence["recovery_triggered"],
        evidence["generation_recovery_count"],
        evidence["grounding_recovery_count"],
        evidence["recovery_max_output_tokens_observed"],
        evidence["recovery_live_validated"],
    ) == expected


def test_n03_a_recovery_at_the_ordinary_cap_is_not_live_validation(retrieval: Any):
    """A malformed reply that did not stop at the limit is asked again as it was."""
    malformed = Reply('{"verdict": "supported"}')
    network = Network(generation=[Reply("<cite>")], check=[Reply(None, "stop", 120), malformed])

    evidence = recovery_evidence(
        _report(retrieval, network, "broad-project-scope"), recovery_cap=1500
    )

    assert evidence["recovery_triggered"] is True
    assert evidence["recovery_max_output_tokens_observed"] == 800
    assert evidence["recovery_live_validated"] is False


# --- N-04: a reply cut off without a finish reason still gets the larger cap ------------


def test_n04_clear_truncation_without_a_finish_reason_gets_the_1500_recovery(
    retrieval: Any,
):
    cut = Reply('{"answer": "Marketing automation, web and', None, 800, 13.0)
    network = Network(generation=[cut, Reply("<cite>", seconds=45.0)])

    answer = run(
        service_over(retrieval, network).answer(QUESTION["multi-marketing-automation-web"])
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert [r["max_tokens"] for r in network.requests if not r["check"]] == [800, 1500]
    first = answer.provider_calls[0]
    assert (first.finish_reason, first.failure and first.failure.category.value) == (
        None,
        "output_truncated",
    )


@pytest.mark.parametrize(
    ("text", "finish"),
    [
        ('{"answer": }', None),  # complete, invalid
        ('{"foo": "bar"}', None),  # the wrong shape
        ("I cannot answer that.", None),  # prose
        ('{"answer": "cut', "stop"),  # says it finished: believed
    ],
    ids=["complete-invalid", "wrong-schema", "prose", "finished"],
)
def test_n04_nothing_weaker_than_a_cut_off_object_gets_the_larger_cap(
    retrieval: Any,
    text: str,
    finish: str | None,
):
    """Whatever these end as — an unusable reply, or a declined answer for the
    wrong shape — none of them is asked again at the larger cap."""
    network = Network(generation=[Reply(text, finish, 50)])
    service = service_over(retrieval, network)

    try:
        answer = run(service.answer(QUESTION["multi-marketing-automation-web"]))
    except GenerationUnavailableError as exc:
        assert exc.failure is not None
        assert exc.failure.category.value == "unparseable_output"
    else:
        assert answer.outcome is not AnswerOutcome.ANSWERED
    generations = [r["max_tokens"] for r in network.requests if not r["check"]]
    assert generations == [800], "no recovery, no larger cap"


def test_n04_the_detector():
    assert cut_off_json('{"answer": "x", "sources": ["S1"')
    assert cut_off_json('```json\n{"answer": "x')
    assert not cut_off_json('{"answer": "x"}')
    assert not cut_off_json('{"answer": } trailing')
    assert not cut_off_json('Sure: {"answer": "x')
    assert not cut_off_json("")


# --- N-05: one process per checkpoint ---------------------------------------------------


def test_n05_a_second_claim_is_refused_until_the_first_is_released(tmp_path: Path):
    checkpoint = tmp_path / "c.json"
    first = acquire_lock(checkpoint, logical_run_id="L", at="t0")

    with pytest.raises(CheckpointLockedError, match="another resume may be running"):
        acquire_lock(checkpoint, logical_run_id="L", at="t1")
    content = json.loads(lock_path(checkpoint).read_text(encoding="utf-8"))
    assert set(content) == {"logical_run_id", "pid", "host", "created_at", "checkpoint"}
    assert re.fullmatch(r"[0-9a-f]{16}", content["host"]), "a hash, not the machine's name"

    first.release()
    acquire_lock(checkpoint, logical_run_id="L", at="t2").release()
    assert not lock_path(checkpoint).exists()


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_n05_a_lock_of_a_dead_process_is_replaced_and_recorded(tmp_path: Path):
    checkpoint = tmp_path / "c.json"
    holder = acquire_lock(checkpoint, logical_run_id="L", at="t0")
    content = json.loads(holder.path.read_text(encoding="utf-8"))
    holder.path.write_text(json.dumps({**content, "pid": _dead_pid()}), encoding="utf-8")

    lock = acquire_lock(checkpoint, logical_run_id="L", at="t1")

    assert lock.audit is not None and "stale_lock_recovered" in lock.audit
    assert json.loads(lock.path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_n05_a_concurrent_resume_is_refused_and_break_lock_is_recorded(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.limit_at = {IDS[3]}
    code, _, _ = _run(capsys, "--output", str(tmp_path / "run.json"))
    assert code == EXIT_ACCEPTANCE_FAILED
    assert not lock_path(checkpoint).exists(), "released when the run ended"
    provider.limit_at = set()

    # Another process is resuming: its lock is held by a live process.
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        held = acquire_lock(checkpoint, logical_run_id="L", at="t0")
        content = json.loads(held.path.read_text(encoding="utf-8"))
        held.path.write_text(json.dumps({**content, "pid": other.pid}), encoding="utf-8")
        asked = sum(provider.answered.values())

        code, _, err = _run(
            capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "x.json")
        )
        assert code == EXIT_INVALID
        assert "is locked by process" in err and "Nothing was requested" in err
        assert sum(provider.answered.values()) == asked

        # Preflight reads only, and is not blocked.
        code, out, _ = _run(
            capsys,
            "--resume-from",
            str(checkpoint),
            "--output",
            str(tmp_path / "x.json"),
            "--preflight-only",
        )
        assert code == EXIT_OK and "Preflight only" in out

        final = tmp_path / "s2.json"
        code, _, _ = _run(
            capsys, "--resume-from", str(checkpoint), "--output", str(final), "--break-lock"
        )
    finally:
        other.kill()
        other.wait()

    assert code == EXIT_OK
    segment = _read(final)["run"]["execution_segments"][1]
    assert segment["lock"]["lock_broken_by_operator"]["pid"] == other.pid
    assert not lock_path(checkpoint).exists()


# --- N-06: paused is valid but incomplete -----------------------------------------------


@pytest.fixture
def paused(capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider) -> Any:
    provider.limit_at = {IDS[9]}
    _run(capsys, "--output", str(tmp_path / "run.json"))
    return _read(tmp_path / "run.json")


def test_n06_a_paused_prefix_is_valid_but_incomplete_and_never_passes(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, paused: Any
):
    check = validate_artifact(paused)

    assert check.problems == ()
    assert check.state == "VALID_INCOMPLETE"
    assert check.verdict == "FAIL" and "the run is not complete" in check.reasons
    assert not check.accepted(publication=True)
    code, report = _validate(capsys, tmp_path / "run.json", "--development")
    assert code == EXIT_OK and "state          VALID_INCOMPLETE" in report


def _drop(payload: Any, index: int) -> Any:
    payload["questions"].pop(index)
    return payload


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda p: _drop(p, 3),
        lambda p: p["questions"].insert(4, dict(p["questions"][4])) or p,
        lambda p: p["questions"].reverse() or p,
    ],
    ids=["missing-middle", "duplicate", "reordered"],
)
def test_n06_a_paused_artifact_with_a_hole_a_duplicate_or_a_swap_is_invalid(
    paused: Any, corrupt: Any
):
    check = validate_artifact(corrupt(paused))

    assert check.state == "INVALID"
    assert check.problems
    assert not check.accepted(publication=False)


def test_n06_a_complete_run_is_valid_complete(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    _run(capsys, "--output", str(tmp_path / "run.json"))
    check = validate_artifact(_read(tmp_path / "run.json"))

    assert (check.state, check.verdict) == ("VALID_COMPLETE", "PASS")


# --- Phase 8: the combination ---------------------------------------------------------


def test_phase8_the_whole_operational_path_in_one_logical_run(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    checkpoint = tmp_path / "run.checkpoint.json"
    copy = tmp_path / "copy.checkpoint.json"

    # 1-3: explicit dataset, a pause at question 10, a valid checkpoint.
    provider.limit_at = {IDS[9]}
    code, _, err = _run(capsys, "--output", str(tmp_path / "run.json"))
    assert code == EXIT_ACCEPTANCE_FAILED and f"--dataset {DATASET}" in err
    assert validate_artifact(_read(tmp_path / "run.json")).state == "VALID_INCOMPLETE"  # 10
    shutil.copy(checkpoint, copy)

    # 4: resume, paused again further on.
    provider.limit_at = {IDS[17]}
    _run(capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "s2.json"))

    # 5: the old copy is refused.
    code, _, err = _run(capsys, "--resume-from", str(copy), "--output", str(tmp_path / "x.json"))
    assert code == EXIT_INVALID and "STALE_CHECKPOINT" in err

    # 6: a concurrent resume of the latest is refused.
    held = acquire_lock(checkpoint, logical_run_id="L", at="t")
    try:
        code, _, err = _run(
            capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "x.json")
        )
        assert code == EXIT_INVALID and "is locked by process" in err
    finally:
        held.release()

    # 7: the latest resume completes the run, and it is a release acceptance.
    provider.limit_at = set()
    final = tmp_path / "s3.json"
    code, out, _ = _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))
    data = _read(final)
    assert code == EXIT_OK
    assert [q["id"] for q in data["questions"]] == IDS
    assert data["metrics"]["recovery"]["recovery_live_validated"] is False  # 11: observed, no gate
    assert "RECOVERY_LIVE_VALIDATED = false" in out
    assert validate_artifact(data).state == "VALID_COMPLETE"
    assert _validate(capsys, final)[0] == EXIT_OK
    assert [e["segment"] for e in _ledger(tmp_path) if e["tier"] == "acceptance"] == [1, 2, 3]
