"""N-05, closed: one process per *logical run*, whichever checkpoint copy it starts from.

The checkpoint lock guards one file. Two copies of a checkpoint at two paths are
two files — each could be claimed, and both copies continued at once, each
asking the same remaining questions and each producing an artifact of the same
logical run. The run lock is named after the logical run and lives beside the
ledger, so every copy meets at one file, held as an operating-system lock by
the process that continues the run.

Offline: the CLI's acceptance path over the offline provider of
`test_cli_resume`. The "other process" is a real second Python process holding
the run's lock, as a resume running in another terminal would.
"""

# ruff: noqa: F811 - pytest fixtures are parameters named like their imports

from __future__ import annotations

import json
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.cli import EXIT_INVALID, EXIT_OK
from portfolio_rag.evaluation.checkpoint import (
    AcceptanceCheckpoint,
    CheckpointError,
    CheckpointLockedError,
    acquire_run_lock,
    run_lock_path,
)
from portfolio_rag.rag.service import GroundedAnswerService
from tests.integration.test_cli_eval import (  # noqa: F401 - fixtures, used by name
    _local_defaults,
    real_provider_names,
)
from tests.integration.test_cli_resume import (
    IDS,
    Provider,
    _ledger,
    _run,
    provider,  # noqa: F401 - fixture, used by name
)

REPOSITORY = Path(__file__).resolve().parents[2]


@contextmanager
def held_elsewhere(ledger: Path, logical_run_id: str) -> Iterator[subprocess.Popen[str]]:
    """A second, live process that holds the run's lock until it is ended."""
    code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from portfolio_rag.evaluation.checkpoint import acquire_run_lock\n"
        "lock = acquire_run_lock(Path(sys.argv[1]), sys.argv[2], at='t-other',"
        " checkpoint=Path('elsewhere/run.checkpoint.json'))\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    other = subprocess.Popen(  # noqa: S603 - fixed arguments, this interpreter
        [sys.executable, "-c", code, str(ledger), logical_run_id],
        cwd=REPOSITORY,
        env={"PYTHONPATH": str(REPOSITORY / "src")},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert other.stdout is not None
        assert other.stdout.readline().strip() == "held"
        yield other
    finally:
        other.kill()
        other.wait(timeout=10)


def _paused_run(capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider) -> Path:
    """A logical run paused by a 429 after five questions; its checkpoint."""
    provider.limit_at = {IDS[5]}
    _run(capsys, "--output", str(tmp_path / "run.json"))
    provider.limit_at = set()
    return tmp_path / "run.checkpoint.json"


def _logical(checkpoint: Path) -> str:
    return AcceptanceCheckpoint.load(checkpoint).logical_run_id


def test_two_copies_at_two_paths_cannot_be_continued_at_once(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    original = _paused_run(capsys, tmp_path, provider)
    copy = tmp_path / "elsewhere" / "copy.checkpoint.json"
    copy.parent.mkdir()
    shutil.copy(original, copy)
    logical = _logical(original)
    assert _logical(copy) == logical, "the same logical run, at another path"
    ledger_before = _ledger(tmp_path)
    copy_before = copy.read_bytes()
    asked = sum(provider.answered.values())

    # Another process is continuing the run — from the original, say.
    with held_elsewhere(tmp_path / "ledger.jsonl", logical):
        code, _, err = _run(
            capsys, "--resume-from", str(copy), "--output", str(tmp_path / "b.json")
        )
        assert code == EXIT_INVALID
        assert f"logical run {logical} is being continued by process" in err
        assert "--break-lock does not apply" in err
        # Its own path is free, the run is not: --break-lock changes nothing.
        code, _, err = _run(
            capsys,
            "--resume-from",
            str(original),
            "--break-lock",
            "--output",
            str(tmp_path / "a.json"),
        )
        assert code == EXIT_INVALID
        assert "is being continued by process" in err

    # Refused before anything: no question, no canary, no ledger line, no write.
    assert sum(provider.answered.values()) == asked
    assert _ledger(tmp_path) == ledger_before
    assert copy.read_bytes() == copy_before
    assert not (tmp_path / "b.json").exists()

    # The other process is gone: the run can be continued — from either copy,
    # once. Continuing one makes the other an older state of the run.
    code, _, _ = _run(capsys, "--resume-from", str(copy), "--output", str(tmp_path / "b.json"))
    assert code == EXIT_OK
    code, _, err = _run(
        capsys, "--resume-from", str(original), "--output", str(tmp_path / "a.json")
    )
    assert code == EXIT_INVALID
    assert "STALE_CHECKPOINT" in err


def test_a_new_run_holds_its_lock_from_before_its_first_request(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,
):
    seen: list[str] = []
    answer = GroundedAnswerService.answer

    async def probing(self: GroundedAnswerService, question: str, **kwargs: Any) -> Any:
        if not seen:
            logical = _logical(tmp_path / "run.checkpoint.json")
            # A copy of this checkpoint, continued now from anywhere, meets the lock.
            with pytest.raises(CheckpointLockedError):
                acquire_run_lock(tmp_path / "ledger.jsonl", logical, at="t-copy")
            seen.append(logical)
        return await answer(self, question, **kwargs)

    monkeypatch.setattr(GroundedAnswerService, "answer", probing)
    code, _, _ = _run(capsys, "--output", str(tmp_path / "run.json"))

    assert code == EXIT_OK
    (logical,) = seen
    # Released when the run ended.
    acquire_run_lock(tmp_path / "ledger.jsonl", logical, at="t-after").release()


def test_different_logical_runs_do_not_block_each_other(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    first = acquire_run_lock(ledger, "a" * 32, at="t0")
    second = acquire_run_lock(ledger, "b" * 32, at="t0")
    with pytest.raises(CheckpointLockedError):
        acquire_run_lock(ledger, "a" * 32, at="t1")
    first.release()
    second.release()
    acquire_run_lock(ledger, "a" * 32, at="t2").release()


def test_a_killed_holder_leaves_no_lock_and_an_active_one_is_never_taken(tmp_path: Path):
    ledger = tmp_path / "ledger.jsonl"
    logical = "c" * 32
    with held_elsewhere(ledger, logical) as other:
        with pytest.raises(CheckpointLockedError):
            acquire_run_lock(ledger, logical, at="t1")
        other.send_signal(signal.SIGKILL)  # no cleanup of any kind
        other.wait(timeout=10)
        # The kernel released it with the process; the file is still there.
        assert run_lock_path(ledger, logical).exists()
        lock = acquire_run_lock(ledger, logical, at="t2")
    content = json.loads(run_lock_path(ledger, logical).read_text(encoding="utf-8"))
    assert set(content) == {"logical_run_id", "pid", "host", "created_at", "checkpoint"}
    assert content["logical_run_id"] == logical and len(content["host"]) == 16
    lock.release()


def test_only_a_minted_logical_run_id_becomes_a_lock_file(tmp_path: Path):
    for unsafe in ("../../etc/passwd", "L", "A" * 32, ""):
        with pytest.raises(CheckpointError):
            acquire_run_lock(tmp_path / "ledger.jsonl", unsafe, at="t0")
    assert not (tmp_path / "locks").exists() or not any((tmp_path / "locks").iterdir())
