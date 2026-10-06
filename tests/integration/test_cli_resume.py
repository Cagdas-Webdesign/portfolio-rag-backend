"""A resumable acceptance run, driven as a shell would: canary, pause, resume.

The offline stack stands in for the providers (see `real_provider_names`), with
one addition: a provider that says which question each request belongs to, so
that a test can stop the run with a 429 at an exact question and count, per
question, what was asked — and prove that a resume asks nothing twice.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag import cli
from portfolio_rag.cli import EXIT_ACCEPTANCE_FAILED, EXIT_INVALID, EXIT_OK, GitState, main
from portfolio_rag.evaluation import load_dataset, select_suite
from portfolio_rag.evaluation.canary import CanaryResult, CanaryStatus
from portfolio_rag.evaluation.dataset import EvaluationSuite
from portfolio_rag.infrastructure.llm.deterministic import MODEL_NAME, DeterministicLLMProvider
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse
from portfolio_rag.rag.service import GroundedAnswerService
from tests.integration.test_cli_eval import (  # noqa: F401 - fixtures, used by name
    _local_defaults,
    real_provider_names,
)

DATASET = Path("evaluation/portfolio-questions.yaml")
COMMIT = "c0ffee" + "0" * 34
SOURCE = "5" * 64
SUITE = select_suite(DATASET, load_dataset(DATASET), EvaluationSuite.RELEASE_ACCEPTANCE).questions
IDS = [question.id for question in SUITE]

_DECLINE = json.dumps(
    {
        "answer": "The passages do not cover this.",
        "sources": [],
        "support": "none",
        "verdict": "not_supported",
    }
)


class Provider:
    """Which question each request is for, a 429 on demand, a decline for the
    questions the corpus cannot answer — and every call counted."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.answered: Counter[str] = Counter()
        self.limit_at: set[str] = set()
        self.crash_at: str | None = None

    @staticmethod
    def question_of(text: str) -> str:
        matches = [question for question in SUITE if question.question in text]
        return max(matches, key=lambda question: len(question.question)).id

    async def generate(self, llm: Any, request: GenerationRequest) -> GenerationResponse:
        qid = self.question_of(" ".join(message.content for message in request.messages))
        self.calls[qid] += 1
        if qid in self.limit_at:
            raise LLMProviderError(
                "scripted", retryable=True, kind=ProviderFailureKind.RATE_LIMITED, status_code=429
            )
        if next(q for q in SUITE if q.id == qid).must_retrieve_nothing:
            return GenerationResponse(text=_DECLINE, model=MODEL_NAME, finish_reason="stop")
        return await ORIGINAL_GENERATE(llm, request)


ORIGINAL_GENERATE = DeterministicLLMProvider.generate
ORIGINAL_ANSWER = GroundedAnswerService.answer


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch, real_provider_names: None) -> Provider:  # noqa: F811
    scripted = Provider()

    async def generate(self: DeterministicLLMProvider, request: GenerationRequest) -> Any:
        return await scripted.generate(self, request)

    async def answer(self: GroundedAnswerService, question: str, **kwargs: Any) -> Any:
        qid = Provider.question_of(question)
        scripted.answered[qid] += 1
        if qid == scripted.crash_at:
            raise KeyboardInterrupt  # the terminal was closed mid-question
        return await ORIGINAL_ANSWER(self, question, **kwargs)

    monkeypatch.setattr(DeterministicLLMProvider, "generate", generate)
    monkeypatch.setattr(GroundedAnswerService, "answer", answer)
    return scripted


def _tree(monkeypatch: pytest.MonkeyPatch, **changes: Any) -> None:
    state = GitState(COMMIT, False, None, SOURCE)._replace(**changes)
    monkeypatch.setattr(cli, "_git_state", lambda: state)


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(DATASET),
            "--e2e",
            "--tier",
            "acceptance",
            # Every question reaches the provider: there is something to pause.
            "--min-similarity",
            "-1",
            *argv,
        ]
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _read(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _ledger(tmp_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _validate(capsys: pytest.CaptureFixture[str], path: Path, *argv: str) -> tuple[int, str]:
    code = main(["eval", "validate-acceptance", str(path), *argv])
    return code, capsys.readouterr().out


# --- the canary ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("result", "shown"),
    [
        (
            CanaryResult(
                status=CanaryStatus.RATE_LIMITED, status_code=429, retry_after_seconds=60.0
            ),
            "rate_limited, HTTP 429, retry after 60s",
        ),
        (CanaryResult(status=CanaryStatus.AUTH_FAILURE, status_code=401), "auth_failure"),
        (CanaryResult(status=CanaryStatus.AUTH_FAILURE, status_code=403), "auth_failure, HTTP 403"),
        (CanaryResult(status=CanaryStatus.PROVIDER_FAILURE, status_code=502), "provider_failure"),
        (
            CanaryResult(
                status=CanaryStatus.MALFORMED_RESPONSE, failure_category="reply_not_ok_json"
            ),
            "malformed_response",
        ),
    ],
    ids=["429", "401", "403", "5xx", "malformed"],
)
def test_a_failed_canary_keeps_the_run_from_starting(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    result: CanaryResult,
    shown: str,
):
    async def canary(llm: Any) -> CanaryResult:
        return result

    monkeypatch.setattr(cli, "run_canary", canary)
    target = tmp_path / "run.json"

    code, out, err = _run(capsys, "--output", str(target))

    assert code == EXIT_INVALID
    assert f"Not started: the provider canary did not pass ({shown}" in err
    assert re.search(rf"canary_status\s+{result.status.value}", out)
    assert not provider.answered, "no question was asked"
    assert not target.exists() and not (tmp_path / "run.checkpoint.json").exists()
    (entry,) = _ledger(tmp_path)
    # The canary's spend counts today; it is no acceptance attempt.
    assert (entry["tier"], entry["canary_status"]) == ("canary", result.status.value)


def test_a_passing_canary_lets_the_run_start_and_is_recorded(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    target = tmp_path / "run.json"

    code, _, _ = _run(capsys, "--output", str(target))

    assert code == EXIT_OK
    data = _read(target)
    assert data["operations"]["canary"]["canary_status"] == "pass"
    assert data["run"]["execution_segments"][0]["canary"]["canary_status"] == "pass"
    assert [entry["tier"] for entry in _ledger(tmp_path)] == ["canary", "acceptance"]


def test_no_canary_is_recorded_as_skipped(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    target = tmp_path / "run.json"

    code, _, _ = _run(capsys, "--output", str(target), "--no-canary")

    assert code == EXIT_OK
    assert _read(target)["operations"]["canary"] == {"canary_status": "skipped"}
    assert [entry["tier"] for entry in _ledger(tmp_path)] == ["acceptance"]


# --- pause and resume ------------------------------------------------------------


def test_a_429_pauses_the_run_and_a_resume_completes_it_without_asking_anything_twice(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    first, final = tmp_path / "run.json", tmp_path / "run.segment-2.json"
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.limit_at = {IDS[10]}

    code, _, err = _run(capsys, "--output", str(first))

    # Ten questions done, the eleventh refused: paused, not failed question by question.
    assert code == EXIT_ACCEPTANCE_FAILED
    assert f"Run paused after {IDS[10]}: rate_limited" in err
    resume = (
        f"uv run portfolio-rag eval run --dataset {DATASET} --e2e --tier acceptance "
        f"--resume-from {checkpoint} --output {final}"
    )
    assert resume in err
    saved = _read(checkpoint)
    assert saved["status"] == "paused" and saved["pause_reason"] == "rate_limited"
    assert saved["progress"]["completed_question_ids"] == IDS[:10]
    assert saved["progress"]["next_question_id"] == IDS[10]
    paused = _read(first)
    assert paused["run"]["complete"] is False
    assert [q["id"] for q in paused["questions"]] == IDS[:10]
    assert paused["release_acceptance"]["verdict"] == "FAIL"
    calls_before = Counter(provider.calls)
    answered_before = Counter(provider.answered)

    provider.limit_at = set()
    code, out, _ = _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))

    assert code == EXIT_OK, out
    assert f"Resumed: 10 of 24 questions completed earlier; continuing at {IDS[10]}" in out
    for qid in IDS[:10]:
        # No embedding, no search, no generation, no check for a completed question.
        assert provider.answered[qid] == answered_before[qid] == 1
        assert provider.calls[qid] == calls_before[qid]
    assert all(provider.answered[qid] == 1 for qid in IDS[11:])
    assert provider.answered[IDS[10]] == 2, "the interrupted question is asked again"

    data = _read(final)
    run_ = data["run"]
    assert run_["complete"] is True
    assert [q["id"] for q in data["questions"]] == IDS
    assert run_["logical_run_id"] == run_["run_id"] == saved["logical_run_id"]
    assert run_["resume_count"] == 1
    assert [s["stop_reason"] for s in run_["execution_segments"]] == ["rate_limited", None]
    assert run_["execution_segments"][0]["interrupted_question_ids"] == [IDS[10]]
    assert [e["question_id"] for e in run_["rate_limit_events"]] == [IDS[10]]
    assert data["release_acceptance"]["verdict"] == "PASS"
    assert _read(checkpoint)["status"] == "complete"

    code, report = _validate(capsys, final)
    assert code == EXIT_OK, report
    assert "Accepted: a passing release acceptance that may be published." in report

    acceptance = [entry for entry in _ledger(tmp_path) if entry["tier"] == "acceptance"]
    assert [(e["segment"], e["status"]) for e in acceptance] == [(1, "aborted"), (2, "complete")]
    assert {e["logical_run_id"] for e in acceptance} == {run_["logical_run_id"]}
    # Spend is counted once: segments add up to the logical run's observed calls.
    assert sum(e["calls"] for e in acceptance) == data["operations"]["observed"]["calls"]


def test_a_run_can_pause_more_than_once(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.limit_at = {IDS[4]}
    _run(capsys, "--output", str(tmp_path / "run.json"))
    provider.limit_at = {IDS[15]}
    code, _, err = _run(
        capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "run.segment-2.json")
    )
    assert code == EXIT_ACCEPTANCE_FAILED
    assert "run.segment-3.json" in err

    provider.limit_at = set()
    final = tmp_path / "run.segment-3.json"
    code, _, _ = _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))

    data = _read(final)
    assert code == EXIT_OK
    assert data["run"]["resume_count"] == 2
    completed = [s["completed_question_ids"] for s in data["run"]["execution_segments"]]
    assert completed == [IDS[:4], IDS[4:15], IDS[15:]]
    assert _validate(capsys, final)[0] == EXIT_OK


def test_a_crash_loses_at_most_the_question_being_asked(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.crash_at = IDS[6]

    code, _, _ = _run(capsys, "--output", str(tmp_path / "run.json"))

    assert code == cli.EXIT_INTERNAL_ERROR
    crashed = _read(checkpoint)
    assert crashed["status"] == "running"
    assert crashed["progress"]["completed_question_ids"] == IDS[:6]

    provider.crash_at = None
    final = tmp_path / "run.segment-2.json"
    code, out, _ = _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))

    assert code == EXIT_OK
    assert "ended without closing (process interrupted)" in out
    data = _read(final)
    assert [q["id"] for q in data["questions"]] == IDS
    assert data["run"]["execution_segments"][0]["stop_reason"] == "process_interrupted"
    assert _validate(capsys, final)[0] == EXIT_OK
    crash_line = next(
        e for e in _ledger(tmp_path) if e.get("abort_reason") == "process_interrupted"
    )
    assert crash_line["segment"] == 1 and crash_line["calls"] > 0


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"revision": "d" * 40}, "git_revision"),
        ({"tag": "v9.9.9"}, "git_tag"),
        ({"source_identity": "9" * 64}, "release_source_identity"),
    ],
    ids=["commit", "tag", "source-identity"],
)
def test_a_resume_under_another_identity_is_refused_before_anything_is_asked(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,
    change: dict[str, Any],
    field: str,
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.limit_at = {IDS[3]}
    _run(capsys, "--output", str(tmp_path / "run.json"))
    answered = sum(provider.answered.values())
    provider.limit_at = set()
    _tree(monkeypatch, **change)

    code, _, err = _run(
        capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "x.json")
    )

    assert code == EXIT_INVALID
    assert "Resume refused: this is not the same logical run." in err
    assert f"  - {field}: checkpoint" in err
    assert sum(provider.answered.values()) == answered, "nothing was asked"
    assert not (tmp_path / "x.json").exists()


def test_a_resume_with_another_retrieval_policy_is_refused(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.limit_at = {IDS[3]}
    _run(capsys, "--output", str(tmp_path / "run.json"))

    code, _, err = _run(
        capsys,
        "--resume-from",
        str(checkpoint),
        "--output",
        str(tmp_path / "x.json"),
        "--top-k",
        "3",
    )

    assert code == EXIT_INVALID
    assert "retrieval_policy.top_k: checkpoint" in err


def test_a_complete_run_is_not_resumed_and_an_existing_checkpoint_is_not_overwritten(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, provider: Provider
):
    target = tmp_path / "run.json"
    _run(capsys, "--output", str(target))

    code, _, err = _run(capsys, "--output", str(target))
    assert code == EXIT_INVALID
    assert "already exists" in err

    code, _, err = _run(
        capsys,
        "--resume-from",
        str(tmp_path / "run.checkpoint.json"),
        "--output",
        str(tmp_path / "again.json"),
    )
    assert code == EXIT_INVALID
    assert "is not continued: it is complete" in err


def test_resume_flags_belong_to_an_acceptance_run(capsys: pytest.CaptureFixture[str]):
    code = main(
        ["eval", "run", "--dataset", str(DATASET), "--resume-from", "x.json", "--retrieval-only"]
    )
    assert code == EXIT_INVALID
    assert "--resume-from belongs to an --e2e --tier acceptance run" in capsys.readouterr().err


def test_the_acceptance_tier_paces_its_provider_calls_by_default():
    parser = cli._build_parser()
    acceptance = parser.parse_args(
        ["eval", "run", "--e2e", "--tier", "acceptance", "--output", "x.json"]
    )
    smoke = parser.parse_args(["eval", "run", "--e2e", "--tier", "smoke", "--output", "x.json"])
    chosen = parser.parse_args(
        [
            "eval",
            "run",
            "--e2e",
            "--tier",
            "acceptance",
            "--output",
            "x.json",
            "--generation-delay-seconds",
            "5",
        ]
    )

    assert cli._generation_delay(acceptance) == cli.ACCEPTANCE_PROVIDER_CALL_DELAY_SECONDS == 2.0
    assert cli._generation_delay(smoke) == 0.0
    assert cli._generation_delay(chosen) == 5.0
