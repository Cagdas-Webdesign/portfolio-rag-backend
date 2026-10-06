"""The final offline torture matrix before a live acceptance run.

Combinations, not isolated cases, through the production orchestration where
it can run offline: the CLI's acceptance path with canary, checkpoint, pause,
resume and validator (`test_cli_resume`'s offline provider), and the real
Workers AI adapter, pacer and answering service on a simulated network and
clock (`test_recovery_timeout_and_rate_limit`). No credential, no network, no
provider call. Each test names the matrix row it covers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.cli import EXIT_ACCEPTANCE_FAILED, EXIT_INTERNAL_ERROR, EXIT_INVALID, EXIT_OK
from portfolio_rag.evaluation.canary import CanaryStatus, run_canary
from portfolio_rag.evaluation.pacing import GenerationPacing, PacedLLMProvider
from portfolio_rag.infrastructure.embedding import DeterministicEmbeddingProvider
from portfolio_rag.infrastructure.llm import WorkersAIChatProvider
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ports.errors import (
    EmbeddingProviderError,
    LLMProviderError,
    ProviderFailureKind,
    VectorStoreError,
)
from portfolio_rag.ports.llm import (
    GenerationRequest,
    GenerationResponse,
    MessageRole,
    PromptMessage,
)
from portfolio_rag.rag.service import AnswerOutcome
from tests.integration.test_cli_eval import (  # noqa: F401 - fixtures, used by name
    _local_defaults,
    real_provider_names,
)
from tests.integration.test_cli_resume import (
    IDS,
    SUITE,
    Provider,
    _ledger,
    _read,
    _run,
    _validate,
    provider,  # noqa: F401 - fixture, used by name
)
from tests.integration.test_recovery_timeout_and_rate_limit import (
    EMPTY_AT_LIMIT,
    FAKE_TOKEN,
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

TEXT = {question.id: question.question for question in SUITE}


class Faults:
    """Retrieval-side faults at an exact question, injected at the ports."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.embedding_at: set[str] = set()
        self.search_at: set[str] = set()
        self.search_kind = ProviderFailureKind.UNREACHABLE
        self.current: str | None = None
        embed, query = DeterministicEmbeddingProvider.embed, InMemoryVectorStore.query
        faults = self

        async def embedding(self: Any, inputs: Any) -> Any:
            text = " ".join(item.text for item in inputs)
            matches = [qid for qid, question in TEXT.items() if question in text]
            faults.current = max(matches, key=lambda qid: len(TEXT[qid])) if matches else None
            if faults.current in faults.embedding_at:
                raise EmbeddingProviderError(
                    "scripted", retryable=True, kind=ProviderFailureKind.TIMEOUT
                )
            return await embed(self, inputs)

        async def search(self: Any, request: Any) -> Any:
            if faults.current in faults.search_at:
                raise VectorStoreError("scripted", kind=faults.search_kind, status_code=None)
            return await query(self, request)

        monkeypatch.setattr(DeterministicEmbeddingProvider, "embed", embedding)
        monkeypatch.setattr(InMemoryVectorStore, "query", search)


@pytest.fixture
def faults(monkeypatch: pytest.MonkeyPatch, provider: Provider) -> Faults:  # noqa: F811
    return Faults(monkeypatch)


# --- 1-3: the recovery path, slow, on the real adapter and service -------------------


def test_1_generation_length_then_slow_1500_recovery_succeeds(retrieval: Any):  # noqa: F811
    network = Network(generation=[TRUNCATED, Reply("<cite>", seconds=45.0)])
    answer = run(
        service_over(retrieval, network).answer(QUESTION["multi-marketing-automation-web"])
    )
    assert answer.outcome is AnswerOutcome.ANSWERED


def test_2_grounding_empty_at_length_then_slow_1500_recovery_succeeds(retrieval: Any):  # noqa: F811
    network = Network(
        generation=[Reply("<cite>")],
        check=[EMPTY_AT_LIMIT, Reply('{"verdict": "supported"}', seconds=45.0)],
    )
    answer = run(service_over(retrieval, network).answer(QUESTION["broad-project-scope"]))
    assert answer.outcome is AnswerOutcome.ANSWERED


def test_3_both_recoveries_in_one_question_fit_the_deadline(retrieval: Any):  # noqa: F811
    network = Network(
        generation=[TRUNCATED, Reply("<cite>", seconds=45.0)],
        check=[EMPTY_AT_LIMIT, Reply('{"verdict": "supported"}', seconds=45.0)],
    )
    answer = run(service_over(retrieval, network).answer(QUESTION["broad-project-scope"]))
    assert answer.outcome is AnswerOutcome.ANSWERED
    assert (answer.generation_attempts, answer.grounding_check_attempts) == (2, 2)


# --- 4: attempts made visible ----------------------------------------------------------


def test_4_a_recovery_that_times_out_once_then_answers_records_its_attempts(
    retrieval: Any,  # noqa: F811
):
    hanging = Reply("<cite>", seconds=3600.0)
    network = Network(generation=[TRUNCATED, hanging, Reply("<cite>", seconds=45.0)])

    answer = run(
        service_over(retrieval, network).answer(QUESTION["multi-marketing-automation-web"])
    )

    first, recovery = (c for c in answer.provider_calls if c.call_type.value == "generation")
    assert (first.http_attempts, first.transport_attempts) == (1, 1)
    # One logical recovery call, two requests on the wire: the timeout and the answer.
    assert (recovery.max_output_tokens, recovery.http_attempts) == (1500, 2)
    assert (recovery.transport_attempts, recovery.pacing_attempts) == (2, 1)
    assert len([r for r in network.requests if r["max_tokens"] == 1500]) == 2
    fields = recovery.fields()
    assert (fields["http_attempts"], fields["transport_attempts"]) == (2, 2)


def _scripted(*responses: httpx.Response) -> WorkersAIChatProvider:
    replies = list(responses)

    def handler(_: httpx.Request) -> httpx.Response:
        return replies.pop(0) if len(replies) > 1 else replies[0]

    return WorkersAIChatProvider(
        account_id="test-account",
        api_token=FAKE_TOKEN,
        max_attempts=1,
        client=httpx.AsyncClient(
            base_url="https://api.cloudflare.test/client/v4",
            transport=httpx.MockTransport(handler),
        ),
    )


def _ok(text: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "success": True,
            "errors": [],
            "result": {
                "choices": [{"index": 0, "message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            },
        },
    )


def _limited(code: int | None, retry_after: str = "2") -> httpx.Response:
    errors = [{"code": code, "message": "SECRET-PROVIDER-TEXT"}] if code is not None else []
    return httpx.Response(429, json={"errors": errors}, headers={"Retry-After": retry_after})


def _request() -> GenerationRequest:
    return GenerationRequest(messages=(PromptMessage(role=MessageRole.USER, content="q"),))


def test_4_a_pacer_round_is_visible_and_the_wait_it_honoured():
    waits: list[float] = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    paced = PacedLLMProvider(
        _scripted(_limited(3040, "7"), _ok('{"x": 1}')),
        GenerationPacing(),
        sleeper=sleep,
    )
    response = run(paced.generate(_request()))

    assert (response.pacing_attempts, response.transport_attempts, response.http_attempts) == (
        2,
        1,
        2,
    )
    assert response.retry_after_seconds == 7.0
    assert waits == [7.0], "the retry policy itself is unchanged"


def test_4_a_paced_failure_reports_every_request_and_round():
    async def sleep(_: float) -> None:
        return None

    paced = PacedLLMProvider(
        _scripted(_limited(3036)), GenerationPacing(max_attempts=3), sleeper=sleep
    )
    with pytest.raises(LLMProviderError) as caught:
        run(paced.generate(_request()))

    error = caught.value
    assert (error.attempts, error.pacing_attempts, error.transport_attempts) == (3, 3, 1)
    assert (error.provider_error_code, error.rate_limit_kind) == (
        3036,
        "daily_free_allocation_exhausted",
    )


# --- 5-7: a 429 pauses, whatever its code, and names only documented codes -------------


@pytest.mark.parametrize(
    ("code", "kind"),
    [(3036, "daily_free_allocation_exhausted"), (3040, "capacity_exceeded"), (9999, None)],
    ids=["5-daily-allocation", "6-capacity", "7-undocumented"],
)
def test_5_to_7_a_429_pauses_and_records_its_code(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: Provider,  # noqa: F811
    code: int,
    kind: str | None,
):
    original = Provider.generate

    async def limited(self: Provider, llm: Any, request: GenerationRequest) -> Any:
        qid = self.question_of(" ".join(m.content for m in request.messages))
        if qid == IDS[5]:
            error = LLMProviderError(
                "scripted",
                retryable=True,
                kind=ProviderFailureKind.RATE_LIMITED,
                status_code=429,
                provider_error_code=code,
                rate_limit_kind=kind,
            )
            raise error
        return await original(self, llm, request)

    monkeypatch.setattr(Provider, "generate", limited)
    target = tmp_path / "run.json"

    exit_code, _, err = _run(capsys, "--output", str(target))

    assert exit_code == EXIT_ACCEPTANCE_FAILED
    assert f"Run paused after {IDS[5]}: rate_limited" in err
    (event,) = _read(target)["run"]["rate_limit_events"]
    assert (event["question_id"], event["provider_error_code"], event["rate_limit_kind"]) == (
        IDS[5],
        code,
        kind,
    )
    assert "SECRET-PROVIDER-TEXT" not in target.read_text(encoding="utf-8")


# --- 8, 9, 15, 16, 17: retrieval faults pause and resume --------------------------------


@pytest.mark.parametrize("stage", ["embedding", "vector_store"], ids=["8-embedding", "9-vectorize"])
def test_8_9_a_transient_retrieval_failure_pauses_and_resumes_without_rerunning_anything(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,  # noqa: F811
    faults: Faults,
    stage: str,
):
    checkpoint = tmp_path / "run.checkpoint.json"
    target = faults.embedding_at if stage == "embedding" else faults.search_at
    target.add(IDS[8])

    code, _, err = _run(capsys, "--output", str(tmp_path / "run.json"))

    assert code == EXIT_ACCEPTANCE_FAILED
    assert f"Run paused after {IDS[8]}: retrieval_unavailable" in err
    saved = _read(checkpoint)
    assert saved["progress"]["completed_question_ids"] == IDS[:8]
    assert saved["segments"][0]["interrupted_question_ids"] == [IDS[8]]
    paused = _read(tmp_path / "run.json")
    assert IDS[8] not in [q["id"] for q in paused["questions"]], "17: not finalized"
    answered = dict(provider.answered)

    target.clear()
    final = tmp_path / "run.segment-2.json"
    code, _, _ = _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))

    assert code == EXIT_OK
    data = _read(final)
    ids = [q["id"] for q in data["questions"]]
    assert ids == IDS and len(set(ids)) == len(ids), "16: every question exactly once"
    for qid in IDS[:8]:
        assert provider.answered[qid] == answered[qid] == 1, "15: completed never re-run"
    assert provider.answered[IDS[8]] == 2
    assert data["run"]["execution_segments"][0]["stop_reason"] == "retrieval_unavailable"
    assert data["release_acceptance"]["verdict"] == "PASS"
    assert _validate(capsys, final)[0] == EXIT_OK


def test_a_lasting_retrieval_failure_stays_a_hard_failure(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,  # noqa: F811
    faults: Faults,
):
    faults.search_at.add(IDS[3])
    faults.search_kind = ProviderFailureKind.MALFORMED_RESPONSE
    target = tmp_path / "run.json"

    code, _, _ = _run(capsys, "--output", str(target))

    data = _read(target)
    assert code == EXIT_ACCEPTANCE_FAILED
    assert data["run"]["complete"] is True, "not paused: asking again would change nothing"
    errored = next(q for q in data["questions"] if q["id"] == IDS[3])
    assert errored["outcome"] == "error"
    assert errored["retrieval_error"] == {
        "retrieval_stage": "vector_store",
        "transient": False,
        "detail": "malformed_response",
        "status_code": None,
        "retry_after_seconds": None,
    }
    assert "gate no_pipeline_errors FAIL" in data["release_acceptance"]["reasons"]


# --- 10, 11: process interruptions ---------------------------------------------------


def test_10_11_a_question_interrupted_twice_by_the_process_blocks_publication(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,  # noqa: F811
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.crash_at = IDS[4]
    assert _run(capsys, "--output", str(tmp_path / "run.json"))[0] == EXIT_INTERNAL_ERROR

    # Interrupted again on the same question: the second strike.
    code, _, _ = _run(
        capsys, "--resume-from", str(checkpoint), "--output", str(tmp_path / "s2.json")
    )
    assert code == EXIT_INTERNAL_ERROR

    provider.crash_at = None
    final = tmp_path / "s3.json"
    code, _, err = _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))

    assert f"WARNING: {IDS[4]} has now been interrupted by the process more than 1" in err
    data = _read(final)
    assert code == EXIT_ACCEPTANCE_FAILED
    assert data["run"]["complete"] is True
    stops = [s["stop_reason"] for s in data["run"]["execution_segments"]]
    assert stops == ["process_interrupted", "process_interrupted", None], "10: kept on record"
    release = data["release_acceptance"]
    assert release["conditions"]["process_interruptions_within_limit"] is False
    assert f"interrupted by the process more than once: {IDS[4]}" in release["reasons"]
    code, report = _validate(capsys, final)
    assert code == EXIT_INVALID and "Rejected for publication" in report
    assert _validate(capsys, final, "--development")[0] == EXIT_OK, "consistent, not publishable"


def test_a_validator_refuses_an_interruption_erased_from_the_record(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,  # noqa: F811
):
    checkpoint = tmp_path / "run.checkpoint.json"
    provider.crash_at = IDS[4]
    _run(capsys, "--output", str(tmp_path / "run.json"))
    provider.crash_at = None
    final = tmp_path / "s2.json"
    _run(capsys, "--resume-from", str(checkpoint), "--output", str(final))

    data = _read(final)
    data["run"]["execution_segments"][0]["interrupted_question_ids"] = []
    final.write_text(json.dumps(data), encoding="utf-8")
    code, report = _validate(capsys, final, "--development")

    assert code == EXIT_INVALID
    assert "interrupted without naming the question it was asking" in report


def test_a_429_pause_is_not_counted_against_the_interruption_limit(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,  # noqa: F811
):
    checkpoint = tmp_path / "s1.checkpoint.json"
    for segment in range(1, 4):
        provider.limit_at = {IDS[2]} if segment < 3 else set()
        output = tmp_path / f"s{segment}.json"
        argv = ["--output", str(output)]
        if segment > 1:
            argv = ["--resume-from", str(checkpoint), *argv]
        _run(capsys, *argv)

    data = _read(tmp_path / "s3.json")
    assert [s["stop_reason"] for s in data["run"]["execution_segments"]][:2] == [
        "rate_limited",
        "rate_limited",
    ]
    assert data["release_acceptance"]["verdict"] == "PASS"


# --- 12-14: the canary judges the provider, not a formatting habit -------------------


class _Says:
    def __init__(self, text: str | None, finish_reason: str = "stop") -> None:
        self.text, self.finish_reason = text, finish_reason

    @property
    def model(self) -> str:
        return "canary-test"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.text is None:
            raise LLMProviderError(
                "no text",
                kind=ProviderFailureKind.MALFORMED_RESPONSE,
                finish_reason=self.finish_reason,
            )
        return GenerationResponse(
            text=self.text, model="canary-test", finish_reason=self.finish_reason
        )


@pytest.mark.parametrize(
    ("text", "finish", "status", "category"),
    [
        ('{"ok": true}', "stop", CanaryStatus.PASS, None),
        ('```json\n{"ok": true}\n```', "stop", CanaryStatus.PASS, None),
        ('```\n{"ok":true}\n```', "stop", CanaryStatus.PASS, None),
        ('{"ok": tru', "stop", CanaryStatus.MALFORMED_RESPONSE, "reply_not_ok_json"),
        ('{"ok": false}', "stop", CanaryStatus.MALFORMED_RESPONSE, "reply_not_ok_json"),
        ('Sure! {"ok": true}', "stop", CanaryStatus.MALFORMED_RESPONSE, "reply_not_ok_json"),
        ("", "stop", CanaryStatus.MALFORMED_RESPONSE, "reply_not_ok_json"),
        ('{"ok": tr', "length", CanaryStatus.MALFORMED_RESPONSE, "output_truncated"),
        (None, "length", CanaryStatus.MALFORMED_RESPONSE, "output_truncated"),
        (None, "stop", CanaryStatus.MALFORMED_RESPONSE, "malformed_response"),
    ],
    ids=[
        "12-plain",
        "13-fenced-json",
        "13-fenced-bare",
        "14-malformed",
        "14-wrong-object",
        "14-prose",
        "14-empty",
        "14-truncated",
        "14-no-text-at-limit",
        "14-no-text",
    ],
)
def test_12_to_14_the_canary_accepts_what_production_accepts_and_nothing_more(
    text: str | None, finish: str, status: CanaryStatus, category: str | None
):
    result = run(run_canary(_Says(text, finish)))

    assert result.status is status
    assert result.failure_category == category


# --- 17, 18: no incomplete PASS, no leak ---------------------------------------------


def test_17_18_a_paused_run_never_passes_and_nothing_secret_is_written(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    provider: Provider,  # noqa: F811
):
    provider.limit_at = {IDS[20]}
    target = tmp_path / "run.json"

    _run(capsys, "--output", str(target))

    data = _read(target)
    assert data["run"]["complete"] is False
    assert data["release_acceptance"]["verdict"] == "FAIL"
    assert "the run is not complete" in data["release_acceptance"]["reasons"]
    # A partial export is rejected outright: its question count names the
    # suite, and it records fewer — as every stopped run always has been.
    code, report = _validate(capsys, target)
    assert code == EXIT_INVALID and "Rejected" in report
    assert "verdict        FAIL" in report
    for written in (target, tmp_path / "run.checkpoint.json", tmp_path / "ledger.jsonl"):
        text = written.read_text(encoding="utf-8")
        assert FAKE_TOKEN not in text and "test-value-not-a-real-credential" not in text
        assert "Bearer" not in text and "Authorization" not in text
    assert all(entry["tier"] in {"canary", "acceptance"} for entry in _ledger(tmp_path))


# --- F-06: attempt counts on the adapter alone ----------------------------------------


def _adapter(*responses: Any, attempts: int = 3) -> WorkersAIChatProvider:
    replies = list(responses)

    async def no_wait(_: float) -> None:
        return None

    def handler(request: httpx.Request) -> httpx.Response:
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply  # type: ignore[no-any-return]

    return WorkersAIChatProvider(
        account_id="test-account",
        api_token=FAKE_TOKEN,
        max_attempts=attempts,
        sleeper=no_wait,
        client=httpx.AsyncClient(
            base_url="https://api.cloudflare.test/client/v4",
            transport=httpx.MockTransport(handler),
        ),
    )


def test_one_request_is_one_attempt():
    response = run(_adapter(_ok('{"x": 1}')).generate(_request()))
    assert (response.transport_attempts, response.pacing_attempts, response.http_attempts) == (
        1,
        1,
        1,
    )


def test_two_failed_transports_and_a_success_are_three_attempts():
    response = run(
        _adapter(httpx.Response(503), httpx.Response(503), _ok('{"x": 1}')).generate(_request())
    )
    assert (response.transport_attempts, response.http_attempts) == (3, 3)


@pytest.mark.parametrize(
    ("reply", "status"),
    [(_limited(3040), 429), (httpx.ReadTimeout("slow"), None)],
    ids=["429", "timeout"],
)
def test_a_call_that_fails_reports_every_attempt_it_made(reply: Any, status: int | None):
    with pytest.raises(LLMProviderError) as caught:
        run(_adapter(reply).generate(_request()))

    error = caught.value
    assert (error.attempts, error.transport_attempts, error.pacing_attempts) == (3, 3, 1)
    assert error.status_code == status
