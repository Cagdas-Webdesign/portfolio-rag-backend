"""The comparison harness, offline: payloads, parsing, fail-closed scoring, limits.

No test here reaches a network. Groq is answered by an `httpx.MockTransport`,
Cloudflare by a scripted adapter.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationResponse, TokenUsage
from portfolio_rag.rag.verification import GROUNDING_CHECK_INSTRUCTIONS
from tests.grounding_compare import harness
from tests.grounding_compare.cases import CASES, NOT_MODEL_CASES
from tests.grounding_compare.harness import (
    CallResult,
    CloudflareJudge,
    GroqJudge,
    build_request,
    run_cases,
    score,
)

FAKE_KEY = "gsk_test_not_a_real_key"
CORPUS = harness._corpus_index(Path("knowledge"))
REQUESTS = {case.id: build_request(case, CORPUS) for case in CASES}


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def groq_reply(content: Any, *, status: int = 200, **extra: Any) -> httpx.Response:
    body = {
        "choices": [
            {
                "message": {"content": content, "reasoning": "SECRET THOUGHTS"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 600,
            "completion_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 110},
        },
        **extra,
    }
    return httpx.Response(status, json=body)


def groq_judge(*replies: httpx.Response, **kwargs: Any) -> tuple[GroqJudge, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return replies[min(len(seen) - 1, len(replies) - 1)]

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GroqJudge(api_key=FAKE_KEY, client=client, **kwargs), seen


# --- the cases --------------------------------------------------------------------


def test_every_case_resolves_to_the_production_grounding_request():
    for case in CASES:
        request = REQUESTS[case.id]
        assert request.messages[0].content == GROUNDING_CHECK_INSTRUCTIONS
        assert case.question in request.messages[1].content
        assert case.answer in request.messages[1].content
        assert request.max_output_tokens == harness.MAX_OUTPUT_TOKENS


def test_corpus_cases_carry_the_real_passage_text():
    case = next(c for c in CASES if c.id == "corpus-degree-false")
    assert "nicht als abgeschlossenes Hochschulstudium" in REQUESTS[case.id].messages[1].content


def test_cited_labels_follow_the_passage_order():
    case = next(c for c in CASES if c.id == "corpus-typescript-moved")
    evidence = REQUESTS[case.id].messages[1].content
    assert evidence.index("[SOURCE S1]") < evidence.index("[SOURCE S2]")


def test_the_case_set_is_small_labelled_and_covers_the_required_kinds():
    assert 10 <= len(CASES) <= 25
    assert len({c.id for c in CASES}) == len(CASES)
    assert {c.expected for c in CASES} == {"supported", "not_supported"}
    categories = " ".join(c.category for c in CASES)
    for kind in (
        "belegt",
        "teilweise",
        "erfunden",
        "technische",
        "Ausbildung",
        "widersprüchlich",
        "Injection",
    ):
        assert kind in categories, kind
    assert set(NOT_MODEL_CASES) == {"fehlende Quellen", "ungültige Zitate", "ungültiges JSON"}


# --- the Groq request -------------------------------------------------------------


def test_the_groq_payload_is_the_same_request_in_openai_shape():
    judge, seen = groq_judge(groq_reply('{"verdict": "supported"}'))
    run(judge.judge(REQUESTS[CASES[0].id]))

    sent = seen[0]
    body = json.loads(sent.content)
    assert str(sent.url) == harness.GROQ_URL
    assert sent.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"][0] == {"role": "system", "content": GROUNDING_CHECK_INSTRUCTIONS}
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 800
    assert body["temperature"] == 0.2
    assert body["include_reasoning"] is False
    assert "reasoning_effort" not in body


def test_a_reasoning_effort_is_sent_only_when_asked_for():
    judge, seen = groq_judge(groq_reply('{"verdict": "supported"}'), reasoning_effort="low")
    run(judge.judge(REQUESTS[CASES[0].id]))

    assert json.loads(seen[0].content)["reasoning_effort"] == "low"


def test_the_api_key_never_appears_in_a_result():
    judge, _ = groq_judge(groq_reply('{"verdict": "supported"}'))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert FAKE_KEY not in json.dumps(
        result.__dict__ if hasattr(result, "__dict__") else str(result)
    )
    assert FAKE_KEY not in repr(judge)


# --- reading the Groq reply -------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "outcome", "rule"),
    [
        ('{"verdict": "supported"}', "supported", None),
        ('{"verdict": "not_supported"}', "not_supported", None),
        ('```json\n{"verdict": "supported"}\n```', "supported", None),
        ("supported", "unusable", "reply_not_json"),
        ('{"verdict": true}', "unusable", "verdict_not_recognized"),
        ('{"answer": "supported"}', "unusable", "verdict_missing"),
        ("", "unusable", "reply_empty"),
        (None, "unusable", "reply_empty"),
    ],
)
def test_the_reply_is_read_by_the_production_parser(content: Any, outcome: str, rule: str | None):
    judge, _ = groq_judge(groq_reply(content))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.rule) == (outcome, rule)


def test_usage_and_reasoning_tokens_are_recorded_and_reasoning_text_is_not():
    judge, _ = groq_judge(groq_reply('{"verdict": "supported"}'))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (600, 120, 110)
    assert "SECRET" not in repr(result)


def test_a_body_that_is_not_a_completion_is_unusable():
    judge, _ = groq_judge(httpx.Response(200, json={"unexpected": True}))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.rule) == ("unusable", "malformed_body")


# --- failures: never retried, never a confirmation -----------------------------------


def test_a_429_is_reported_once_and_never_retried():
    judge, seen = groq_judge(httpx.Response(429, headers={"retry-after": "12"}, json={}))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.retry_after) == ("rate_limited", "12")
    assert len(seen) == 1


def test_an_http_error_is_an_error_and_its_body_is_not_kept():
    judge, _ = groq_judge(httpx.Response(400, json={"error": {"message": "UPSTREAM TEXT"}}))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.status_code) == ("error", 400)
    assert "UPSTREAM" not in repr(result)


def test_a_timeout_is_an_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    judge = GroqJudge(
        api_key=FAKE_KEY, client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.rule) == ("error", "timeout")


class _ScriptedCloudflare:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    async def generate(self, request: Any) -> GenerationResponse:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return GenerationResponse(
            text=self.outcome,
            model="m",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=700, output_tokens=300),
        )

    async def aclose(self) -> None:
        return None


def test_the_cloudflare_judge_reads_the_same_way():
    result = run(
        CloudflareJudge(_ScriptedCloudflare('{"verdict": "not_supported"}')).judge(
            REQUESTS[CASES[0].id]
        )
    )

    assert (result.outcome, result.input_tokens, result.output_tokens) == (
        "not_supported",
        700,
        300,
    )


def test_a_cloudflare_reply_cut_off_at_the_limit_is_unusable_not_a_verdict():
    error = LLMProviderError(
        "no text",
        kind=ProviderFailureKind.MALFORMED_RESPONSE,
        finish_reason="length",
        usage=TokenUsage(input_tokens=700, output_tokens=800),
    )
    result = run(CloudflareJudge(_ScriptedCloudflare(error)).judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.finish_reason, result.output_tokens) == (
        "unusable",
        "length",
        800,
    )


# --- the run: hard limits ---------------------------------------------------------


class _Fixed:
    def __init__(self, name: str, outcomes: list[str]) -> None:
        self.name, self.model, self.outcomes, self.calls = name, "m", outcomes, 0

    async def judge(self, request: Any) -> CallResult:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        return CallResult(outcome)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        return None


async def _no_sleep(_: float) -> None:
    return None


def test_the_run_never_exceeds_max_calls():
    a, b = _Fixed("a", ["supported"]), _Fixed("b", ["supported"])
    results = run(run_cases(CASES, REQUESTS, [a, b], max_calls=5, delay_seconds=1, sleep=_no_sleep))

    assert a.calls + b.calls == 5
    assert (
        sum(r is not None for r in results["a"].values())
        + sum(r is not None for r in results["b"].values())
        == 5
    )


def test_a_rate_limited_judge_is_not_asked_again():
    limited = _Fixed("groq", ["supported", "rate_limited", "supported"])
    results = run(run_cases(CASES, REQUESTS, [limited], max_calls=100, sleep=_no_sleep))

    assert limited.calls == 2
    assert sum(r is None for r in results["groq"].values()) == len(CASES) - 2


# --- scoring: fail-closed and the exclusion criterion ---------------------------------


def _results(outcome_for: Any) -> dict[str, CallResult | None]:
    return {case.id: CallResult(outcome_for(case)) for case in CASES}


def test_a_perfect_judge_scores_perfectly():
    summary = score(CASES, _results(lambda c: c.expected), model="m")

    assert summary["falsely_accepted"] == summary["falsely_rejected"] == 0
    assert summary["safe_for_production"] is True


def test_one_false_accept_makes_the_judge_unsafe_and_is_named():
    summary = score(CASES, _results(lambda c: "supported"), model="m")

    assert summary["falsely_accepted"] > 0
    assert "corpus-degree-false" in summary["false_accept_case_ids"]
    assert summary["safe_for_production"] is False


@pytest.mark.parametrize("outcome", ["unusable", "error", "rate_limited"])
def test_anything_short_of_supported_counts_as_a_refusal(outcome: str):
    summary = score(CASES, _results(lambda c: outcome), model="m")

    assert summary["correctly_accepted"] == summary["falsely_accepted"] == 0
    assert summary["correctly_rejected"] == sum(
        c.scored and c.expected == "not_supported" for c in CASES
    )


def test_unscored_cases_count_towards_nothing():
    summary = score(CASES, _results(lambda c: "supported"), model="m")

    assert "synthetic-contradicting-sources" not in summary["false_accept_case_ids"]
    assert summary["scored_cases"] == sum(c.scored for c in CASES)


def test_an_incomplete_run_is_never_safe():
    partial = {
        case.id: (CallResult(case.expected) if i < 3 else None) for i, case in enumerate(CASES)
    }

    assert score(CASES, partial, model="m")["safe_for_production"] is False


# --- the command line: dry by default ------------------------------------------------


def test_without_live_nothing_is_sent(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a dry run must not build a client")

    monkeypatch.setattr(httpx, "AsyncClient", no_network)
    assert harness.main(["--providers", "groq,cloudflare"]) == 0
    assert "Dry run: no request was sent" in capsys.readouterr().out


def test_live_needs_a_call_limit_and_a_new_output_path(tmp_path: Path):
    with pytest.raises(SystemExit):
        harness.main(["--live", "--providers", "groq"])
    existing = tmp_path / "x.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit):
        harness.main(
            ["--live", "--providers", "groq", "--max-calls", "3", "--output", str(existing)]
        )


# --- diagnosing an HTTP error: identifiers only -----------------------------------


def test_an_http_error_keeps_error_type_and_code_and_nothing_else():
    body = {
        "error": {
            "message": "UPSTREAM PROSE with gsk_live_secret",
            "type": "invalid_request_error",
            "code": "json_validate_failed",
            "failed_generation": "GENERATED TEXT",
        }
    }
    judge, _ = groq_judge(httpx.Response(400, json=body))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.status_code, result.error_type, result.error_code) == (
        400,
        "invalid_request_error",
        "json_validate_failed",
    )
    kept = repr(result)
    assert "UPSTREAM" not in kept and "GENERATED" not in kept and "gsk_" not in kept


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"type": "has spaces in it", "code": "x" * 65}},
        {"error": {"type": 400, "code": None}},
        {"error": "a bare string"},
        {"detail": "no error object"},
        ["not", "an", "object"],
    ],
)
def test_anything_that_is_not_an_identifier_is_dropped(body: Any):
    judge, _ = groq_judge(httpx.Response(400, json=body))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.error_type, result.error_code) == (None, None)


def test_a_non_json_error_body_yields_no_fields():
    judge, _ = groq_judge(httpx.Response(502, text="<html>bad gateway</html>"))
    result = run(judge.judge(REQUESTS[CASES[0].id]))

    assert (result.outcome, result.status_code, result.error_type, result.error_code) == (
        "error",
        502,
        None,
        None,
    )


# --- selecting single cases ---------------------------------------------------------


def test_a_case_id_restricts_the_plan_to_that_case(capsys: pytest.CaptureFixture[str]):
    assert harness.main(["--case-id", "synthetic-contradicting-sources"]) == 0

    out = capsys.readouterr().out
    assert "cases: 1 (0 scored)" in out
    assert "planned calls: 1" in out


def test_an_unknown_case_id_is_refused():
    with pytest.raises(SystemExit):
        harness.main(["--case-id", "no-such-case"])


def test_a_live_run_of_one_case_sends_one_request_and_exports_only_that_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(400, json={"error": {"type": "invalid_request_error", "code": "x1"}})

    real_client = httpx.AsyncClient

    def mocked_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", mocked_client)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    output = tmp_path / "one.json"

    code = harness.main(
        [
            "--live",
            "--providers",
            "groq",
            "--max-calls",
            "1",
            "--case-id",
            "synthetic-contradicting-sources",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert len(sent) == 1
    export = json.loads(output.read_text())
    assert export["selected_case_ids"] == ["synthetic-contradicting-sources"]
    assert [c["id"] for c in export["cases"]] == ["synthetic-contradicting-sources"]
    result = export["judges"]["groq"]["results"]["synthetic-contradicting-sources"]
    assert (result["status_code"], result["error_type"], result["error_code"]) == (
        400,
        "invalid_request_error",
        "x1",
    )
    assert FAKE_KEY not in output.read_text()
