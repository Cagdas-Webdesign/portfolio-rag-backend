"""F-01/F-02/F-03 of the reliability audit, held offline.

The two pipeline errors of the v1.1.1 release acceptance ended a reply at the
800-token cap. The recovery that followed in v1.1.2 asks for 1500 tokens — and
a reasoning model at the slowest throughput ever recorded (38.7 tokens/s)
needs about 39 s for that, past the 30 s every request used to get, and a
question's whole recovery path is longer than the 60 s production deadline.

Time is simulated, not waited for: the in-process transport reads the read
timeout the adapter set on each request and, when a reply is scripted to take
longer, raises the timeout a real client would. Everything else is real —
the knowledge corpus, retrieval (offline embeddings), context, the answering
service, the failure policy, the Workers AI adapter and its parsing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.composition import build_query_components, evaluation_question_deadline
from portfolio_rag.core.config import LLMProviderName, Settings, get_settings
from portfolio_rag.evaluation import load_dataset
from portfolio_rag.evaluation.pacing import DEFAULT_MAX_ATTEMPTS as PACER_ATTEMPTS
from portfolio_rag.infrastructure.llm import WorkersAIChatProvider
from portfolio_rag.infrastructure.llm.workers_ai import (
    DEFAULT_MAX_ATTEMPTS as TRANSPORT_ATTEMPTS,
)
from portfolio_rag.infrastructure.llm.workers_ai import (
    MIN_OUTPUT_TOKENS_PER_SECOND,
    request_timeout_seconds,
)
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationRequest, MessageRole, PromptMessage
from portfolio_rag.rag.errors import GenerationUnavailableError, provider_failure
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.service import (
    MAX_GENERATION_ATTEMPTS,
    MAX_GROUNDING_CHECK_ATTEMPTS,
    AnswerOutcome,
    GroundedAnswerService,
)
from portfolio_rag.rag.verification import GROUNDING_CHECK_INSTRUCTIONS
from tests.support import run

FAKE_TOKEN = "test-token-not-a-real-credential"  # noqa: S105 - a fixture value
DATASET = load_dataset(Path("evaluation/portfolio-questions.yaml"))
QUESTION = {question.id: question.question for question in DATASET.questions}

#: A provider message that must never travel further than the adapter.
PROVIDER_TEXT = "SECRET-PROVIDER-TEXT account 0123abcd has exhausted its allocation"

_LABEL = re.compile(r"\[SOURCE (S\d+)\]")


@dataclass
class Reply:
    """One scripted provider reply, and how long it takes to arrive."""

    content: str | None
    finish_reason: str | None = "stop"
    """``None`` is a provider that says nothing about why it stopped."""
    completion_tokens: int = 200
    seconds: float = 2.0


@dataclass
class Provider:
    """Workers AI on an in-process transport, with simulated time.

    A reply scripted to take longer than the read timeout the adapter set on
    its request is a ``ReadTimeout`` — exactly what httpx raises live.
    """

    generation: list[Reply]
    check: list[Reply] = field(default_factory=lambda: [Reply('{"verdict": "supported"}')])
    requests: list[dict[str, Any]] = field(default_factory=list)

    def handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        is_check = payload["messages"][0]["content"] == GROUNDING_CHECK_INSTRUCTIONS
        script = self.check if is_check else self.generation
        seen = sum(1 for r in self.requests if r["check"] == is_check)
        reply = script[min(seen, len(script) - 1)]
        read_timeout = request.extensions["timeout"]["read"]
        self.requests.append(
            {
                "check": is_check,
                "max_tokens": payload.get("max_tokens"),
                "read_timeout": read_timeout,
                "seconds": reply.seconds,
            }
        )
        if reply.seconds > read_timeout:
            raise httpx.ReadTimeout("simulated: the reply took longer", request=request)
        content = reply.content
        if content == "<cite>":
            label = _LABEL.search(payload["messages"][-1]["content"])
            assert label is not None, "retrieval put no passage in the prompt"
            content = json.dumps(
                {
                    "answer": f"A grounded answer [{label.group(1)}].",
                    "sources": [label.group(1)],
                    "support": "stated",
                }
            )
        message: dict[str, Any] = {"role": "assistant", "content": content}
        return httpx.Response(
            200,
            json={
                "success": True,
                "errors": [],
                "result": {
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": reply.finish_reason}
                    ],
                    "usage": {"prompt_tokens": 1500, "completion_tokens": reply.completion_tokens},
                },
            },
        )

    def adapter(self, **kwargs: Any) -> WorkersAIChatProvider:
        return WorkersAIChatProvider(
            account_id="test-account",
            api_token=FAKE_TOKEN,
            client=httpx.AsyncClient(
                base_url="https://api.cloudflare.test/client/v4",
                transport=httpx.MockTransport(self.handle),
            ),
            **kwargs,
        )


def _workers_settings() -> Settings:
    return Settings(
        llm_provider=LLMProviderName.CLOUDFLARE_WORKERS_AI,
        cloudflare_account_id="test-account",
        cloudflare_workers_ai_token=FAKE_TOKEN,
    )


@pytest.fixture(scope="module")
def retrieval() -> Any:
    """The real corpus behind real retrieval, with offline embeddings and every
    match admitted, so each question reaches generation."""
    components = build_query_components(
        Settings(knowledge_root=Path("knowledge"), retrieval_min_similarity=-1.0)
    )
    run(components.prepare())
    return components.retrieval


def _service(retrieval: Any, provider: Provider) -> GroundedAnswerService:
    """The answering service as an unpaced evaluation builds it — with the
    evaluation's question deadline, not the production one."""
    return GroundedAnswerService(
        retrieval=retrieval,
        llm=provider.adapter(),
        context_policy=ContextPolicy(),
        deadline_seconds=evaluation_question_deadline(_workers_settings()),
    )


TRUNCATED = Reply('{"answer": "Marketing automation, web and', "length", 800, 13.0)
EMPTY_AT_LIMIT = Reply(None, "length", 800, 13.0)
SLOW_RECOVERY = 45.0  # beyond the old 30 s, within the new 50 s


# --- the per-request timeout --------------------------------------------------------


def test_the_ordinary_cap_keeps_its_timeout_and_only_the_recovery_gets_more():
    assert request_timeout_seconds(30.0, 800) == 30.0
    assert request_timeout_seconds(30.0, 1500) == pytest.approx(50.0)
    assert request_timeout_seconds(30.0, None) == 30.0
    # Derived from the slowest throughput recorded (38.7 tokens/s), with margin.
    assert MIN_OUTPUT_TOKENS_PER_SECOND < 38.7
    assert request_timeout_seconds(30.0, 1500) > 1500 / 38.7


def test_each_request_is_sent_with_the_timeout_of_its_own_cap():
    provider = Provider(generation=[Reply('{"x": 1}', seconds=1.0)])
    adapter = provider.adapter()
    for cap in (800, 1500):
        request = GenerationRequest(
            messages=(PromptMessage(role=MessageRole.USER, content="q"),),
            max_output_tokens=cap,
        )
        run(adapter.generate(request))

    assert [r["read_timeout"] for r in provider.requests] == [30.0, pytest.approx(50.0)]


# --- A, B, Phase 8: the two historical pipeline errors -------------------------------


def test_multi_marketing_automation_web_recovers_from_a_truncated_answer_in_a_slow_recovery(
    retrieval: Any,
):
    """v1.1.1: generation #1 stopped at 800 with half a JSON object. The
    recovery at 1500 now gets time for the slowest throughput on record."""
    provider = Provider(generation=[TRUNCATED, Reply("<cite>", seconds=SLOW_RECOVERY)])

    answer = run(_service(retrieval, provider).answer(QUESTION["multi-marketing-automation-web"]))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.citations
    assert answer.generation_attempts == 2
    generations = [r for r in provider.requests if not r["check"]]
    assert [(r["max_tokens"], r["read_timeout"]) for r in generations] == [
        (800, 30.0),
        (1500, pytest.approx(50.0)),
    ]
    assert generations[1]["seconds"] > 30.0, "the old timeout would have cut this off"
    first, second = (call for call in answer.provider_calls if call.call_type.value == "generation")
    assert (first.result.value, first.finish_reason, first.max_output_tokens) == (
        "unusable_reply",
        "length",
        800,
    )
    assert (second.result.value, second.max_output_tokens) == ("parsed", 1500)


def test_broad_project_scope_recovers_from_an_empty_check_in_a_slow_recovery(retrieval: Any):
    """v1.1.1: the grounding check spent all 800 tokens reasoning and returned
    no text. Its recovery at 1500 now has the time it needs."""
    provider = Provider(
        generation=[Reply("<cite>")],
        check=[EMPTY_AT_LIMIT, Reply('{"verdict": "supported"}', seconds=SLOW_RECOVERY)],
    )

    answer = run(_service(retrieval, provider).answer(QUESTION["broad-project-scope"]))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.grounding_check_attempts == 2
    checks = [r for r in provider.requests if r["check"]]
    assert [(r["max_tokens"], r["read_timeout"]) for r in checks] == [
        (800, 30.0),
        (1500, pytest.approx(50.0)),
    ]


# --- C: the whole legitimate recovery chain fits the question deadline ---------------


def test_the_longest_legitimate_question_fits_the_evaluation_deadline(retrieval: Any):
    """Generation 800 → recovery 1500 → check 800 → check recovery 1500, every
    step near its own limit: 28 + 45 + 28 + 45 = 146 s — impossible under the
    60 s production deadline, inside the evaluation's."""
    provider = Provider(
        generation=[Reply(TRUNCATED.content, "length", 800, 28.0), Reply("<cite>", seconds=45.0)],
        check=[Reply(None, "length", 800, 28.0), Reply('{"verdict": "supported"}', seconds=45.0)],
    )

    answer = run(_service(retrieval, provider).answer(QUESTION["broad-project-scope"]))
    simulated = sum(r["seconds"] for r in provider.requests)

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert len(provider.requests) == 4
    deadline = evaluation_question_deadline(_workers_settings())
    assert 60.0 < simulated < deadline
    # Every step at its own full timeout still fits, with retrieval's allowance.
    assert deadline == pytest.approx(30.0 + (30.0 + 50.0) + (30.0 + 50.0))


def test_the_production_deadline_is_unchanged():
    assert Settings().request_deadline_seconds == 60.0
    components = build_query_components(Settings(knowledge_root=Path("knowledge")))
    assert components.answers.deadline_seconds == 60.0


def test_the_evaluation_cli_runs_with_the_evaluation_deadline(monkeypatch: pytest.MonkeyPatch):
    from portfolio_rag import cli

    seen: list[float] = []
    build = build_query_components

    def spy(settings: Settings, **kwargs: Any) -> Any:
        seen.append(settings.request_deadline_seconds)
        return build(settings, **kwargs)

    monkeypatch.setattr(cli, "build_query_components", spy)
    cli.main(["eval", "run", "--dataset", "evaluation/questions.yaml", "--retrieval-only"])

    assert seen == [evaluation_question_deadline(get_settings())]


# --- D, E: a really hanging call still times out, with the attempts it always had ----


def test_a_recovery_slower_than_its_own_timeout_still_times_out_with_unchanged_attempts(
    retrieval: Any,
):
    hanging = Reply("<cite>", seconds=3600.0)
    provider = Provider(generation=[TRUNCATED, hanging])

    with pytest.raises(GenerationUnavailableError) as caught:
        run(_service(retrieval, provider).answer(QUESTION["multi-marketing-automation-web"]))

    failure = caught.value.failure
    assert failure is not None
    assert (failure.category.value, failure.detail) == ("timeout", "timeout")
    recoveries = [r for r in provider.requests if r["max_tokens"] == 1500]
    # One recovery, its transport attempts and nothing more: the budget is unchanged.
    assert len(recoveries) == TRANSPORT_ATTEMPTS == 3
    # The step's failure counts every request of the step: the first, plus the
    # recovery's transport attempts.
    assert failure.attempts == 1 + TRANSPORT_ATTEMPTS
    assert all(r["read_timeout"] == pytest.approx(50.0) for r in recoveries)


def test_the_attempt_budget_is_unchanged():
    assert (MAX_GENERATION_ATTEMPTS, MAX_GROUNDING_CHECK_ATTEMPTS) == (2, 2)
    assert (TRANSPORT_ATTEMPTS, PACER_ATTEMPTS) == (3, 3)


# --- F, G: what a 429 says about itself ------------------------------------------------


def _limited(body: Any, retry_after: str | None = None) -> WorkersAIChatProvider:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}

    def handler(_: httpx.Request) -> httpx.Response:
        if isinstance(body, str):
            return httpx.Response(429, text=body, headers=headers)
        return httpx.Response(429, json=body, headers=headers)

    return WorkersAIChatProvider(
        account_id="test-account",
        api_token=FAKE_TOKEN,
        max_attempts=1,
        client=httpx.AsyncClient(
            base_url="https://api.cloudflare.test/client/v4",
            transport=httpx.MockTransport(handler),
        ),
    )


def _rejected(adapter: WorkersAIChatProvider) -> LLMProviderError:
    request = GenerationRequest(messages=(PromptMessage(role=MessageRole.USER, content="q"),))
    with pytest.raises(LLMProviderError) as caught:
        run(adapter.generate(request))
    return caught.value


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        (3036, "daily_free_allocation_exhausted"),
        (3040, "capacity_exceeded"),
        (9999, None),
    ],
    ids=["daily-allocation", "capacity", "undocumented"],
)
def test_a_429_keeps_its_numeric_code_and_retry_after_and_no_provider_text(
    code: int, kind: str | None
):
    body = {"success": False, "errors": [{"code": code, "message": PROVIDER_TEXT}]}

    error = _rejected(_limited(body, retry_after="17"))
    failure = provider_failure(error)

    assert error.kind is ProviderFailureKind.RATE_LIMITED
    assert (error.status_code, error.provider_error_code) == (429, code)
    assert error.retry_after_seconds == 17.0
    assert error.rate_limit_kind == kind, "only a documented code is named"
    fields = failure.fields()
    assert fields["failure_detail"] == "rate_limited"
    assert (fields["status_code"], fields["provider_error_code"]) == (429, code)
    assert fields["rate_limit_kind"] == kind
    assert fields["retry_after_seconds"] == 17.0
    exported = json.dumps(fields) + error.message + error.describe()
    assert "SECRET-PROVIDER-TEXT" not in exported
    assert "0123abcd" not in exported
    assert FAKE_TOKEN not in exported


@pytest.mark.parametrize(
    "body",
    [{"errors": []}, {"errors": [{"code": "3036"}]}, {"errors": [{"code": True}]}, "not json"],
    ids=["no-code", "string-code", "bool-code", "not-json"],
)
def test_a_429_without_a_usable_code_is_still_a_plain_rate_limit(body: Any):
    error = _rejected(_limited(body))
    failure = provider_failure(error)

    assert error.kind is ProviderFailureKind.RATE_LIMITED
    assert (error.provider_error_code, error.rate_limit_kind) == (None, None)
    assert (failure.detail, failure.status_code) == ("rate_limited", 429)
    assert failure.category.value == "retryable_provider_error"


def test_a_429_through_the_pipeline_is_still_a_rate_limit_stop(retrieval: Any):
    """The guard reads the same facts it always did; the code only adds to them."""
    from portfolio_rag.evaluation.e2e import AbortReason
    from portfolio_rag.evaluation.runner import run_e2e_evaluation
    from tests.e2e_stack import dataset_of

    body = {"success": False, "errors": [{"code": 3036, "message": PROVIDER_TEXT}]}
    service = GroundedAnswerService(
        retrieval=retrieval, llm=_limited(body), context_policy=ContextPolicy()
    )
    question = next(q for q in DATASET.questions if q.id == "broad-project-scope")

    class Guard:
        def observe(self, record: Any) -> AbortReason | None:
            assert record.error.provider_error_code == 3036
            return AbortReason.RATE_LIMITED if record.error.status_code == 429 else None

    report = run(run_e2e_evaluation(dataset_of(question, question), service, guard=Guard()))

    assert report.aborted is not None and report.aborted.reason is AbortReason.RATE_LIMITED
    (record,) = report.records
    assert record.error is not None
    assert record.error.rate_limit_kind == "daily_free_allocation_exhausted"
