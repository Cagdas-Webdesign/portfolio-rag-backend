"""Paket 3, E0: the facts a malformed response keeps, and the experiment tool.

Two halves. The first pins that a response which is not a completion no longer
loses what it did report — its finish reason and token usage — on the way from
adapter to telemetry, and that nothing else of it is kept. The second pins the
experiment's design rather than its numbers: one context per question shared
by every variant, the variant as the only difference, the real parser, and
stop rules that end a run before it spends what it should not.

No request leaves the process.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.evaluation import EvaluationQuestion, QuestionCategory
from portfolio_rag.evaluation.experiment import (
    DEFAULT_VARIANTS,
    EXPERIMENT_FORMAT_VERSION,
    ExperimentError,
    ExperimentLimits,
    ExperimentMetadata,
    ExperimentRun,
    PreparedQuestion,
    StopReason,
    export_experiment,
    plan_calls,
    prepare_questions,
)
from portfolio_rag.infrastructure.llm import MistralChatProvider
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import GenerationRequest, MessageRole, PromptMessage, ResponseFormat
from portfolio_rag.rag.errors import provider_failure
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.telemetry import CallResult
from tests.doubles import ScriptedLLMProvider, ScriptedReply, grounded
from tests.e2e_stack import ANSWERABLE, e2e_retrieval
from tests.support import run
from tests.unit.test_retrieval_service import build as build_retrieval
from tests.workers_ai_transport import LIMIT, VALID, workers_ai

REQUEST = GenerationRequest(
    messages=(PromptMessage(role=MessageRole.USER, content="Which framework?"),),
    max_output_tokens=LIMIT,
    response_format=ResponseFormat.JSON_OBJECT,
)
USAGE = {"prompt_tokens": 2400, "completion_tokens": LIMIT}


def body(choice: Any = None, *, usage: dict[str, int] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"choices": [] if choice is None else [choice]}
    if usage is not None:
        result["usage"] = usage
    return {"success": True, "errors": [], "result": result}


def malformed(*responses: Any) -> LLMProviderError:
    provider, _ = workers_ai(*responses)
    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))
    assert caught.value.kind is ProviderFailureKind.MALFORMED_RESPONSE
    return caught.value


# --- 1-6. a malformed response keeps what it reported, and nothing else -----------


def test_a_message_without_text_keeps_its_finish_reason_and_usage():
    error = malformed(
        body(
            {"message": {"role": "assistant", "content": None}, "finish_reason": "length"},
            usage=USAGE,
        )
    )

    assert error.finish_reason == "length"
    assert error.usage is not None
    assert (error.usage.input_tokens, error.usage.output_tokens) == (2400, LIMIT)


def test_a_malformed_response_without_usage_reports_none():
    error = malformed(body({"message": {"role": "assistant"}, "finish_reason": "stop"}))

    assert error.finish_reason == "stop"
    assert error.usage is None


@pytest.mark.parametrize(
    ("response", "finish_reason", "usage"),
    [
        (body(usage=USAGE), None, True),  # no choices: usage is known, finish is not
        (body("not-an-object", usage=USAGE), None, True),
        (body({"finish_reason": "length"}, usage=USAGE), "length", True),  # no message
        ({"success": True, "result": None}, None, False),  # no result at all
        ({"success": False, "errors": [{"code": 1}]}, None, False),
    ],
    ids=["no-choices", "choice-not-object", "no-message", "no-result", "success-false"],
)
def test_each_malformed_shape_keeps_exactly_the_facts_it_had(
    response: dict[str, Any], finish_reason: str | None, usage: bool
):
    error = malformed(response)

    assert error.finish_reason == finish_reason
    assert (error.usage is not None) is usage


def test_the_facts_travel_into_the_failure_without_any_content():
    error = malformed(
        body(
            {
                "message": {"role": "assistant", "content": None, "reasoning": "SECRET-THOUGHT"},
                "finish_reason": "length",
            },
            usage=USAGE,
        )
    )

    failure = provider_failure(error)
    assert failure.detail == "malformed_response"
    assert failure.finish_reason == "length"
    assert (failure.input_tokens, failure.output_tokens) == (2400, LIMIT)
    assert failure.reply_characters is None, "no reply text arrived"
    everything = json.dumps(
        {**failure.fields(), "message": error.message, **vars(error)}, default=str
    )
    assert "SECRET-THOUGHT" not in everything
    assert "content" not in failure.fields()


def test_the_mistral_adapter_keeps_the_same_facts():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 400},
            },
        )

    provider = MistralChatProvider(
        api_key="test-value-not-a-real-credential",
        client=httpx.AsyncClient(
            base_url="https://api.mistral.test", transport=httpx.MockTransport(handler)
        ),
    )
    with pytest.raises(LLMProviderError) as caught:
        run(provider.generate(REQUEST))

    assert caught.value.kind is ProviderFailureKind.MALFORMED_RESPONSE
    assert caught.value.finish_reason == "length"
    assert caught.value.usage is not None and caught.value.usage.output_tokens == 400


def test_a_completion_is_read_exactly_as_before():
    provider, _ = workers_ai(
        body(
            {"message": {"role": "assistant", "content": VALID}, "finish_reason": "stop"},
            usage=USAGE,
        )
    )

    response = run(provider.generate(REQUEST))

    assert response.text == VALID
    assert response.finish_reason == "stop"
    assert response.usage is not None and response.usage.input_tokens == 2400


# --- 7-10. one context per question, the variant as the only difference -----------


def _prepared() -> tuple[PreparedQuestion, ...]:
    return run(prepare_questions([ANSWERABLE], e2e_retrieval(), ContextPolicy()))


def test_every_variant_and_repetition_sends_the_same_context():
    prepared = _prepared()
    llm = ScriptedLLMProvider(grounded("FastAPI [S1].", "S1"))
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=2), llm=llm, limits=_limits()
    )

    run(experiment.run())

    assert len(llm.requests) == 4
    assert len({request.messages for request in llm.requests}) == 1, "identical messages"
    assert {request.max_output_tokens for request in llm.requests} == {LIMIT}
    assert {request.temperature for request in llm.requests} == {llm.requests[0].temperature}


def test_variant_a_asks_for_json_and_variant_b_asks_for_nothing():
    (question,) = _prepared()

    assert question.request_for(ResponseFormat.JSON_OBJECT).response_format is (
        ResponseFormat.JSON_OBJECT
    )
    assert question.request_for(ResponseFormat.TEXT).response_format is ResponseFormat.TEXT


def test_the_text_variant_reaches_the_wire_without_a_response_format():
    (question,) = _prepared()
    provider, seen = workers_ai(
        body({"message": {"role": "assistant", "content": VALID}, "finish_reason": "stop"})
    )

    run(provider.generate(question.request_for(ResponseFormat.TEXT)))
    run(provider.generate(question.request_for(ResponseFormat.JSON_OBJECT)))

    assert "response_format" not in seen[0]
    assert seen[1]["response_format"] == {"type": "json_object"}
    assert seen[0]["messages"] == seen[1]["messages"]


def test_the_context_is_fingerprinted_not_kept():
    (question,) = _prepared()

    assert len(question.context_sha256) == 64
    assert question.context_sources >= 1


def test_variants_alternate_and_flip_order_every_repetition():
    prepared = _prepared()

    plan = plan_calls(prepared, DEFAULT_VARIANTS, repetitions=3)

    assert [(call.repetition, call.variant) for call in plan] == [
        (1, ResponseFormat.JSON_OBJECT),
        (1, ResponseFormat.TEXT),
        (2, ResponseFormat.TEXT),
        (2, ResponseFormat.JSON_OBJECT),
        (3, ResponseFormat.JSON_OBJECT),
        (3, ResponseFormat.TEXT),
    ]


def test_every_reply_is_judged_by_the_real_parser():
    prepared = _prepared()
    llm = ScriptedLLMProvider(
        grounded("FastAPI [S1].", "S1"),
        ScriptedReply(raw_text="Sure! It is FastAPI."),
        ScriptedReply(
            raw_text='```json\n{"answer": "FastAPI.", "sources": ["S1"], "support": "stated"}\n```'
        ),
        ScriptedReply(raw_text="{" + "\n" * 99, finish_reason="length"),
    )
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=2), llm=llm, limits=_limits()
    )

    run(experiment.run())

    results = [(call.record.result, call.record.failure) for call in experiment.calls]
    assert results[0] == (CallResult.PARSED, None)
    assert results[1][0] is CallResult.UNUSABLE_REPLY
    assert results[1][1] is not None and results[1][1].detail == "reply_not_json"
    assert results[2] == (CallResult.PARSED, None), "a fenced object is tolerated, as in production"
    assert results[3][0] is CallResult.UNUSABLE_REPLY
    assert experiment.calls[3].record.reply_visible_characters == 1


def test_a_question_that_retrieves_nothing_is_refused_before_any_call():
    empty, _, _ = build_retrieval([])
    question = EvaluationQuestion(id="q-x", category=QuestionCategory.DIRECT, question="Anything?")

    with pytest.raises(ExperimentError, match="retrieves nothing"):
        run(prepare_questions([question], empty, ContextPolicy()))


# --- 11-15. stop rules, budget, partial results ----------------------------------------


def _limits(**overrides: Any) -> ExperimentLimits:
    return ExperimentLimits(**{"max_calls": 100, **overrides})


def _error(kind: ProviderFailureKind, status: int | None = None) -> ScriptedReply:
    return ScriptedReply(error=LLMProviderError("scripted", kind=kind, status_code=status))


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (_error(ProviderFailureKind.RATE_LIMITED, 429), StopReason.RATE_LIMITED),
        (_error(ProviderFailureKind.HTTP_STATUS, 401), StopReason.AUTH_FAILURE),
        (_error(ProviderFailureKind.HTTP_STATUS, 403), StopReason.AUTH_FAILURE),
        (_error(ProviderFailureKind.HTTP_STATUS, 503), StopReason.PROVIDER_ERROR),
        (_error(ProviderFailureKind.TIMEOUT), StopReason.PROVIDER_ERROR),
        (_error(ProviderFailureKind.UNSPECIFIED), StopReason.PROVIDER_ERROR),
    ],
    ids=["429", "401", "403", "503", "timeout", "unexpected"],
)
def test_a_provider_failure_stops_the_run_and_keeps_what_was_paid_for(
    failure: ScriptedReply, reason: StopReason
):
    prepared = _prepared()
    llm = ScriptedLLMProvider(grounded("FastAPI [S1].", "S1"), failure, grounded("Never.", "S1"))
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=3), llm=llm, limits=_limits()
    )

    stopped = run(experiment.run())

    assert stopped is reason
    assert llm.call_count == 2, "no call after the one that stopped it, and no retry of it"
    assert [call.record.result for call in experiment.calls] == [
        CallResult.PARSED,
        CallResult.PROVIDER_ERROR,
    ]


def test_a_malformed_response_is_counted_and_the_run_goes_on():
    prepared = _prepared()
    llm = ScriptedLLMProvider(
        _error(ProviderFailureKind.MALFORMED_RESPONSE), grounded("FastAPI [S1].", "S1")
    )
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=2), llm=llm, limits=_limits()
    )

    assert run(experiment.run()) is StopReason.COMPLETED
    assert len(experiment.calls) == 4
    assert experiment.calls[0].record.result is CallResult.PROVIDER_ERROR


def test_max_calls_stops_before_the_limit_is_exceeded():
    prepared = _prepared()
    llm = ScriptedLLMProvider(grounded("FastAPI [S1].", "S1"))
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=5),
        llm=llm,
        limits=_limits(max_calls=3),
    )

    assert run(experiment.run()) is StopReason.MAX_CALLS
    assert llm.call_count == 3


def _usage_limits(budget: float) -> ExperimentLimits:
    return _limits(
        neuron_budget=budget, input_neurons_per_million=31818, output_neurons_per_million=68182
    )


def test_the_neuron_budget_stops_before_a_call_that_could_cross_it():
    prepared = _prepared()
    reply = body(
        {"message": {"role": "assistant", "content": VALID}, "finish_reason": "stop"}, usage=USAGE
    )
    provider, seen = workers_ai(reply)
    # 2400 in + 800 out ≈ 76.4 + 54.5 = 130.9 estimated neurons per call.
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=5),
        llm=provider,
        limits=_usage_limits(300),
    )

    assert run(experiment.run()) is StopReason.NEURON_BUDGET
    assert len(seen) == 2, "a third call could take 262 past 300"
    assert experiment.estimated_neurons == pytest.approx(261.8, abs=0.1)
    assert experiment.estimated_neurons <= 300


def test_missing_usage_is_unestimated_never_zero():
    prepared = _prepared()
    provider, _ = workers_ai(
        body({"message": {"role": "assistant", "content": VALID}, "finish_reason": "stop"})
    )
    experiment = ExperimentRun(
        plan=plan_calls(prepared, (ResponseFormat.JSON_OBJECT,), repetitions=2),
        llm=provider,
        limits=_usage_limits(1000),
    )

    run(experiment.run())

    assert experiment.unestimated_calls == 2
    assert experiment.estimated_neurons == 0.0
    assert all(call.record.input_tokens is None for call in experiment.calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_calls": 0},
        {"max_calls": 5, "neuron_budget": 100},
        {
            "max_calls": 5,
            "neuron_budget": -1,
            "input_neurons_per_million": 1,
            "output_neurons_per_million": 1,
        },
    ],
)
def test_limits_that_cannot_be_enforced_are_refused(kwargs: dict[str, Any]):
    with pytest.raises(ValueError):
        ExperimentLimits(**kwargs)


def test_the_export_holds_the_calls_the_contexts_and_a_summary_per_variant():
    prepared = _prepared()
    llm = ScriptedLLMProvider(
        grounded("FastAPI [S1].", "S1"),
        ScriptedReply(raw_text="Prose."),
        _error(ProviderFailureKind.RATE_LIMITED, 429),
    )
    experiment = ExperimentRun(
        plan=plan_calls(prepared, DEFAULT_VARIANTS, repetitions=3), llm=llm, limits=_limits()
    )
    run(experiment.run())

    payload = export_experiment(
        experiment,
        prepared,
        ExperimentMetadata(
            experiment_id="e" * 32,
            generated_at=datetime(2026, 10, 3, tzinfo=UTC),
            git_revision="a" * 40,
            git_dirty=False,
            dataset_path="evaluation/portfolio-questions.yaml",
            dataset_sha256="f" * 64,
            provider="cloudflare_workers_ai",
            model=ScriptedLLMProvider.MODEL,
            repetitions=3,
            variants=DEFAULT_VARIANTS,
            delay_seconds=8.0,
        ),
    )

    assert payload["format"] == EXPERIMENT_FORMAT_VERSION
    assert payload["experiment"]["stop_reason"] == "rate_limited"
    assert len(payload["calls"]) == 3, "the partial run, as far as it got"
    assert [c["variant"] for c in payload["calls"]] == ["json_object", "text", "text"]
    assert payload["contexts"][0]["context_sha256"] == prepared[0].context_sha256
    a, b = payload["summary"]["json_object"], payload["summary"]["text"]
    assert (a["calls"], a["failures"], a["failure_rate"]) == (1, 0, 0.0)
    assert (b["calls"], b["failures"], b["failure_rate"]) == (2, 2, 1.0)
    assert b["failure_details"] == {"rate_limited": 1, "reply_not_json": 1}
    assert payload["cost"]["estimated_neurons"] is None, "no rates, no estimate"
    text = json.dumps(payload)
    assert "Prose." not in text and "FastAPI [S1]" not in text
    assert "You are a knowledge assistant" not in text


# --- 16. nothing in the serving path reaches the tool -----------------------------------

_SERVING = ("api", "rag", "domain", "ports", "core", "infrastructure", "application", "ingestion")


def test_no_serving_module_imports_the_experiment():
    root = Path("src/portfolio_rag")
    offenders = []
    for package in (*(root / name for name in _SERVING), root / "main.py", root / "composition.py"):
        for path in [package] if package.is_file() else package.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = (
                    [node.module or ""]
                    if isinstance(node, ast.ImportFrom)
                    else [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
                if any("experiment" in name for name in names):
                    offenders.append(str(path))
    assert offenders == []
