"""Run the gold grounding cases against one or two judges — offline by default.

    PYTHONPATH=src uv run python -m tests.grounding_compare.harness            # plan only
    PYTHONPATH=src uv run python -m tests.grounding_compare.harness --live \\
        --providers groq --max-calls 20 --output evaluation/results/....json

Without ``--live`` nothing leaves this machine: the plan is printed and the
run ends. With it, every call is one HTTP request — no transport retry, no
recovery, no repeat on 429 — and the run stops at ``--max-calls`` or at the
first rate limit of a provider. Keys are read from the environment and never
printed or written: ``GROQ_API_KEY``; ``PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID``
and ``PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN``.

Both judges receive the identical `GenerationRequest` built by
`rag.verification.build_grounding_check_request`, and every reply is read by
`rag.verification.read_grounding_check` — the production prompt and parser.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import httpx2 as httpx

from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.evaluation.budget import cost_profile
from portfolio_rag.ingestion.chunking import chunk_knowledge_base
from portfolio_rag.ingestion.loader import collect_knowledge_base
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import GenerationRequest, ResponseFormat
from portfolio_rag.rag.context import build_context
from portfolio_rag.rag.tokens import estimate_tokens
from portfolio_rag.rag.verification import (
    GROUNDING_CHECK_VERSION,
    build_grounding_check_request,
    read_grounding_check,
)
from tests.doubles import make_chunk
from tests.grounding_compare.cases import (
    CASES,
    CorpusPassage,
    GroundingCase,
    InlinePassage,
)

GROQ_URL: Final = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL: Final = "openai/gpt-oss-20b"
CLOUDFLARE_MODEL: Final = "@cf/openai/gpt-oss-120b"

#: The production first-attempt cap. No recovery runs here, so a reply that
#: stops at it is recorded as what it is.
MAX_OUTPUT_TOKENS: Final = 800
TIMEOUT_SECONDS: Final = 30.0

Outcome = Literal["supported", "not_supported", "unusable", "error", "rate_limited"]


@dataclass(frozen=True, slots=True)
class CallResult:
    outcome: Outcome
    rule: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    latency_seconds: float | None = None
    status_code: int | None = None
    retry_after: str | None = None
    #: From an HTTP error body: ``error.type`` and ``error.code`` only, and only
    #: when they look like identifiers. Never the message, never generated text.
    error_type: str | None = None
    error_code: str | None = None


class Judge(Protocol):
    name: str
    model: str

    async def judge(self, request: GenerationRequest) -> CallResult: ...

    async def aclose(self) -> None: ...


# --- building the identical inputs ---------------------------------------------


def _corpus_index(root: Path) -> dict[tuple[str, str], KnowledgeChunk]:
    report = collect_knowledge_base(root)
    index: dict[tuple[str, str], KnowledgeChunk] = {}
    for chunk in chunk_knowledge_base(report.documents):
        if chunk.heading_path:
            index.setdefault((chunk.document_id, chunk.heading_path[-1]), chunk)
    return index


def build_request(
    case: GroundingCase, corpus: dict[tuple[str, str], KnowledgeChunk]
) -> GenerationRequest:
    """The production grounding-check request for *case*, labels ``S1…`` in order."""
    chunks: list[KnowledgeChunk] = []
    for position, passage in enumerate(case.passages):
        if isinstance(passage, CorpusPassage):
            key = (passage.document_id, passage.heading)
            if key not in corpus:
                raise LookupError(f"case {case.id}: no passage {key} in the corpus")
            chunks.append(corpus[key])
        else:
            assert isinstance(passage, InlinePassage)
            chunks.append(
                make_chunk(
                    f"{case.id}--{position:04d}",
                    passage.text,
                    document_id=f"case-{position}",
                    ordinal=position,
                )
            )
    context = build_context(
        [RetrievedChunk(chunk=chunk, similarity=0.9) for chunk in chunks], available_tokens=10_000
    )
    if len(context.sources) != len(chunks):
        raise ValueError(f"case {case.id}: a passage did not fit or was a duplicate")
    return build_grounding_check_request(
        question=case.question,
        answer=case.answer,
        cited=context.sources,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def case_set_sha256(cases: Sequence[GroundingCase]) -> str:
    blob = json.dumps([asdict(case) for case in cases], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# --- the judges --------------------------------------------------------------------


def _read(text: str) -> tuple[Outcome, str | None]:
    verdict, rule = read_grounding_check(text)
    if verdict.value == "supported":
        return "supported", None
    if verdict.value == "not_supported":
        return "not_supported", None
    return "unusable", rule


def groq_payload(
    request: GenerationRequest, *, model: str, reasoning_effort: str | None
) -> dict[str, Any]:
    """The grounding request in Groq's OpenAI-compatible shape."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        # Never read, so never asked for: reasoning stays on Groq's side.
        "include_reasoning": False,
    }
    if request.max_output_tokens is not None:
        payload["max_completion_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.response_format is ResponseFormat.JSON_OBJECT:
        payload["response_format"] = {"type": "json_object"}
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return payload


@dataclass
class GroqJudge:
    api_key: str = field(repr=False)
    model: str = GROQ_DEFAULT_MODEL
    reasoning_effort: str | None = None
    client: httpx.AsyncClient | None = None
    name: str = "groq"

    def __post_init__(self) -> None:
        self._owns = self.client is None
        self._client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_SECONDS))

    async def judge(self, request: GenerationRequest) -> CallResult:
        payload = groq_payload(request, model=self.model, reasoning_effort=self.reasoning_effort)
        started = time.perf_counter()
        try:
            response = await self._client.post(
                GROQ_URL, json=payload, headers={"Authorization": f"Bearer {self.api_key}"}
            )
        except httpx.TimeoutException:
            return CallResult("error", rule="timeout", latency_seconds=_since(started))
        except httpx.RequestError as exc:
            return CallResult("error", rule=type(exc).__name__, latency_seconds=_since(started))
        latency = _since(started)
        if response.status_code == 429:
            return CallResult(
                "rate_limited",
                status_code=429,
                retry_after=response.headers.get("retry-after"),
                latency_seconds=latency,
            )
        if response.status_code >= 400:
            # The body is never echoed: it is upstream text. Only the two
            # identifier fields are kept — see `safe_error_fields`.
            error_type, error_code = safe_error_fields(response)
            return CallResult(
                "error",
                rule="http_status",
                status_code=response.status_code,
                latency_seconds=latency,
                error_type=error_type,
                error_code=error_code,
            )
        try:
            body = response.json()
            choice = body["choices"][0]
            content = choice["message"].get("content")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            return CallResult("unusable", rule="malformed_body", latency_seconds=latency)
        usage = body.get("usage") if isinstance(body, dict) else None
        details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
        outcome, rule = _read(content if isinstance(content, str) else "")
        return CallResult(
            outcome,
            rule=rule,
            input_tokens=_int(usage, "prompt_tokens"),
            output_tokens=_int(usage, "completion_tokens"),
            reasoning_tokens=_int(details, "reasoning_tokens"),
            finish_reason=choice.get("finish_reason"),
            latency_seconds=latency,
        )

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()


@dataclass
class CloudflareJudge:
    """The production adapter, one HTTP attempt per call."""

    provider: Any
    model: str = CLOUDFLARE_MODEL
    name: str = "cloudflare"

    async def judge(self, request: GenerationRequest) -> CallResult:
        started = time.perf_counter()
        try:
            response = await self.provider.generate(request)
        except LLMProviderError as exc:
            if exc.status_code == 429:
                return CallResult("rate_limited", status_code=429, latency_seconds=_since(started))
            usage = exc.usage
            return CallResult(
                "unusable" if exc.kind.value == "malformed_response" else "error",
                rule=exc.kind.value,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                finish_reason=exc.finish_reason,
                status_code=exc.status_code,
                latency_seconds=_since(started),
            )
        outcome, rule = _read(response.text)
        return CallResult(
            outcome,
            rule=rule,
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
            finish_reason=response.finish_reason,
            latency_seconds=_since(started),
        )

    async def aclose(self) -> None:
        await self.provider.aclose()


#: What an identifier looks like. Anything else — prose, a quoted generation,
#: something carrying a key — is dropped, not trimmed.
_SAFE_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def safe_error_fields(response: httpx.Response) -> tuple[str | None, str | None]:
    """``error.type`` and ``error.code`` of an OpenAI-style error body, or ``None``.

    ``{"error": {"message": "…", "type": "invalid_request_error", "code": "…",
    "failed_generation": "…"}}``: only the two identifiers are read. The message
    and any generated text are never looked at.
    """
    try:
        body = response.json()
    except ValueError:
        return None, None
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return None, None

    def identifier(value: object) -> str | None:
        if isinstance(value, str) and _SAFE_IDENTIFIER.match(value):
            return value
        return None

    return identifier(error.get("type")), identifier(error.get("code"))


def _since(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def _int(mapping: object, key: str) -> int | None:
    value = mapping.get(key) if isinstance(mapping, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# --- running and scoring --------------------------------------------------------


Sleeper = Callable[[float], Awaitable[None]]


async def run_cases(
    cases: Sequence[GroundingCase],
    requests: dict[str, GenerationRequest],
    judges: Sequence[Judge],
    *,
    max_calls: int,
    delay_seconds: float = 0.0,
    sleep: Sleeper = asyncio.sleep,
) -> dict[str, dict[str, CallResult | None]]:
    """Ask every judge about every case, in order — never more than *max_calls*.

    A judge that answers 429 is not asked again: its remaining cases stay
    ``None`` (not run). Nothing is retried.
    """
    results: dict[str, dict[str, CallResult | None]] = {
        judge.name: {case.id: None for case in cases} for judge in judges
    }
    calls = 0
    stopped: set[str] = set()
    for case in cases:
        for judge in judges:
            if judge.name in stopped or calls >= max_calls:
                continue
            if calls and delay_seconds:
                await sleep(delay_seconds)
            result = await judge.judge(requests[case.id])
            calls += 1
            results[judge.name][case.id] = result
            if result.outcome == "rate_limited":
                stopped.add(judge.name)
    return results


def score(
    cases: Sequence[GroundingCase], results: dict[str, CallResult | None], *, model: str
) -> dict[str, Any]:
    """Count against the gold labels. Anything short of ``supported`` is a refusal
    to confirm — fail-closed, exactly as the service reads it."""
    scored = [case for case in cases if case.scored]
    ran = [(case, results[case.id]) for case in scored if results.get(case.id) is not None]
    counts = {
        "correctly_accepted": 0,
        "correctly_rejected": 0,
        "falsely_accepted": 0,
        "falsely_rejected": 0,
    }
    false_accepts: list[str] = []
    for case, result in ran:
        assert result is not None
        accepted = result.outcome == "supported"
        if case.expected == "supported":
            counts["correctly_accepted" if accepted else "falsely_rejected"] += 1
        elif accepted:
            counts["falsely_accepted"] += 1
            false_accepts.append(case.id)
        else:
            counts["correctly_rejected"] += 1
    answered = [r for _, r in ran if r and r.outcome in ("supported", "not_supported", "unusable")]
    verdicts = [r for r in answered if r.outcome != "unusable"]
    tokens = [r for _, r in ran if r and r.input_tokens is not None and r.output_tokens is not None]
    latencies = [r.latency_seconds for _, r in ran if r and r.latency_seconds is not None]
    profile = cost_profile("cloudflare_workers_ai", model)
    return {
        "scored_cases": len(scored),
        "ran": len(ran),
        **counts,
        "false_accept_case_ids": false_accepts,
        "json_verdict_rate": round(len(verdicts) / len(answered), 3) if answered else None,
        "unusable_or_error": sum(
            1 for _, r in ran if r and r.outcome in ("unusable", "error", "rate_limited")
        ),
        "mean_input_tokens": _mean([r.input_tokens for r in tokens]),
        "mean_output_tokens": _mean([r.output_tokens for r in tokens]),
        "mean_latency_seconds": _mean(latencies),
        "estimated_neurons": round(
            sum(profile.estimate(r.input_tokens or 0, r.output_tokens or 0) for r in tokens), 1
        )
        if profile
        else None,
        "safe_for_production": not false_accepts and len(ran) == len(scored),
    }


def _mean(values: Sequence[float | int | None]) -> float | None:
    numbers = [float(v) for v in values if v is not None]
    return round(sum(numbers) / len(numbers), 1) if numbers else None


# --- command line ---------------------------------------------------------------


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - a local developer tool
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grounding_compare", description=__doc__)
    parser.add_argument("--live", action="store_true", help="Actually call the providers.")
    parser.add_argument("--providers", default="groq", help="groq, cloudflare or groq,cloudflare")
    parser.add_argument(
        "--max-calls", type=int, default=0, help="Hard limit; required with --live."
    )
    parser.add_argument("--delay-seconds", type=float, default=8.0)
    parser.add_argument("--groq-model", default=GROQ_DEFAULT_MODEL)
    parser.add_argument("--groq-reasoning-effort", choices=["low", "medium", "high"])
    parser.add_argument("--root", type=Path, default=Path("knowledge"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        metavar="ID",
        help="Run only this case; may be given more than once. Default: every case.",
    )
    args = parser.parse_args(argv)

    names = [name.strip() for name in args.providers.split(",") if name.strip()]
    if not names or any(name not in ("groq", "cloudflare") for name in names):
        parser.error("--providers takes groq, cloudflare or both")
    known = {case.id for case in CASES}
    unknown = [case_id for case_id in args.case_id if case_id not in known]
    if unknown:
        parser.error(f"unknown --case-id: {', '.join(unknown)}")
    selected = tuple(case for case in CASES if not args.case_id or case.id in args.case_id)
    corpus = _corpus_index(args.root)
    requests = {case.id: build_request(case, corpus) for case in selected}
    planned = len(selected) * len(names)
    est = {
        case.id: sum(estimate_tokens(m.content) for m in requests[case.id].messages)
        for case in selected
    }
    print(
        f"cases: {len(selected)} ({sum(c.scored for c in selected)} scored) · "
        f"judges: {', '.join(names)}"
    )
    print(
        f"planned calls: {planned} · estimated input tokens per call: "
        f"{min(est.values())}-{max(est.values())}"
    )
    print(f"case set sha256: {case_set_sha256(CASES)[:16]} · prompt: {GROUNDING_CHECK_VERSION}")
    if not args.live:
        print("Dry run: no request was sent. Add --live --max-calls N to call the providers.")
        return 0
    if args.max_calls <= 0:
        parser.error("--live needs --max-calls")
    if args.output is None or args.output.exists():
        parser.error("--live needs a new --output path (existing files are never overwritten)")

    judges: list[Judge] = []
    if "groq" in names:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            parser.error("GROQ_API_KEY is not set")
        judges.append(
            GroqJudge(
                api_key=key, model=args.groq_model, reasoning_effort=args.groq_reasoning_effort
            )
        )
    if "cloudflare" in names:
        from portfolio_rag.infrastructure.llm import WorkersAIChatProvider

        account = os.environ.get("PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID", "")
        token = os.environ.get("PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN", "")
        if not (account and token):
            parser.error("Cloudflare Workers AI credentials are not set")
        judges.append(
            CloudflareJudge(
                WorkersAIChatProvider(account_id=account, api_token=token, max_attempts=1)
            )
        )

    async def _run() -> dict[str, dict[str, CallResult | None]]:
        try:
            return await run_cases(
                selected,
                requests,
                judges,
                max_calls=args.max_calls,
                delay_seconds=args.delay_seconds,
            )
        finally:
            for judge in judges:
                await judge.aclose()

    results = asyncio.run(_run())
    export = {
        "format": "grounding-compare-v1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "grounding_check_version": GROUNDING_CHECK_VERSION,
        "case_set_sha256": case_set_sha256(CASES),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_calls": args.max_calls,
        # A subset is a diagnosis, never a verdict on a model.
        "selected_case_ids": [case.id for case in selected] if args.case_id else None,
        "judges": {
            judge.name: {
                "model": judge.model,
                "reasoning_effort": getattr(judge, "reasoning_effort", None),
                "summary": score(selected, results[judge.name], model=judge.model),
                "results": {
                    case_id: asdict(result) if result else None
                    for case_id, result in results[judge.name].items()
                },
            }
            for judge in judges
        },
        "cases": [
            {
                "id": c.id,
                "category": c.category,
                "origin": c.origin,
                "expected": c.expected,
                "scored": c.scored,
            }
            for c in selected
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for judge in judges:
        print(judge.name, json.dumps(export["judges"][judge.name]["summary"], ensure_ascii=False))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
