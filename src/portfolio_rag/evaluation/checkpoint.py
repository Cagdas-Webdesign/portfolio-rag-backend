"""Checkpoint and resume for an acceptance run: one logical run, several segments.

An acceptance run asks 24 questions against a rate-limited, daily-capped
provider. When the provider says no more on question 11 — a 429, refused
credentials, an outage — the ten answers before it were paid for and are
valid. Without a checkpoint they are thrown away with the run.

**One logical run, several execution segments.** A run that is stopped from
outside is *paused*, not failed: the questions it completed are written down,
and a later invocation continues the same logical run from the next question.
The final artifact is the whole logical run — every question exactly once, in
suite order — and the gates are computed over all of it, by the same export
code a single uninterrupted run uses.

**Nothing about the run may change between segments.** A resume continues the
logical run only when everything that decides an answer is identical: commit,
tag, version, release source identity, dataset, suite, the ordered question
ids, corpus, embedding, retrieval policy, generation provider and model,
prompt and grounding-check versions, response formats, output limits and the
recovery policy (:data:`IDENTITY_FIELDS`). Any difference refuses the resume.
There is no best-effort continuation. Pacing is not part of the identity: a
run paused by a rate limit may legitimately continue more slowly, and every
segment records the pacing it ran with.

**A checkpoint holds results, never content.** A record is written as the
facts the export needs — outcome, answer, citations, provider-call telemetry —
and its retrieved passages as chunk ids and similarities. On resume the ids
are resolved against the corpus, which the identity has already proven
identical; a passage's text never touches the file. No credential, account id,
prompt or passage is written.

**Completed means completed.** A question counts as completed when it finished
on its own terms — answered, refused, or failed in a way the pipeline
classified. A question that ended *because* of the stop — the 429 that paused
the run, the streak of provider failures that made it systemic — is not
completed: it is asked again on resume. A crash can lose at most the question
that was being asked; the checkpoint is replaced atomically after every
completed one.

**Integrity is a check, not a seal.** The file carries a SHA-256 over its own
content, so an edited checkpoint is refused. Whoever can recompute the hash can
forge one; that is why the final artifact records every segment's identity and
the validator recomputes it from the artifact itself.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.domain.retrieval import RetrievedChunk, SourceCitation
from portfolio_rag.evaluation.dataset import EvaluationQuestion
from portfolio_rag.evaluation.e2e import AbortReason, E2EFailure, E2ERecord
from portfolio_rag.evaluation.metrics import score_retrieval
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.errors import (
    GenerationFailure,
    GenerationFailureCategory,
    RetrievalFailure,
    RetrievalStage,
)
from portfolio_rag.rag.generation import SupportVerdict
from portfolio_rag.rag.retrieval import RetrievalOutcome
from portfolio_rag.rag.service import AnswerOutcome
from portfolio_rag.rag.telemetry import CallResult, ProviderCallRecord, ProviderCallType
from portfolio_rag.rag.verification import GroundingVerdict

#: The shape of a checkpoint file. Bumped when a field changes meaning.
CHECKPOINT_FORMAT: Final = "acceptance-checkpoint-v1"

#: Stops that pause a logical run rather than end it: the provider's, the
#: credentials', the budget's. The questions asked before them stand, and the
#: run can continue later. An internal defect is not here — that is a crash in
#: this code, fixed by a change, which is a new release candidate.
PAUSE_REASONS: Final = frozenset(
    {
        AbortReason.RATE_LIMITED,
        AbortReason.AUTH_FAILURE,
        AbortReason.SYSTEMIC_PROVIDER_FAILURE,
        AbortReason.BUDGET_GUARD,
        AbortReason.RETRIEVAL_UNAVAILABLE,
    }
)

#: The stop of a segment whose process ended without closing it — a crash, a
#: killed terminal. Its completed questions were checkpointed one by one; the
#: question it was asking is lost and asked again.
PROCESS_INTERRUPTED: Final = "process_interrupted"

#: How often one question may be cut short by a process interruption before
#: the logical run can no longer be a release acceptance. A crash or a closed
#: terminal happens; the same question interrupted again is indistinguishable
#: from re-asking a question until it looks right, so it is not published.
PROCESS_INTERRUPTION_LIMIT_PER_QUESTION: Final = 1

#: Every stop after which another segment may follow.
RESUMABLE_STOPS: Final = frozenset(
    {reason.value for reason in PAUSE_REASONS} | {PROCESS_INTERRUPTED}
)

#: What makes two segments the same logical run, as paths into an export's
#: ``run`` section. The ordered question ids are added beside these. Pacing,
#: notes, timestamps and run ids are deliberately absent.
IDENTITY_FIELDS: Final = (
    ("git_revision",),
    ("git_tag",),
    ("git_dirty",),
    ("project_version",),
    ("release_source_identity",),
    ("tier",),
    ("rerun_of",),
    ("selected_question_ids",),
    ("dataset", "path"),
    ("dataset", "sha256"),
    ("dataset", "version"),
    ("dataset", "question_count"),
    ("suite",),
    ("suite_version",),
    ("suite_sha256",),
    ("question_count",),
    ("corpus", "sha256"),
    ("corpus", "documents"),
    ("corpus", "chunks"),
    ("embedding", "provider"),
    ("embedding", "model"),
    ("embedding", "dimensions"),
    ("embedding", "representation_version"),
    ("embedding", "identity"),
    ("query_representation_version",),
    ("vector_store",),
    ("retrieval_policy", "top_k"),
    ("retrieval_policy", "min_similarity"),
    ("retrieval_policy", "visibility"),
    ("generation", "provider"),
    ("generation", "model"),
    ("generation", "prompt_version"),
    ("generation", "grounding_check_version"),
    ("generation", "response_format"),
    ("generation", "grounding_check_response_format"),
    ("generation", "max_prompt_tokens"),
    ("generation", "output_reserve_tokens"),
    ("generation", "max_output_tokens"),
    ("generation", "recovery_output_tokens"),
    ("generation", "generation_attempt_limit"),
    ("generation", "grounding_check_attempt_limit"),
    ("generation", "transport_attempts"),
)


class CheckpointError(Exception):
    """A checkpoint that cannot be read, or may not be continued."""


# --- identity -----------------------------------------------------------------


def run_identity(run: Mapping[str, Any], question_ids: Sequence[str]) -> dict[str, Any]:
    """The identity of a logical run, from an export's ``run`` section.

    One function for the three places that need it — the checkpoint written
    by a run, the comparison before a resume, and the validator reading the
    final artifact — so that the three cannot disagree about what "the same
    run" means.
    """
    identity: dict[str, Any] = {}
    for path in IDENTITY_FIELDS:
        value: Any = run
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        identity[".".join(path)] = value
    identity["question_ids"] = list(question_ids)
    return identity


def identity_digest(identity: Mapping[str, Any]) -> str:
    """SHA-256 of an identity, independent of key order."""
    return hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()


def identity_mismatches(stored: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Every identity field that differs, named, with both values."""
    problems: list[str] = []
    for key in sorted(set(stored) | set(current)):
        before, now = stored.get(key), current.get(key)
        if before != now:
            if key == "question_ids":
                problems.append("question_ids: the ordered question ids differ")
            else:
                problems.append(f"{key}: checkpoint {_short(before)}, now {_short(now)}")
    return problems


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 24 else text[:21] + "…"


# --- records ------------------------------------------------------------------


def record_to_checkpoint(record: E2ERecord) -> dict[str, Any]:
    """Everything an export needs of one record, as plain data — no passage text."""
    retrieval = record.retrieval
    outcome = retrieval.outcome
    return {
        "question_id": record.question.id,
        "retrieval_observed": record.retrieval_observed,
        "retrieval": {
            "retrieved": [[item.chunk.id, item.similarity] for item in retrieval.retrieved],
            "outcome": (
                None
                if outcome is None
                else {
                    "matches_returned": outcome.matches_returned,
                    "below_threshold": outcome.below_threshold,
                    "unresolved": outcome.unresolved,
                    "withheld": outcome.withheld,
                    "duration_seconds": outcome.duration_seconds,
                }
            ),
        },
        "outcome": record.outcome.value if record.outcome is not None else None,
        "error_code": record.error_code,
        "error": record.error.fields() if record.error is not None else None,
        "answer": record.answer,
        "citations": [citation.model_dump(mode="json") for citation in record.citations],
        "verified_citations": record.verified_citations,
        "cites_expected_source": record.cites_expected_source,
        "unknown_labels": list(record.unknown_labels),
        "context_sources": record.context_sources,
        "generation_seconds": record.generation_seconds,
        "duration_seconds": record.duration_seconds,
        "failures": [failure.value for failure in record.failures],
        "support": record.support.value if record.support is not None else None,
        "claimed_labels": list(record.claimed_labels),
        "grounding_check": (
            record.grounding_check.value if record.grounding_check is not None else None
        ),
        "grounding_check_seconds": record.grounding_check_seconds,
        "generation_attempts": record.generation_attempts,
        "grounding_check_attempts": record.grounding_check_attempts,
        "provider_calls": [_call_to_checkpoint(call) for call in record.provider_calls],
        "retrieval_failure": (
            record.retrieval_failure.fields() if record.retrieval_failure is not None else None
        ),
    }


def record_from_checkpoint(
    data: Mapping[str, Any],
    question: EvaluationQuestion,
    chunks: Mapping[str, KnowledgeChunk],
) -> E2ERecord:
    """Rebuild a record. *chunks* is the corpus by chunk id; a retrieved id it
    does not hold means the corpus is not the one the checkpoint was made with."""
    if data.get("question_id") != question.id:
        raise CheckpointError(f"record for {data.get('question_id')!r} where {question.id} is due")
    raw = data["retrieval"]
    retrieved: list[RetrievedChunk] = []
    for chunk_id, similarity in raw["retrieved"]:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise CheckpointError(f"{question.id}: retrieved chunk {chunk_id} is not in the corpus")
        retrieved.append(RetrievedChunk(chunk=chunk, similarity=float(similarity)))
    counts = raw["outcome"]
    outcome = (
        None
        if counts is None
        else RetrievalOutcome(
            chunks=tuple(retrieved),
            matches_returned=int(counts["matches_returned"]),
            below_threshold=int(counts["below_threshold"]),
            unresolved=int(counts["unresolved"]),
            withheld=int(counts["withheld"]),
            duration_seconds=float(counts["duration_seconds"]),
        )
    )
    return E2ERecord(
        question=question,
        retrieval=score_retrieval(question, tuple(retrieved), outcome=outcome),
        retrieval_observed=bool(data["retrieval_observed"]),
        outcome=AnswerOutcome(data["outcome"]) if data["outcome"] is not None else None,
        error_code=data["error_code"],
        error=_failure(data["error"]) if data["error"] is not None else None,
        answer=str(data["answer"]),
        citations=tuple(SourceCitation.model_validate(item) for item in data["citations"]),
        verified_citations=int(data["verified_citations"]),
        cites_expected_source=bool(data["cites_expected_source"]),
        unknown_labels=tuple(data["unknown_labels"]),
        context_sources=int(data["context_sources"]),
        generation_seconds=float(data["generation_seconds"]),
        duration_seconds=float(data["duration_seconds"]),
        failures=tuple(E2EFailure(value) for value in data["failures"]),
        support=SupportVerdict(data["support"]) if data["support"] is not None else None,
        claimed_labels=tuple(data["claimed_labels"]),
        grounding_check=(
            GroundingVerdict(data["grounding_check"])
            if data["grounding_check"] is not None
            else None
        ),
        grounding_check_seconds=float(data["grounding_check_seconds"]),
        generation_attempts=int(data["generation_attempts"]),
        grounding_check_attempts=int(data["grounding_check_attempts"]),
        provider_calls=tuple(_call_from_checkpoint(call) for call in data["provider_calls"]),
        retrieval_failure=_retrieval_failure(data.get("retrieval_failure")),
    )


def _retrieval_failure(data: Mapping[str, Any] | None) -> RetrievalFailure | None:
    if data is None:
        return None
    return RetrievalFailure(
        stage=RetrievalStage(data["retrieval_stage"]),
        transient=bool(data["transient"]),
        detail=str(data["detail"]),
        status_code=data["status_code"],
        retry_after_seconds=data["retry_after_seconds"],
    )


def _call_to_checkpoint(call: ProviderCallRecord) -> dict[str, Any]:
    return {
        "call_type": call.call_type.value,
        "attempt": call.attempt,
        "model": call.model,
        "response_format": call.response_format.value,
        "elapsed_seconds": call.elapsed_seconds,
        "result": call.result.value,
        "max_output_tokens": call.max_output_tokens,
        "finish_reason": call.finish_reason,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "reply_characters": call.reply_characters,
        "reply_visible_characters": call.reply_visible_characters,
        "failure": call.failure.fields() if call.failure is not None else None,
        "http_attempts": call.http_attempts,
        "transport_attempts": call.transport_attempts,
        "pacing_attempts": call.pacing_attempts,
        "retry_after_seconds": call.retry_after_seconds,
    }


def _call_from_checkpoint(data: Mapping[str, Any]) -> ProviderCallRecord:
    return ProviderCallRecord(
        call_type=ProviderCallType(data["call_type"]),
        attempt=int(data["attempt"]),
        model=str(data["model"]),
        response_format=ResponseFormat(data["response_format"]),
        elapsed_seconds=float(data["elapsed_seconds"]),
        result=CallResult(data["result"]),
        max_output_tokens=data["max_output_tokens"],
        finish_reason=data["finish_reason"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        reply_characters=data["reply_characters"],
        reply_visible_characters=data["reply_visible_characters"],
        failure=_failure(data["failure"]) if data["failure"] is not None else None,
        # Absent from checkpoints written before attempts were recorded.
        http_attempts=data.get("http_attempts"),
        transport_attempts=data.get("transport_attempts"),
        pacing_attempts=data.get("pacing_attempts"),
        retry_after_seconds=data.get("retry_after_seconds"),
    )


def _failure(data: Mapping[str, Any]) -> GenerationFailure:
    return GenerationFailure(
        category=GenerationFailureCategory(data["failure_category"]),
        detail=str(data["failure_detail"]),
        step=ProviderCallType(data["failure_step"]),
        status_code=data["status_code"],
        retryable=data["retryable"],
        attempts=int(data["attempts"]),
        retry_after_seconds=data["retry_after_seconds"],
        finish_reason=data["finish_reason"],
        reply_characters=data["reply_characters"],
        reply_visible_characters=data["reply_visible_characters"],
        input_tokens=data["input_tokens"],
        output_tokens=data["output_tokens"],
        # Absent from checkpoints written before 429 codes were recorded.
        provider_error_code=data.get("provider_error_code"),
        rate_limit_kind=data.get("rate_limit_kind"),
        transport_attempts=data.get("transport_attempts"),
        pacing_attempts=data.get("pacing_attempts"),
    )


# --- segments -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SegmentUsage:
    """What one segment asked of the generation provider. Spend accounting is
    per segment: the day's known consumption must never count a call twice."""

    calls: int = 0
    interrupted_calls: int = 0
    """Calls made for questions the stop interrupted — paid for, not kept."""

    input_tokens: int = 0
    output_tokens: int = 0
    usage_reported: int = 0
    estimated_neurons: float = 0.0

    def fields(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "interrupted_calls": self.interrupted_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usage_reported": self.usage_reported,
            "usage_coverage": round(self.usage_reported / self.calls, 3) if self.calls else None,
            "estimated_neurons": round(self.estimated_neurons, 1),
        }


def segment_usage(
    asked: Sequence[E2ERecord],
    *,
    interrupted: int,
    neurons: Callable[[Sequence[ProviderCallRecord]], float],
) -> SegmentUsage:
    """Usage of the questions a segment asked; the last *interrupted* of them
    were cut off by its stop. *neurons* prices a set of calls."""
    calls = [call for record in asked for call in record.provider_calls]
    cut = asked[len(asked) - interrupted :] if interrupted else ()
    reported = [call for call in calls if call.usage_reported]
    return SegmentUsage(
        calls=len(calls),
        interrupted_calls=sum(len(record.provider_calls) for record in cut),
        input_tokens=sum(call.input_tokens or 0 for call in reported),
        output_tokens=sum(call.output_tokens or 0 for call in reported),
        usage_reported=len(reported),
        estimated_neurons=neurons(calls),
    )


def rate_limit_events(asked: Sequence[E2ERecord], *, segment: int, at: str) -> list[dict[str, Any]]:
    """Every question of a segment that ended on a rate limit. Facts only."""
    return [
        {
            "segment": segment,
            "question_id": record.question.id,
            "at": at,
            "status_code": record.error.status_code,
            "attempts": record.error.attempts,
            "retry_after_seconds": record.error.retry_after_seconds,
            "provider_error_code": record.error.provider_error_code,
            "rate_limit_kind": record.error.rate_limit_kind,
        }
        for record in asked
        if record.error is not None
        and (record.error.status_code == 429 or record.error.detail == "rate_limited")
    ]


# --- the checkpoint -----------------------------------------------------------


@dataclass(slots=True)
class AcceptanceCheckpoint:
    """The persistent state of one logical acceptance run.

    ``status`` is ``running`` while a segment is under way (a checkpoint left
    in that state is a crashed segment, and may be resumed), ``paused`` after
    an external stop, ``complete`` when every question is done, and
    ``aborted`` after an internal defect, which is never resumed.
    """

    path: Path
    logical_run_id: str
    started_at: str
    identity: dict[str, Any]
    records: list[dict[str, Any]] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)
    rate_limit_events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    pause_reason: str | None = None
    stopped_at_question: str | None = None
    updated_at: str | None = None

    # --- reading ---------------------------------------------------------

    @property
    def question_ids(self) -> list[str]:
        ids: list[str] = self.identity["question_ids"]
        return ids

    @property
    def completed_ids(self) -> list[str]:
        return [str(record["question_id"]) for record in self.records]

    @property
    def resumable(self) -> bool:
        return self.status in {"running", "paused"}

    def restore(
        self, questions: Sequence[EvaluationQuestion], chunks: Sequence[KnowledgeChunk]
    ) -> tuple[E2ERecord, ...]:
        """The completed records, rebuilt against the suite and the corpus."""
        by_id = {question.id: question for question in questions}
        corpus = {chunk.id: chunk for chunk in chunks}
        restored: list[E2ERecord] = []
        for data in self.records:
            question = by_id.get(str(data.get("question_id")))
            if question is None:
                raise CheckpointError(f"{data.get('question_id')!r} is not a question of the suite")
            restored.append(record_from_checkpoint(data, question, corpus))
        return tuple(restored)

    # --- writing ---------------------------------------------------------

    def begin_segment(self, *, run_id: str, started_at: str, **details: Any) -> int:
        """Open the next execution segment and persist it before any question."""
        number = len(self.segments) + 1
        self.segments.append(
            {
                "segment": number,
                "run_id": run_id,
                "identity_sha256": identity_digest(self.identity),
                "started_at": started_at,
                "ended_at": None,
                "status": "running",
                "stop_reason": None,
                "stop_question_id": None,
                "completed_question_ids": [],
                "interrupted_question_ids": [],
                **SegmentUsage().fields(),
                "ledger_recorded": False,
                **details,
            }
        )
        self.status, self.pause_reason, self.stopped_at_question = "running", None, None
        self.save(updated_at=started_at)
        return number

    def progress(
        self,
        completed: Sequence[E2ERecord],
        *,
        segment_completed: Sequence[str],
        usage: SegmentUsage,
        at: str,
    ) -> None:
        """Persist every question completed so far. Called after each one."""
        self.records = [record_to_checkpoint(record) for record in completed]
        self.segments[-1].update(
            {"completed_question_ids": list(segment_completed), **usage.fields()}
        )
        self.save(updated_at=at)

    def finish_segment(
        self,
        completed: Sequence[E2ERecord],
        *,
        segment_completed: Sequence[str],
        interrupted: Sequence[str],
        usage: SegmentUsage,
        stop: AbortReason | None,
        stop_question_id: str | None,
        events: Sequence[Mapping[str, Any]],
        at: str,
    ) -> None:
        """Close the current segment as complete, paused or aborted."""
        self.records = [record_to_checkpoint(record) for record in completed]
        if stop is None:
            status = "complete" if len(self.records) == len(self.question_ids) else "paused"
        elif stop in PAUSE_REASONS:
            status = "paused"
        else:
            status = "aborted"
        self.segments[-1].update(
            {
                "ended_at": at,
                "status": status,
                "stop_reason": stop.value if stop is not None else None,
                "stop_question_id": stop_question_id,
                "completed_question_ids": list(segment_completed),
                "interrupted_question_ids": list(interrupted),
                **usage.fields(),
            }
        )
        self.rate_limit_events.extend(dict(event) for event in events)
        self.status = status
        self.pause_reason = stop.value if stop is not None else None
        self.stopped_at_question = stop_question_id
        self.save(updated_at=at)

    def close_crashed_segment(self, *, at: str) -> dict[str, Any] | None:
        """Close a segment its process never closed, as ``process_interrupted``.

        Returns that segment, so its spend — known for the questions it
        completed, unknown for the one it was asking — can be entered in the
        ledger once. ``None`` when the last segment was closed properly.
        """
        if self.status != "running" or not self.segments:
            return None
        segment = self.segments[-1]
        completed = self.completed_ids
        segment.update(
            {
                "ended_at": self.updated_at,
                "status": "paused",
                "stop_reason": PROCESS_INTERRUPTED,
                "stop_question_id": None,
                "interrupted_question_ids": (
                    [self.question_ids[len(completed)]]
                    if len(completed) < len(self.question_ids)
                    else []
                ),
            }
        )
        self.status, self.pause_reason = "paused", PROCESS_INTERRUPTED
        self.save(updated_at=at)
        return segment

    def mark_ledger_recorded(self, *, at: str) -> None:
        self.segments[-1]["ledger_recorded"] = True
        self.save(updated_at=at)

    def save(self, *, updated_at: str) -> None:
        self.updated_at = updated_at
        write_json_atomic(self.path, self.payload())

    def payload(self) -> dict[str, Any]:
        completed = self.completed_ids
        remaining = [qid for qid in self.question_ids if qid not in set(completed)]
        body: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "logical_run_id": self.logical_run_id,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "pause_reason": self.pause_reason,
            "stopped_at_question": self.stopped_at_question,
            "identity": self.identity,
            "identity_sha256": identity_digest(self.identity),
            "progress": {
                "total_question_ids": list(self.question_ids),
                "completed_question_ids": completed,
                "next_question_id": remaining[0] if remaining else None,
                "completed": len(completed),
                "remaining": len(remaining),
            },
            "segments": self.segments,
            "rate_limit_events": self.rate_limit_events,
            "records": self.records,
        }
        return {**body, "integrity_sha256": _digest(body)}

    # --- the export ------------------------------------------------------

    def export_fields(self, *, completed_at: str | None) -> dict[str, Any]:
        """What the final artifact records about its execution, under ``run``."""
        return {
            "logical_run_id": self.logical_run_id,
            "original_started_at": self.started_at,
            "completed_at": completed_at,
            "resume_count": max(len(self.segments) - 1, 0),
            # The suite in its order: what the identity names, whatever has
            # been completed so far.
            "planned_question_ids": list(self.question_ids),
            "execution_segments": [
                {key: value for key, value in segment.items() if key != "ledger_recorded"}
                for segment in self.segments
            ],
            "rate_limit_events": list(self.rate_limit_events),
        }

    # --- loading ---------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> AcceptanceCheckpoint:
        """Read and verify a checkpoint. Any doubt is a :class:`CheckpointError`."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CheckpointError(f"checkpoint {path} could not be read: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise CheckpointError(f"{path} is not a {CHECKPOINT_FORMAT} file")
        stored = payload.pop("integrity_sha256", None)
        if stored != _digest(payload):
            raise CheckpointError(f"{path} does not match its own integrity hash: it was edited")
        try:
            identity = payload["identity"]
            if payload["identity_sha256"] != identity_digest(identity):
                raise CheckpointError(f"{path}: the identity does not match its digest")
            checkpoint = cls(
                path=path,
                logical_run_id=str(payload["logical_run_id"]),
                started_at=str(payload["started_at"]),
                identity=dict(identity),
                records=list(payload["records"]),
                segments=list(payload["segments"]),
                rate_limit_events=list(payload["rate_limit_events"]),
                status=str(payload["status"]),
                pause_reason=payload["pause_reason"],
                stopped_at_question=payload["stopped_at_question"],
                updated_at=payload["updated_at"],
            )
        except (KeyError, TypeError) as exc:
            raise CheckpointError(f"{path} is missing a field: {exc}") from exc
        problems = checkpoint.structure_problems()
        if problems:
            raise CheckpointError(f"{path}: {'; '.join(problems)}")
        return checkpoint

    def structure_problems(self) -> list[str]:
        """Completed questions must be a prefix of the suite, each once."""
        problems: list[str] = []
        completed = self.completed_ids
        if len(set(completed)) != len(completed):
            problems.append("a question is recorded as completed more than once")
        if completed != self.question_ids[: len(completed)]:
            problems.append("the completed questions are not the suite's first questions in order")
        segment_ids = [
            qid for segment in self.segments for qid in segment["completed_question_ids"]
        ]
        if segment_ids != completed:
            problems.append("the segments do not account for exactly the completed questions")
        return problems


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace *path* with *payload* so that a reader sees the old file or the
    new one, never half of either: write a temporary file in the same
    directory, flush it to disk, and rename it over the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    # The rename is durable once the directory entry is; not every platform
    # lets a directory be opened for that, and the data is already safe.
    with contextlib.suppress(OSError):
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


# --- the validator's view -----------------------------------------------------


def process_interruptions(segments: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How often each question was cut short by a process interruption."""
    counts: dict[str, int] = {}
    for segment in segments:
        if segment.get("stop_reason") == PROCESS_INTERRUPTED:
            for qid in segment.get("interrupted_question_ids") or []:
                counts[str(qid)] = counts.get(str(qid), 0) + 1
    return counts


def over_interruption_limit(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    """The questions interrupted by the process more often than allowed."""
    return sorted(
        qid
        for qid, count in process_interruptions(segments).items()
        if count > PROCESS_INTERRUPTION_LIMIT_PER_QUESTION
    )


def segment_problems(payload: Mapping[str, Any]) -> list[str]:
    """How the execution record of an artifact contradicts the artifact.

    Checks, from the file alone: every question exactly once and in the order
    the segments completed them; every segment run under the identity the
    artifact itself states; segments numbered, all but the last paused by an
    external stop; and the usage the segments report adding up to the calls
    the questions record.
    """
    run = payload.get("run") or {}
    questions = payload.get("questions") or []
    segments = run.get("execution_segments")
    if not isinstance(segments, list) or not segments:
        return ["the execution segments are missing"]
    if not all(isinstance(segment, Mapping) for segment in segments):
        return ["an execution segment is not an object"]
    problems: list[str] = []
    ids = [question.get("id") for question in questions]

    if [segment.get("segment") for segment in segments] != list(range(1, len(segments) + 1)):
        problems.append("the execution segments are not numbered 1..n")
    if run.get("resume_count") != len(segments) - 1:
        problems.append("resume_count does not match the execution segments")
    if run.get("logical_run_id") != run.get("run_id"):
        problems.append("the logical run id is not the artifact's run id")

    completed = [qid for segment in segments for qid in segment.get("completed_question_ids") or []]
    if len(set(completed)) != len(completed):
        problems.append("a question was completed in more than one segment")
    if completed != ids:
        problems.append("the segments' completed questions are not the recorded questions in order")

    # The suite in its order. A paused run records a prefix of it; a complete
    # run records all of it. Older artifacts without it are complete runs.
    planned = [str(qid) for qid in run.get("planned_question_ids") or ids]
    if ids != planned[: len(ids)]:
        problems.append("the recorded questions are not the suite's questions in order")
    if run.get("complete") is True and ids != planned:
        problems.append("the run is complete but does not record every planned question")

    expected = identity_digest(run_identity(run, planned))
    for segment in segments:
        if segment.get("identity_sha256") != expected:
            problems.append(
                f"segment {segment.get('segment')} ran under a different identity than the "
                "artifact states (commit, source, dataset, suite, order, provider, model or "
                "configuration changed)"
            )

    for segment in segments[:-1]:
        if segment.get("status") != "paused" or segment.get("stop_reason") not in RESUMABLE_STOPS:
            problems.append(
                f"segment {segment.get('segment')} was not paused by an external stop, "
                "yet another segment followed it"
            )
    # Every interrupted question stays on record and is visibly asked again
    # later: never completed before its interruption, completed after it.
    for index, segment in enumerate(segments):
        earlier = {
            qid
            for before in segments[: index + 1]
            for qid in before.get("completed_question_ids") or []
        }
        later = {
            qid
            for after in segments[index + 1 :]
            for qid in after.get("completed_question_ids") or []
        }
        interrupted = list(segment.get("interrupted_question_ids") or [])
        for qid in interrupted:
            if qid in earlier:
                problems.append(
                    f"segment {segment.get('segment')} interrupted {qid}, already completed"
                )
            elif run.get("complete") is True and qid not in later:
                problems.append(
                    f"{qid} was interrupted in segment {segment.get('segment')} "
                    "and never asked again"
                )
        if (
            segment.get("stop_reason") == PROCESS_INTERRUPTED
            and not interrupted
            and len(earlier) < len(ids)
        ):
            problems.append(
                f"segment {segment.get('segment')} was interrupted without naming "
                "the question it was asking"
            )

    last = segments[-1]
    if run.get("complete") is True and last.get("status") != "complete":
        problems.append("the run is complete but its last segment is not")
    if run.get("complete") is not True and last.get("status") == "complete":
        problems.append("the last segment is complete but the run is not")

    recorded_calls = sum(len(question.get("provider_calls") or []) for question in questions)
    kept = sum(
        int(segment.get("calls") or 0) - int(segment.get("interrupted_calls") or 0)
        for segment in segments
    )
    if kept != recorded_calls:
        problems.append(
            f"the segments report {kept} kept provider calls, the questions record {recorded_calls}"
        )
    observed = (payload.get("operations") or {}).get("observed") or {}
    if "calls" in observed and observed["calls"] != sum(
        int(segment.get("calls") or 0) for segment in segments
    ):
        problems.append("the observed calls do not equal the segments' calls")
    if observed.get("estimated_neurons") is not None:
        total = sum(float(segment.get("estimated_neurons") or 0.0) for segment in segments)
        if abs(float(observed["estimated_neurons"]) - total) > 0.1 * len(segments):
            problems.append("the observed estimate does not equal the segments' estimates")
    return problems


# --- staleness: only the latest state of a logical run may be continued ----------


def stale_checkpoint_problem(
    checkpoint: AcceptanceCheckpoint, entries: Sequence[Any]
) -> str | None:
    """Why *checkpoint* is not the latest state of its logical run, if it is not.

    The ledger records every closed segment of every acceptance run. A
    checkpoint that does not know a segment the ledger knows — or holds it as
    still running, or as another run — is an older copy: continuing it would
    ask again questions a later segment already asked. A crashed segment has
    no ledger line yet, and a copy of the current file is the current state;
    neither is stale.
    """
    known = sorted(
        (entry for entry in entries if _segment_of(entry, checkpoint.logical_run_id)),
        key=lambda entry: int(entry.segment),
    )
    if not known:
        return None
    latest = int(known[-1].segment)
    ours = len(checkpoint.segments)
    if latest > ours:
        return (
            f"STALE_CHECKPOINT: logical run {checkpoint.logical_run_id} is at segment {latest} "
            f"in the ledger; this checkpoint ends at segment {ours}. Resume from the latest "
            "checkpoint of the run."
        )
    for entry in known:
        segment = checkpoint.segments[int(entry.segment) - 1]
        if segment.get("run_id") != entry.run_id or segment.get("status") == "running":
            return (
                f"STALE_CHECKPOINT: logical run {checkpoint.logical_run_id}, segment "
                f"{entry.segment}: the ledger recorded it closed as run {entry.run_id}; this "
                "checkpoint holds an earlier state of it."
            )
    return None


def _segment_of(entry: Any, logical_run_id: str) -> bool:
    return (
        getattr(entry, "tier", None) == "acceptance"
        and getattr(entry, "logical_run_id", None) == logical_run_id
        and getattr(entry, "segment", None) is not None
    )


# --- one process per checkpoint ---------------------------------------------------


class CheckpointLockedError(Exception):
    """Another process holds the checkpoint, or may still hold it."""


@dataclass(slots=True)
class CheckpointLock:
    """An exclusive claim on a checkpoint for one process, as a lock file beside
    it. Holds no secret: a run id, a process id, a hash of the host name, a
    timestamp and the checkpoint's own path."""

    path: Path
    audit: dict[str, Any] | None = None
    """Set when the lock replaced one left behind: what it replaced, and why."""

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def lock_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.name}.lock")


def acquire_lock(
    checkpoint: Path, *, logical_run_id: str | None, at: str, break_lock: bool = False
) -> CheckpointLock:
    """Claim *checkpoint* for this process, atomically.

    A lock left by a process that provably no longer exists — same host, the
    process id gone — is replaced, and the replacement is recorded. Any other
    lock refuses, unless the operator explicitly breaks it, which is recorded
    too. Time alone never frees a lock.
    """
    path = lock_path(checkpoint)
    content = {
        "logical_run_id": logical_run_id,
        "pid": os.getpid(),
        "host": _host(),
        "created_at": at,
        "checkpoint": checkpoint.as_posix(),
    }
    audit: dict[str, Any] | None = None
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            held = _read_lock(path)
            if held is not None and _held_by_dead_process(held):
                audit = {"stale_lock_recovered": held, "reason": "holding process no longer exists"}
            elif break_lock:
                audit = {"lock_broken_by_operator": held}
            else:
                raise CheckpointLockedError(
                    f"checkpoint {checkpoint} is locked by "
                    f"{_describe(held)}: another resume may be running. If it is not, "
                    f"remove {path} or pass --break-lock (recorded in the artifact)."
                ) from None
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(content, handle)
        return CheckpointLock(path=path, audit=audit)
    raise CheckpointLockedError(f"checkpoint {checkpoint} was locked while it was being claimed")


# --- one process per logical run ---------------------------------------------------

#: A logical run id as this tool mints it: a request id, 32 lower-case hex
#: digits. Anything else is refused before it becomes part of a file name.
_LOGICAL_RUN_ID: Final = re.compile(r"[0-9a-f]{32}")


@dataclass(slots=True)
class RunLock:
    """An exclusive claim on one logical run, whichever checkpoint file names it.

    The checkpoint lock (:func:`acquire_lock`) guards one file; two copies of a
    checkpoint at two paths are two files, and each can be claimed. This lock
    is named after the logical run instead, beside the ledger every resume of
    that run reads, so the copies meet at one file.

    Held as an operating-system advisory lock (``flock``) on that file for as
    long as the process keeps it open. The kernel releases it when the process
    ends, however it ends, so a crash leaves nothing to clean up and nothing to
    judge: a lock that is held is held by a running process, and is never
    taken from it. The file itself stays, holding only who last claimed it —
    removing it would let a second process lock a file the next one no longer
    sees.
    """

    path: Path
    _descriptor: int | None

    def release(self) -> None:
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def run_lock_path(ledger: Path, logical_run_id: str) -> Path:
    """Where the lock of *logical_run_id* lives: beside the ledger, which is
    what every segment of the run already shares."""
    return ledger.parent / "locks" / f"acceptance-{logical_run_id}.lock"


def acquire_run_lock(
    ledger: Path, logical_run_id: str, *, at: str, checkpoint: Path | None = None
) -> RunLock:
    """Claim *logical_run_id* for this process, or refuse at once.

    Never waits and never breaks: a held lock means a running process is
    continuing the run — ``--break-lock`` does not apply, since there is no
    left-behind lock to break. Nothing about the process is decided from its
    process id or from time.
    """
    if not _LOGICAL_RUN_ID.fullmatch(logical_run_id):
        raise CheckpointError(f"{logical_run_id!r} is not a logical run id")
    path = run_lock_path(ledger, logical_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        held = _read_lock(path)
        raise CheckpointLockedError(
            f"logical run {logical_run_id} is being continued by {_describe(held)}, from "
            f"{(held or {}).get('checkpoint')!r}: one process per logical run, whichever "
            "checkpoint copy it was started from. This lock is held by a running process and "
            f"is released when that process ends; --break-lock does not apply. Lock: {path}"
        ) from None
    except BaseException:
        os.close(descriptor)
        raise
    content = {
        "logical_run_id": logical_run_id,
        "pid": os.getpid(),
        "host": _host(),
        "created_at": at,
        "checkpoint": checkpoint.as_posix() if checkpoint is not None else None,
    }
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, json.dumps(content).encode("utf-8"))
    except BaseException:
        RunLock(path=path, _descriptor=descriptor).release()
        raise
    return RunLock(path=path, _descriptor=descriptor)


def _host() -> str:
    """The host, as a hash: enough to tell this machine from another, nothing
    about which machine it is."""
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:16]


def _read_lock(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _held_by_dead_process(held: Mapping[str, Any]) -> bool:
    """Only a certainty frees a lock: this host, and no such process."""
    pid = held.get("pid")
    if held.get("host") != _host() or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False  # it exists, under another user
    return False


def _describe(held: Mapping[str, Any] | None) -> str:
    if held is None:
        return "an unreadable lock file"
    return (
        f"process {held.get('pid')} (logical run {held.get('logical_run_id')}, "
        f"since {held.get('created_at')})"
    )
