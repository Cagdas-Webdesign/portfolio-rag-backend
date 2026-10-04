"""Release acceptance: what one end-to-end export proves, decided once and by rule.

An end-to-end export already says what a run did. This module says whether
that export is a **release acceptance** — the one artifact that may claim
"this commit, with this configuration, ran this suite under these rules, and
passed" — and nothing else. It adds no reliability behaviour and runs no
provider: it reads an export (plain JSON) and judges it.

**One decision, two callers.** :func:`assess` is what the CLI writes into a
fresh export as ``release_acceptance``, and it is what
:func:`validate_artifact` recomputes from a file later. The validator never
trusts the stored verdict: it re-derives the gates from each question's
recorded failures, recounts the outcomes, re-runs :func:`assess`, and refuses
an artifact whose stored claims differ from what its own data says.

**The rules are frozen by version.** The gates, the conditions and the
safety invariants below are :data:`ACCEPTANCE_PROTOCOL_VERSION`. Changing any
of them is a new protocol version, decided before a run — never a
re-reading of a result after it. The full protocol, including the rerun rule,
is ``docs/RELEASE_ACCEPTANCE.md``.

**What a release acceptance does not prove.** The run executes this commit's
source in a local process against the configured providers. It does not
exercise a container image or a deployment; the artifact says so in its own
fields rather than leaving a reader to assume otherwise.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from portfolio_rag.evaluation.e2e import E2E_FORMAT_VERSION, GATES, INTERNAL_LABEL, E2EFailure

#: The version of the rules below. Recorded in every artifact; an artifact
#: written under another version is not judged by these rules.
ACCEPTANCE_PROTOCOL_VERSION: Final = "release-acceptance-v1"

#: The only tier whose run can be a release acceptance.
ACCEPTANCE_TIER: Final = "acceptance"

#: The four gates of the export, frozen by name. Each is PASS when no question
#: broke it (see :data:`portfolio_rag.evaluation.e2e.GATES`).
RELEASE_GATES: Final = (
    "no_pipeline_errors",
    "citations_verified",
    "unanswerable_refused",
    "nothing_internal_published",
)

#: The fifth gate: a count, required to be exactly zero.
INTERNAL_LEAKS_GATE: Final = "internal_leaks"

_LEAK_FAILURES: Final = frozenset({E2EFailure.LABEL_LEAK.value, E2EFailure.INTERNAL_PASSAGE.value})

#: What a release candidate *is*: every file the acceptance run executes or
#: reads, or that decides what is installed — paths from the repository root.
#: Everything else (documentation, tests, CI, the edge worker, the Dockerfile)
#: cannot change what an acceptance run does, so a change to it alone is not a
#: new candidate. A test classifies every tracked top-level entry, so a new
#: one has to be placed deliberately.
SOURCE_IDENTITY_INCLUDES: Final = (
    "src",
    "knowledge",
    "evaluation",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)

#: Inside the includes, what is output rather than input: exports, summaries
#: and the ledger. Never part of the identity.
SOURCE_IDENTITY_EXCLUDES: Final = ("evaluation/results",)


def source_identity(files: Iterable[tuple[str, bytes]]) -> str:
    """SHA-256 over *files* — ``(path, content)`` pairs, paths POSIX and
    relative to the repository root — in path order. Content, not metadata:
    the same files give the same identity on any machine and in any commit."""
    digest = hashlib.sha256()
    for path, content in sorted(files):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


#: Abort reasons that are the provider's, not this code's or this budget's.
#: Only a run that failed for one of these, or only on availability, may be
#: rerun once without a new commit.
PROVIDER_ABORT_REASONS: Final = frozenset({"rate_limited", "systemic_provider_failure"})

#: The safety invariants frozen for the release, each with the existing tests
#: that hold it. Referenced in every artifact by name; a test asserts every
#: test named here exists, so the reference cannot silently go stale. Nothing
#: here implements an invariant — the pipeline does, and is unchanged.
SAFETY_INVARIANTS: Final[Mapping[str, tuple[str, ...]]] = {
    "strict_generation_parser": (
        "tests/unit/test_generation_contract.py::"
        "test_a_reply_that_is_not_the_requested_object_is_a_failed_generation",
        "tests/unit/test_refusal_contract.py::"
        "test_an_unrecognized_verdict_is_an_unusable_generation",
    ),
    "citation_validation": (
        "tests/unit/test_answer_service.py::"
        "test_an_answer_citing_only_labels_that_do_not_exist_is_not_published",
        "tests/unit/test_answer_service.py::"
        "test_a_model_naming_a_file_instead_of_a_label_gets_no_citation",
    ),
    "grounding_validation": (
        "tests/unit/test_grounding_check.py::"
        "test_a_real_citation_that_does_not_carry_the_claim_fails_closed",
        "tests/unit/test_refusal_contract.py::test_inferred_is_refused_however_real_its_citation",
    ),
    "fail_closed_on_technical_failure": (
        "tests/unit/test_grounding_check.py::"
        "test_a_check_reply_that_is_no_verdict_is_a_technical_failure_that_publishes_nothing",
        "tests/unit/test_structured_output_reliability.py::"
        "test_two_degenerate_samples_fail_closed_after_exactly_two_generations",
    ),
    "controlled_refusal": (
        "tests/unit/test_answer_service.py::"
        "test_no_provider_is_called_when_there_is_nothing_to_ground_an_answer_in",
        "tests/unit/test_answer_service.py::"
        "test_a_substantive_answer_with_no_valid_source_is_not_published",
    ),
    "public_only_retrieval": (
        "tests/unit/test_retrieval_service.py::test_the_filter_is_a_constant_and_not_an_argument",
        "tests/unit/test_retrieval_service.py::"
        "test_an_internal_passage_is_not_returned_even_when_it_matches_best",
    ),
    "conversation_is_not_evidence": (
        "tests/unit/test_conversation_context.py::"
        "test_an_answer_carried_only_by_the_conversation_is_not_published",
        "tests/unit/test_conversation_context.py::"
        "test_a_label_from_the_conversation_cannot_become_a_citation",
    ),
    "recovery_passes_safety_again": (
        "tests/unit/test_generation_regeneration.py::"
        "test_a_regenerated_reply_still_has_to_pass_the_grounding_check",
        "tests/unit/test_generation_regeneration.py::"
        "test_a_regenerated_reply_with_an_invented_label_is_still_refused",
    ),
    "telemetry_stays_internal": (
        "tests/integration/test_chat.py::test_the_response_carries_no_internals",
        "tests/unit/test_provider_call_telemetry.py::test_no_record_or_log_line_holds_content",
    ),
}

#: What the artifact says about the thing it tested. Fixed text, not a claim
#: anyone can widen with a flag.
TESTED_SCOPE: Final = {
    "tested": "source_commit",
    "deployed_image_tested": False,
    "statement": (
        "The run executed the recorded commit's source in a local process against the "
        "configured providers and vector store. It did not exercise a container image or a "
        "deployment. An image is covered by this result only if it was built from the "
        "recorded commit."
    ),
}

#: The provenance a release acceptance must carry, as paths into ``run``.
#: ``None`` or a missing key at any of these is incomplete provenance.
REQUIRED_PROVENANCE: Final = (
    ("run_id",),
    ("generated_at",),
    ("project_version",),
    ("git_revision",),
    ("git_dirty",),
    ("release_source_identity",),
    ("tier",),
    ("suite",),
    ("suite_version",),
    ("suite_sha256",),
    ("question_count",),
    ("dataset", "sha256"),
    ("dataset", "version"),
    ("corpus", "sha256"),
    ("embedding", "provider"),
    ("embedding", "model"),
    ("embedding", "identity"),
    ("vector_store",),
    ("retrieval_policy", "top_k"),
    ("retrieval_policy", "min_similarity"),
    ("retrieval_policy", "visibility"),
    ("generation", "provider"),
    ("generation", "model"),
    ("generation", "prompt_version"),
    ("generation", "grounding_check_version"),
    ("generation", "response_format"),
    ("generation", "max_output_tokens"),
    ("generation", "generation_attempt_limit"),
)

_COMMIT_SHA: Final = re.compile(r"[0-9a-f]{40}")
_VERSION: Final = re.compile(r"\d+\.\d+\.\d+")


# --- the verdict --------------------------------------------------------------


def internal_leaks(payload: Mapping[str, Any]) -> int:
    """Questions whose answer leaked an internal label or whose retrieval
    surfaced an internal passage. Counted from the per-question records."""
    return sum(
        1
        for question in payload.get("questions") or []
        if _LEAK_FAILURES & set(question.get("failures") or [])
    )


def recomputed_gates(payload: Mapping[str, Any]) -> dict[str, bool]:
    """Each gate, derived from the questions' failures and nothing else."""
    broken = {
        failure
        for question in payload.get("questions") or []
        for failure in question.get("failures") or []
    }
    return {
        name: not ({failure.value for failure in failures} & broken)
        for name, failures in GATES.items()
    }


def missing_provenance(payload: Mapping[str, Any]) -> list[str]:
    """The required provenance fields the artifact does not carry."""
    run = payload.get("run") or {}
    missing: list[str] = []
    for path in REQUIRED_PROVENANCE:
        value: Any = run
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if value is None:
            missing.append(".".join(path))
    revision = run.get("git_revision")
    if isinstance(revision, str) and not _COMMIT_SHA.fullmatch(revision):
        missing.append("git_revision (not a full commit sha)")
    tag, version = run.get("git_tag"), run.get("project_version")
    if version is not None and not (isinstance(version, str) and _VERSION.fullmatch(version)):
        missing.append(f"project_version {version!r} (not a MAJOR.MINOR.PATCH version)")
    # The tag is optional — an untagged commit is a normal release candidate.
    if tag is not None and version is not None and tag != f"v{version}":
        # A tag on the commit that names another version is a contradiction
        # in the release's identity, not a detail.
        missing.append(f"git_tag {tag} (does not match project_version {version})")
    return missing


def assess(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Decide whether *payload* is a passing release acceptance.

    Every condition is named and reported, so a FAIL says which rule it broke.
    PASS needs all of them: there is no partial pass and no weighting.
    """
    run = payload.get("run") or {}
    questions = payload.get("questions") or []
    gates = recomputed_gates(payload)
    leaks = internal_leaks(payload)
    missing = missing_provenance(payload)

    conditions = {
        "acceptance_tier": run.get("tier") == ACCEPTANCE_TIER,
        "whole_suite": not run.get("selected_question_ids"),
        "run_complete": run.get("complete") is True,
        "run_not_aborted": run.get("aborted") is None,
        "every_question_recorded": bool(questions) and len(questions) == run.get("question_count"),
        "provenance_complete": not missing,
        "clean_tree": run.get("git_dirty") is False,
        "gates_pass": all(gates[name] for name in RELEASE_GATES),
        "no_internal_leaks": leaks == 0,
    }
    reasons = [_REASONS[name] for name, ok in conditions.items() if not ok]
    if missing:
        reasons.append(f"missing provenance: {', '.join(missing)}")
    reasons += [f"gate {name} FAIL" for name in RELEASE_GATES if not gates[name]]

    return {
        "protocol": ACCEPTANCE_PROTOCOL_VERSION,
        "verdict": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "reasons": reasons,
        "gates": {
            **{name: "PASS" if gates[name] else "FAIL" for name in RELEASE_GATES},
            INTERNAL_LEAKS_GATE: leaks,
        },
        "commit": run.get("git_revision"),
        "release_source_identity": run.get("release_source_identity"),
        "project_version": run.get("project_version"),
        "git_dirty": run.get("git_dirty"),
        "suite": run.get("suite"),
        "suite_version": run.get("suite_version"),
        "question_count": run.get("question_count"),
        "rerun_of": run.get("rerun_of"),
        "scope": dict(TESTED_SCOPE),
        "safety_invariants": sorted(SAFETY_INVARIANTS),
    }


_REASONS: Final = {
    "acceptance_tier": "not an acceptance-tier run",
    "whole_suite": "the run was narrowed to selected questions",
    "run_complete": "the run is not complete",
    "run_not_aborted": "the run was aborted",
    "every_question_recorded": "the recorded questions do not match the question count",
    "provenance_complete": "provenance is incomplete",
    "clean_tree": "the working tree was not clean (or its state is unknown)",
    "gates_pass": "an acceptance gate failed",
    "no_internal_leaks": "internal content leaked",
}


def with_release_acceptance(payload: dict[str, Any]) -> dict[str, Any]:
    """*payload* with its ``release_acceptance`` section, decided by :func:`assess`."""
    return {**payload, "release_acceptance": assess(payload)}


# --- the validator ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcceptanceCheck:
    """What validating one artifact found."""

    verdict: str | None
    """The recomputed verdict, or ``None`` when the artifact is unreadable."""

    reasons: tuple[str, ...]
    """Why the recomputed verdict is FAIL. Empty for PASS."""

    problems: tuple[str, ...]
    """Ways the artifact is malformed, inconsistent with itself, or carries
    something that must not be published. Any problem rejects it outright."""

    @property
    def consistent(self) -> bool:
        return not self.problems

    def accepted(self, *, publication: bool) -> bool:
        """For publication: consistent *and* PASS. For development: consistent."""
        if publication:
            return self.consistent and self.verdict == "PASS"
        return self.consistent


def validate_artifact(payload: Any, *, secrets: Iterable[str] = ()) -> AcceptanceCheck:
    """Check an export as a release-acceptance artifact.

    *secrets* are configured credential values; any of them appearing anywhere
    in the artifact is a problem. Only checked, never echoed.
    """
    if not isinstance(payload, dict):
        return AcceptanceCheck(None, (), ("the artifact is not a JSON object",))

    problems: list[str] = []
    if payload.get("format") != E2E_FORMAT_VERSION:
        problems.append(f"format is {payload.get('format')!r}, not {E2E_FORMAT_VERSION!r}")
    for section in ("run", "gates", "metrics", "questions", "release_acceptance"):
        if section not in payload:
            problems.append(f"missing section `{section}`")
    if problems:
        return AcceptanceCheck(None, (), tuple(problems))

    questions = payload["questions"]
    if not isinstance(questions, list) or not all(isinstance(q, dict) for q in questions):
        return AcceptanceCheck(None, (), ("`questions` is not a list of objects",))
    problems += _consistency_problems(payload)
    problems += _leak_problems(payload, secrets)

    recomputed = assess(payload)
    stored = payload["release_acceptance"]
    if not isinstance(stored, dict) or stored.get("protocol") != ACCEPTANCE_PROTOCOL_VERSION:
        problems.append(
            f"release_acceptance was not written under protocol {ACCEPTANCE_PROTOCOL_VERSION!r}"
        )
    else:
        for key in ("verdict", "conditions", "gates"):
            if stored.get(key) != recomputed[key]:
                problems.append(
                    f"release_acceptance.{key} does not match what the artifact's own data says"
                )
    return AcceptanceCheck(
        verdict=recomputed["verdict"],
        reasons=tuple(recomputed["reasons"]),
        problems=tuple(problems),
    )


def _consistency_problems(payload: Mapping[str, Any]) -> list[str]:
    """Stored aggregates against the records they claim to summarize."""
    problems: list[str] = []
    run = payload["run"]
    questions = payload["questions"]

    ids = [question.get("id") for question in questions]
    if len(set(ids)) != len(ids):
        problems.append("a question id appears more than once")
    if run.get("question_count") != len(questions):
        problems.append(
            f"question_count is {run.get('question_count')!r}, "
            f"but {len(questions)} questions are recorded"
        )
    dataset_count = (run.get("dataset") or {}).get("question_count")
    if isinstance(dataset_count, int) and len(questions) > dataset_count:
        problems.append("more questions recorded than the dataset contains")

    gates = recomputed_gates(payload)
    stored_gates = payload["gates"]
    if not isinstance(stored_gates, dict) or set(stored_gates) != set(RELEASE_GATES):
        problems.append("the gate set differs from the frozen release gates")
    elif stored_gates != gates:
        problems.append("the stored gates do not match the questions' recorded failures")

    expected_pass = run.get("aborted") is None and all(gates.values())
    if payload.get("passed") is not expected_pass:
        problems.append("`passed` does not match the gates and the run status")
    if run.get("complete") is not (run.get("aborted") is None):
        problems.append("`complete` and `aborted` contradict each other")

    outcomes = _count(question.get("outcome") for question in questions)
    try:
        stored_outcomes = payload["metrics"]["answers"]["outcomes"]
        pipeline_errors = payload["metrics"]["robustness"]["pipeline_errors"]
    except (KeyError, TypeError):
        problems.append("the answer and robustness metrics are missing")
    else:
        if {k: v for k, v in stored_outcomes.items() if v} != outcomes:
            problems.append("the outcome counts do not match the recorded questions")
        if pipeline_errors != outcomes.get("error", 0):
            problems.append("the pipeline error count does not match the recorded questions")

    stored_failures = payload.get("failures") or {}
    derived: dict[str, list[str]] = {}
    for question in questions:
        for failure in question.get("failures") or []:
            derived.setdefault(failure, []).append(question.get("id"))
    if {k: sorted(v) for k, v in stored_failures.items()} != {
        k: sorted(v) for k, v in derived.items()
    }:
        problems.append("the failure index does not match the questions' recorded failures")
    return problems


def _count(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


# --- publication safety -------------------------------------------------------

#: Keys that never belong in a published artifact, whatever their value.
_FORBIDDEN_KEYS: Final = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "token",
        "access_token",
        "secret",
        "password",
        "credentials",
        "account_id",
        "index_name",
        "prompt",
        "system_prompt",
        "messages",
        "conversation",
        "turns",
        "passage",
        "passages",
        "content",
        "context_text",
        "raw_text",
        "reply",
    }
)

#: Value shapes that are a credential or a header, wherever they appear.
_SECRET_SHAPES: Final = (
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/-]{8,}"),
    re.compile(r"(?i)\bauthorization\s*:"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _leak_problems(payload: Mapping[str, Any], secrets: Iterable[str]) -> list[str]:
    problems: list[str] = []
    secret_values = [value for value in secrets if len(value) >= 8]
    for path, key, value in _walk(payload, ()):
        where = ".".join(path) or "<root>"
        if key is not None and key.lower() in _FORBIDDEN_KEYS:
            problems.append(f"forbidden field `{key}` at {where}")
        if not isinstance(value, str):
            continue
        if any(pattern.search(value) for pattern in _SECRET_SHAPES):
            problems.append(f"credential-shaped value at {where}")
        if any(secret in value for secret in secret_values):
            problems.append(f"a configured credential appears at {where}")
        if key == "answer" and INTERNAL_LABEL.search(value):
            problems.append(f"an internal context marker in the answer at {where}")
    return problems


def _walk(
    value: Any, path: tuple[str, ...], key: str | None = None
) -> Iterator[tuple[tuple[str, ...], str | None, Any]]:
    yield path, key, value
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            yield from _walk(child, (*path, str(child_key)), str(child_key))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)), None)


# --- the rerun rule -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcceptanceAttempt:
    """One acceptance run as the ledger remembers it. Enough to apply the
    rerun rule; nothing else."""

    run_id: str
    commit_sha: str | None
    git_dirty: bool | None
    status: str
    abort_reason: str | None
    failed_gates: tuple[str, ...] | None
    """``None`` for an entry written before gates were recorded."""

    source_identity: str | None = None
    """``None`` for an entry written before the source identity was recorded;
    such an entry is matched by its commit instead."""

    @property
    def counts(self) -> bool:
        """Whether this attempt used up part of its candidate's allowance. Only
        a clean run of a known commit could have been a release acceptance."""
        return self.commit_sha is not None and self.git_dirty is False

    def is_candidate(self, *, commit: str, source_identity: str | None) -> bool:
        """Whether this attempt ran the same release candidate.

        The candidate is the source identity, not the commit: a commit that
        changed only run outputs is the same candidate under a new sha.
        """
        if self.source_identity is not None and source_identity is not None:
            return self.source_identity == source_identity
        return self.commit_sha == commit

    @property
    def passed(self) -> bool:
        return self.status == "complete" and self.failed_gates == ()

    @property
    def provider_outlier(self) -> bool:
        """Failed on the provider's availability alone — not on safety, not on
        a defect, not on the budget."""
        if self.failed_gates is None:
            return False
        if not set(self.failed_gates) <= {"no_pipeline_errors"}:
            return False
        if self.status == "aborted":
            return self.abort_reason in PROVIDER_ABORT_REASONS
        return bool(self.failed_gates)


def rerun_blockers(
    attempts: Sequence[AcceptanceAttempt],
    *,
    commit: str | None,
    dirty: bool | None,
    rerun_of: str | None,
    source_identity: str | None = None,
) -> list[str]:
    """Why an acceptance run may not start under the no-rerun-until-green rule.

    A release candidate — one source identity — gets one acceptance run. A
    second is allowed once, only as a documented rerun of a first that failed
    on the provider's availability alone. A change to release-relevant content
    is a new source identity, a new candidate, and starts fresh; a new commit
    that changed only run outputs is not. A dirty tree is a development run:
    it is not counted and can never be published.
    """
    if dirty is not False or commit is None:
        if rerun_of is not None:
            return ["--rerun-of applies to a clean commit; this tree is dirty or unknown"]
        return []

    earlier = [
        attempt
        for attempt in attempts
        if attempt.counts and attempt.is_candidate(commit=commit, source_identity=source_identity)
    ]
    ids = ", ".join(attempt.run_id for attempt in earlier)
    candidate = f"source {source_identity[:12]}" if source_identity else f"commit {commit[:12]}"
    if rerun_of is None:
        if not earlier:
            return []
        return [
            f"release candidate {candidate} already has an acceptance run ({ids}). A red run "
            "is documented, not repeated: a change to release-relevant content is a new "
            "candidate; a commit of run outputs alone is not. Only a run that failed on the "
            "provider's availability alone may be rerun, once, with --rerun-of <run id> and "
            "a --note naming the outlier."
        ]

    target = next((attempt for attempt in earlier if attempt.run_id == rerun_of), None)
    if target is None:
        return [
            f"--rerun-of {rerun_of}: no clean acceptance run of release candidate {candidate} "
            "has that id"
        ]
    if len(earlier) >= 2:
        return [f"release candidate {candidate} has used its one permitted rerun ({ids})"]
    if target.passed:
        return [f"--rerun-of {rerun_of}: that run passed; a passing run is not rerun"]
    if not target.provider_outlier:
        return [
            f"--rerun-of {rerun_of}: that run did not fail on provider availability alone "
            f"(failed gates: {', '.join(target.failed_gates or ()) or 'unknown'}; "
            f"abort: {target.abort_reason or 'none'}). It stays the result for this candidate."
        ]
    return []
