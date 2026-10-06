# Release acceptance

The protocol a live end-to-end run has to follow for its export to count as a **release
acceptance**: proof that one commit, with one configuration, ran one named suite under fixed rules,
and passed. Protocol version **`release-acceptance-v2`** (artifacts written under `v1` are still
judged by `v1`; `v2` adds one condition, [execution segments](#checkpoint-and-resume)).

This adds no reliability behaviour. Retrieval, prompt, grounding, citations, failure policy, retries,
deadline, provider, model and budget logic are what they were. It records what a run was, decides
the verdict by rule, and makes the decision checkable afterwards.

Code: `src/portfolio_rag/evaluation/acceptance.py`. The artifact is the existing `e2e-eval-v3`
export, extended — there is no second report format.

## What an artifact proves, and what it does not

The run executes the recorded commit's **source in a local process** against the configured
providers and vector store. It does **not** exercise a container image or the Cloud Run deployment,
and the artifact says so in its own fields (`release_acceptance.scope`: `tested: source_commit`,
`deployed_image_tested: false`). An image is covered by the result only if it was built from the
recorded commit — see [Linking a build to the commit](#linking-a-build-to-the-commit).

## Release candidate identity

| Field | Source |
| --- | --- |
| `run.project_version` | the software version, `portfolio_rag.__version__` — see [Version and tag](#version-and-tag) |
| `run.git_revision` | Git provenance: `git rev-parse HEAD`, taken **once, before the first question** |
| `run.release_source_identity` | the semantic release content: SHA-256 over the release-relevant files — the release candidate itself |
| `run.git_dirty` | `git status --porcelain` from the repository root, untracked files included |
| `run.git_dirty_excludes` | `["evaluation/results"]` — where runs write their own outputs |
| `run.git_tag` | an optional Git release label: `git describe --tags --exact-match HEAD`, or `null` |
| `run.run_id` | the run's correlation id, also on every log line of the run |
| `run.generated_at` | UTC timestamp |
| `format` | `e2e-eval-v3` — the export schema |

`git_revision` says *where in history* the run started. `release_source_identity` says *what*
ran, and it is what makes two runs the same release candidate. A dirty tree is never hidden:
`git_dirty: true` is written as such, and an unknown state is `null`, which is not clean either.

`evaluation/results/` is excluded from the dirty check because it holds the outputs of runs
(exports, summaries, the ledger), never inputs to one. Without the exclusion, a run's own artifact
would make the next run on the same commit dirty, and the rerun rule below could not be applied.
The exclusion is recorded in every export so it is never implicit.

### Release source identity

The SHA-256 over each normalized relative (POSIX) path and the SHA-256 of its content, sorted by
path, over every file git tracks or would track (untracked, not ignored) under:

| Included | Why |
| --- | --- |
| `src/` | the code that runs |
| `knowledge/` | the corpus answers are resolved against |
| `evaluation/` | datasets, suite files, the fixture corpus |
| `pyproject.toml`, `uv.lock`, `.python-version` | what is installed and how it runs |

Excluded inside them: `evaluation/results/` — exports, summaries, the ledger. Outside the list, and
so never part of it: `docs/`, `tests/`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `.github/`,
`.claude/`, `edge/`, `Dockerfile`, `.dockerignore`, `.env.example`, `.gitignore` — none of them can
change what an acceptance run does. A test fails if a new top-level entry appears without being
placed on one side.

Content is read from the working tree; on a clean tree that is the commit's content. The identity is
deterministic: the same files give the same value on any machine and under any commit. So a commit
that only adds results, documentation or tests is **the same release candidate under a new SHA** —
not a new one. Configuration that comes from the environment is recorded in the artifact but is not
part of the identity: a configuration change meant for a release is committed (a default in `src/`,
a dependency in `uv.lock`), which makes it a new candidate.

## Suite identity

`run.suite`, `run.suite_version`, `run.suite_sha256`, `run.suite_path`, `run.question_count`, plus
the dataset's own `run.dataset.{path, version, sha256, question_count}`.

* A suite other than `full` is an id file beside the dataset (`<dataset>.<suite>.yaml`, with its own
  `version`). Its version and the SHA-256 of that file name exactly one selection.
* `full` is the dataset itself and carries the dataset's version and hash.

The acceptance tier runs **`release-acceptance`, version 1: 24 questions**
(`evaluation/portfolio-questions.release-acceptance.yaml`), selected for risk coverage and
documented question by question in `evaluation/README.md`. The 49-question dataset (`full`) remains
**extended validation** and is never a tier's suite. Changing an id is a new suite version — a new
release candidate, decided before a run. The artifact always names the suite that actually ran (for
a tiered run, the tier's suite, never the `--suite` default).

## Configuration provenance

Named fields only — never settings wholesale, never a credential, account id, index name or endpoint,
never a prompt text where its version identifies it.

| Concern | Fields |
| --- | --- |
| Generation | `run.generation.{provider, model, prompt_version, grounding_check_version, response_format, grounding_check_response_format, max_prompt_tokens, output_reserve_tokens, max_output_tokens, generation_attempt_limit, transport_attempts, delay_seconds}` |
| Embedding | `run.embedding.{provider, model, dimensions, representation_version, identity}`, `run.query_representation_version` |
| Retrieval | `run.retrieval_policy.{top_k, min_similarity, visibility}`, `run.vector_store` |
| Corpus | `run.corpus.{documents, chunks, sha256}` |
| Mode | `run.tier`, `run.selected_question_ids`, `run.rerun_of`, `run.notes` |
| Operations | `operations` — forecast, estimated neurons, usage coverage, budget zone, ledger consumption |

`visibility` is not a setting. It is the constant the retrieval service builds into every query
(`PUBLIC_VISIBILITY_FILTER`), recorded so the artifact states it rather than implies it.

## The frozen gates

| Gate | PASS when |
| --- | --- |
| `no_pipeline_errors` | no question raised or came back empty |
| `citations_verified` | no answer without a citation, no unverified citation, no invented label, no dangling mark |
| `unanswerable_refused` | every unknown/internal question got a controlled refusal with no citation |
| `nothing_internal_published` | no internal label in an answer, no internal passage retrieved |
| `internal_leaks` | **= 0** — questions with either of the two failures above |

The gate-to-failure mapping is `GATES` in `evaluation/e2e.py`; a test pins it. The gates are
derived from each question's recorded failures, not stored opinions. Changing a gate is a new
protocol version, decided before a run — never a re-reading of a result after it. No pass mark is
attached to any measurement (hit rates, answered share): those stay numbers.

## The safety freeze

The invariants below are frozen for the release. Paket 5 implements none of them; each is held by
existing tests, which `SAFETY_INVARIANTS` names by node id, and a test fails if any named test
disappears. Every artifact lists the invariants it was produced under
(`release_acceptance.safety_invariants`).

| Invariant | Held by |
| --- | --- |
| `strict_generation_parser` | `test_generation_contract.py`, `test_refusal_contract.py` |
| `citation_validation` | `test_answer_service.py` |
| `grounding_validation` | `test_grounding_check.py`, `test_refusal_contract.py` |
| `fail_closed_on_technical_failure` | `test_grounding_check.py`, `test_structured_output_reliability.py` |
| `controlled_refusal` | `test_answer_service.py` |
| `public_only_retrieval` | `test_retrieval_service.py` |
| `conversation_is_not_evidence` | `test_conversation_context.py` |
| `recovery_passes_safety_again` | `test_generation_regeneration.py` |
| `telemetry_stays_internal` | `test_chat.py`, `test_provider_call_telemetry.py` |

## When a run counts

`release_acceptance.verdict` is **PASS** only when every condition holds; otherwise **FAIL**, with
each broken rule named in `release_acceptance.reasons`. There is no partial pass.

| Condition | Meaning |
| --- | --- |
| `acceptance_tier` | `--tier acceptance` |
| `whole_suite` | not narrowed by `--question-id` |
| `run_complete`, `run_not_aborted` | the run reached its last question; no defect, guard or provider stop ended it |
| `every_question_recorded` | one record per question in the suite |
| `provenance_complete` | every field above present; `git_revision` a full 40-character SHA |
| `clean_tree` | `git_dirty` is `false` |
| `gates_pass` | the four gates PASS |
| `no_internal_leaks` | `internal_leaks` is `0` |
| `execution_segments_consistent` | (`v2`) the run records its execution segments, and they agree with the artifact — see below |
| `process_interruptions_within_limit` | (`v2`) no question was cut short by a process interruption more than once — see below |

A run with a dirty tree is a **development run**: allowed, fully written, readable — and never a
release acceptance.

## No rerun until green

A red acceptance run is the result for its commit. It is kept and documented — never deleted and
re-run until a green one can be shown.

* A change to release-relevant content is a **new source identity, a new release candidate**, and
  gets its own run. A commit of results, documentation or tests alone is not.
* The preflight enforces this from the ledger, by source identity: a candidate that already has a
  clean acceptance run is refused a second one, whatever its commit SHA. (Ledger lines written before
  the identity was recorded are matched by commit.)

### The provider-outlier rule

The one exception, narrow on purpose: when a run failed **only on provider availability** — its only
failed gate is `no_pipeline_errors`, or it was stopped by `rate_limited` or
`systemic_provider_failure` — and nothing was changed, it may be repeated **once**:

```bash
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml --e2e --tier acceptance --rerun-of <run id> --note "<the outlier>" --output …
```

* `--rerun-of` must name a clean acceptance run of the **same release candidate**, and a `--note`
  must say what the outlier was.
* A run that broke a safety gate, was aborted by a defect or the budget guard, or passed, is not
  rerun.
* A candidate that has used its rerun gets no third run.
* Both run ids stay traceable: the rerun records `rerun_of`, and both runs are in the ledger and in
  `evaluation/results/`.

This does not soften `no_pipeline_errors`: the first run stays red, and the rerun is judged by
exactly the same gates.

**Limit of the guard.** It reads the local ledger (`evaluation/results/provider-ledger.jsonl`), which
a person can edit. That is why the ledger and every artifact are committed: every attempt stays
visible in history.

## Checkpoint and resume

An acceptance run is one **logical run** of one or more **execution segments**. Code:
`src/portfolio_rag/evaluation/checkpoint.py`, `src/portfolio_rag/evaluation/canary.py`.

**Canary.** Before its first question a run (and every resumed segment) sends one request through
the generation port: fixed text asking for `{"ok":true}` — read with the answer contract's own
tolerance, plain or in one Markdown fence, nothing else — an output cap of 512 tokens (a
lightweight health-check cap, not live-validated and not proof of generation capacity), no
retrieval, no grounding check, no portfolio question, no retry above the adapter's transport policy.
Anything but a clean `{"ok":true}` — 429, 401/403, 5xx, timeout or network, a malformed or truncated
reply — and the run does not start (exit `1`). `canary_status`, `canary_status_code`,
`canary_failure_category` and `retry_after_seconds` are printed; on PASS they are in the export
(`operations.canary`, and per segment). The canary has its own ledger line (tier `canary`): its
spend counts today, and it is no acceptance attempt. Its cost bound is part of the preflight
forecast. `--no-canary` skips it, recorded as `skipped`.

**Checkpoint.** After every completed question the checkpoint
(`<output>.checkpoint.json`, or `--checkpoint`) is replaced atomically — temporary file, `fsync`,
rename. A crash loses at most the question being asked. It holds the identity, progress, every
completed question's results (outcome, answer, citations, retrieval as chunk ids and similarities,
provider-call telemetry, usage), the segments with their canary, budget and pacing, rate-limit
events — and no passage text, prompt or credential. A SHA-256 over its content refuses an edited
file.

**Pause.** A 429, refused credentials, a systemic provider failure or the budget guard *pauses* the
run: the checkpoint is written, the run stays `complete: false` with no release verdict, and the CLI
prints the exact resume command. The question(s) the stop cut short — the one that got the 429, the
streak of provider failures — are **not completed**: they are no question's failure, they are
listed as interrupted, and the resume asks them again. An internal defect still aborts hard and is
never resumed: its fix is a new candidate.

**Resume.** `--resume-from <checkpoint>` continues the logical run only when commit, tag, version,
release source identity, dataset, suite, the ordered question ids, corpus, embedding, vector store,
retrieval policy, provider, model, prompt and grounding-check versions, response formats, output
limits and the attempt/recovery policy are identical (`IDENTITY_FIELDS`). Any difference is
refused before anything is requested. Completed questions are not asked again — no embedding,
search, generation or check. Pacing may differ between segments and is recorded per segment. A
checkpoint left `running` by a crash is closed as `process_interrupted`, its known spend entered in
the ledger once. The segments of one logical run are **one attempt** under the rerun rule.

**Budget on resume.** Today's known consumption is the ledger's, earlier segments included; the
forecast covers only the remaining questions plus the canary. The hard cap (known today + forecast
≤ nominal daily budget), the excessive band and the reserve override are unchanged.

**The final artifact** is the whole logical run: every question exactly once, in suite order,
gates computed over all of it. `run.logical_run_id` (= `run.run_id`), `original_started_at`,
`completed_at`, `resume_count`, `execution_segments` and `rate_limit_events` describe its execution;
`operations.observed` covers all segments. The validator checks, from the file alone: every question
exactly once and in order, every segment run under the identity the artifact states, every segment
but the last paused by an external stop, and the segments' calls and estimates adding up.

**Pacing.** An acceptance run waits at least 2 s between the starts of two generation-provider calls
(generation and grounding check alike) unless `--generation-delay-seconds` says otherwise; a
`Retry-After` the provider sends is taken, capped at the pacer's longest wait (120 s). Sequential
always. This guards against bursts only; no pacing helps against a daily allocation.

**Timeouts and the recovery path.** A reasoning model's reply is as long as its output cap allows,
so a request's timeout follows its cap: `max(provider_timeout_seconds, cap / 30 tokens/s)` (Workers
AI adapter, `MIN_OUTPUT_TOKENS_PER_SECOND`; the slowest call on record ran at 38.7 tokens/s). The
ordinary 800-token request keeps its 30 s; the 1500-token recovery gets 50 s. An evaluation question
is held to `evaluation_question_deadline` — retrieval allowance plus every step the failure policy
allows at its own timeout: 30 + (30 + 50) + (30 + 50) = 190 s for Workers AI — instead of the 60 s
production deadline, which cannot hold a generation recovery and a check recovery. A paced run has
no question deadline and is bounded by its attempts. Attempt counts are unchanged.

**Retrieval interruptions.** A query embedding or vector search that fails *transiently* — a
timeout, a network failure, a rate limit, a 5xx — pauses the run like a 429 does
(`retrieval_unavailable`): the question is not finalized, the checkpoint keeps every completed one,
and the resume asks it again. Embedding failures are transient when the adapter, having retried,
judged them retryable; vector-search failures are judged at the search, which is a read
(`rag.errors.vector_store_failure`). A failure asking again would not change — refused credentials,
a malformed answer, a vector of the wrong size, a chunk-resolution defect — stays a hard pipeline
error, recorded per question as `retrieval_error` (stage, transient, detail, status).

**Process interruptions.** A crash or a closed terminal leaves the segment `running`; the resume
closes it as `process_interrupted`, names the question it was asking, and asks it again. That stays
in the artifact (segment, question, the summary's *Process interruptions* row), and the validator
checks every interrupted question is asked again later and never was before. **Governance limit:**
one process interruption per question (`PROCESS_INTERRUPTION_LIMIT_PER_QUESTION`). A second one on
the same question lets the run finish but fails `process_interruptions_within_limit` — re-asking a
question until it looks right is not a release acceptance. Provider pauses (429, retrieval,
budget) do not count against it.

**Attempts on the wire.** Every provider call records `http_attempts` (every request at every
layer), `transport_attempts` (the adapter's, last round), `pacing_attempts` (the pacer's rounds),
`retry_after_seconds` (the longest wait asked for) and `terminal_status_code`; the provider-call
metrics sum `http_attempts`. These are observed requests, not billed usage: the neuron estimate is
still computed from reported tokens per logical call, and a failed request's cost is not invented.

**One process, the latest state.** A run claims its checkpoint with a lock file beside it
(`<checkpoint>.lock`: logical run, process id, a hash of the host name, time, path — no secret) before
the checkpoint is read or written, and releases it however the run ends. A second resume of the same
checkpoint is refused before anything is requested. A lock whose process provably no longer exists
on this host is replaced; any other only with `--break-lock`; either replacement is recorded in the
segment (`lock`). Time alone never frees a lock.

**One process per logical run, whichever copy.** The checkpoint lock guards one file; two copies of
a checkpoint at two paths would be two files. So a run also claims its *logical run id*, at
`locks/acceptance-<logical run id>.lock` beside the ledger — a new run before its first request, a
resume as soon as the checkpoint is read, before the staleness check and before a crashed segment
is settled. The claim is an operating-system lock (`flock`) held for the life of the process:
released when it ends, a crash or a `kill -9` included, so it never has to be judged stale, and a
lock that is held is held by a running process — `--break-lock` does not apply to it. The file
keeps who last claimed it (logical run, process id, host hash, time, checkpoint path) and no
secret. Limits: copies continued against different `--ledger` files, or on different machines or
on a network filesystem whose locks do not reach across hosts, do not meet at one lock; the same
is true of the staleness check, which reads that ledger. POSIX only (`fcntl`).

A resume is also refused (`STALE_CHECKPOINT`) when
the ledger knows a later segment of the logical run than the checkpoint does, or a segment the
checkpoint holds as still running or as another run: an older copy would ask again what a later
segment asked. A crashed segment has no ledger line yet and resumes as before.

**Truncation without a finish reason.** A rejected reply counts as cut off at the limit — and gets
the larger recovery cap — when the provider says `length`, or, when it says nothing about why it
stopped, when the reply is a JSON object that never closes (an object or array still open, or a
string unterminated, at its end). A complete but invalid object, prose, or the wrong shape does not
qualify; a reply that says it finished is believed. Attempt counts are unchanged.

**Valid but incomplete.** The validator reports a state: `VALID_COMPLETE`, `VALID_INCOMPLETE` — a
paused or stopped checkpointed run whose recorded questions are exactly the first questions of its
planned suite (`run.planned_question_ids`) — or `INVALID`. An incomplete run is never PASS.

**429 diagnostics.** A Workers AI 429 keeps its status, `Retry-After` and the numeric code from
Cloudflare's error envelope (`provider_error_code`) — never the message beside it. The two codes
Cloudflare documents for 429 are named (`rate_limit_kind`): 3036 `daily_free_allocation_exhausted`,
3040 `capacity_exceeded`. Any other code is recorded and left unnamed; `failure_detail` stays
`rate_limited`, and a 429 pauses the run as before.

Every tiered run names its dataset: the tier suites belong to
`evaluation/portfolio-questions.yaml`, and `--tier` without `--dataset` is refused before anything is
requested. A resume must name the same path — it is part of the run's identity.

```bash
# a new acceptance run (canary, checkpoint, pacing on by default)
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml --e2e --tier acceptance \
  --output evaluation/results/release-acceptance-vX.Y.Z.json \
  --summary evaluation/results/release-acceptance-vX.Y.Z.md
# continue it after a pause — the paused run prints this command
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml --e2e --tier acceptance \
  --resume-from evaluation/results/release-acceptance-vX.Y.Z.checkpoint.json \
  --output evaluation/results/release-acceptance-vX.Y.Z.segment-2.json \
  --summary evaluation/results/release-acceptance-vX.Y.Z.segment-2.md
# the recovery smoke: the two questions that hit the limit, on the acceptance's pacing path
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml --e2e --tier smoke \
  --question-id multi-marketing-automation-web --question-id broad-project-scope \
  --generation-delay-seconds 2 \
  --output evaluation/results/recovery-smoke-vX.Y.Z.json \
  --summary evaluation/results/recovery-smoke-vX.Y.Z.md
# every export reports `metrics.recovery`: RECOVERY_LIVE_VALIDATED is true only when a recovery
# asked for the larger cap and its reply was used. A smoke can pass without one — the model decides
# whether it stops at the limit; false means "not observed", not "failed". Never a gate.
# judge the final export
uv run portfolio-rag eval validate-acceptance evaluation/results/release-acceptance-vX.Y.Z.segment-2.json
```

## Exit codes

| Code | Meaning for `eval run --e2e --tier acceptance` |
| --- | --- |
| `0` | `release_acceptance.verdict` is PASS |
| `4` | the run produced an artifact whose verdict is FAIL — a gate, a dirty tree, an aborted, paused or incomplete run, missing or contradictory provenance, anything the verdict names |
| `3` | an internal defect aborted the run (a crash; the partial artifact is written) |
| `1` | the run did not start: invalid arguments, the preflight refused it (budget, rerun rule), the canary did not pass, or a resume was refused |

Only the acceptance tier has the hard exit. Smoke and local runs keep their codes: a smoke run is a
health check and exits by its gates and its smoke verdict.

## Before publishing an artifact

```bash
uv run portfolio-rag eval validate-acceptance evaluation/results/<run>.json
```

Reads one file and calls nothing. It does not trust the stored verdict:

* the format is `e2e-eval-v3` and every section is present;
* the gates are recomputed from the questions' recorded failures, and the outcome counts, the
  pipeline-error count, the failure index, `passed` and `complete` are checked against the records;
* the gate set is the frozen one, and the verdict is recomputed under
  the protocol the artifact was written under (`release-acceptance-v1` or `-v2`) and compared with
  the stored one; the execution segments of a checkpointed run are checked as described above;
* nothing that must not be published is in the file: no field named like a credential, prompt,
  message list, conversation or passage; no credential-shaped value (bearer tokens, authorization
  headers, private keys); no credential configured in the environment (compared, never printed);
  no internal context label in any answer.

Without `--development`, only a consistent artifact with verdict **PASS** is accepted — that is the
check before publication. With `--development`, any consistent artifact is accepted and its verdict
and reasons are printed: a dirty-tree run stays readable, it just is not a release acceptance.

## Version and tag

`portfolio_rag.__version__` in `src/portfolio_rag/__init__.py` is the project's only static
version: the package build reads it (`[tool.hatch.version]`, `version` is `dynamic` in
`pyproject.toml`), `/health` and OpenAPI default to it, and every export records it as
`run.project_version`. A test asserts there is no second one.

It is **1.2.0**. `v1.0.0`, `v1.0.1`, `v1.1.0`, `v1.1.1` and `v1.1.2` are historical tags. 1.1.0 added,
among others, the optional `conversation` request field; 1.1.2 the output-limit recovery and the
reserve override. 1.2.0 adds resumable acceptance runs (checkpoint, logical-run lock, protocol
`release-acceptance-v2`), the provider canary, output-cap-aware request timeouts, 429 codes and
retrieval pause/resume, so it is a new minor release, not 1.1.3. The tag
is set by hand, as `v` + the version, on the commit that passed — nothing here creates, moves or
pushes one. The tag is optional: an untagged commit is a valid candidate (`git_tag: null`), but a
tag naming another version fails provenance, as does a `project_version` that is not
`MAJOR.MINOR.PATCH`.

An installed environment carries the version it was installed with: after the version changes,
`uv sync` refreshes the package metadata (`importlib.metadata`) to match.

## Linking a build to the commit

No CI/CD is added. The link is the commit: build the image from the same, clean checkout and name it
by that commit, so the image, the Cloud Run revision deployed from it and the acceptance artifact
point at one SHA:

```bash
docker build -t portfolio-rag-assistant:"$(git rev-parse --short=12 HEAD)" \
  --label org.opencontainers.image.revision="$(git rev-parse HEAD)" .
```

The artifact still says `deployed_image_tested: false`, because it is true: the deployed image is
covered by the same source, not by the run.

## The sequence for the final run

1. Review the release-acceptance suite (`release-acceptance` v1; see [Suite identity](#suite-identity)).
2. Run the local gate — Ruff, format, mypy, pytest, `knowledge validate`.
3. Commit; the tree must be clean (`git status` shows nothing outside `evaluation/results/`).
4. `eval run --dataset evaluation/portfolio-questions.yaml --e2e --tier acceptance --preflight-only` — the budget forecast, no calls.
5. The live acceptance run, then `eval validate-acceptance` on its export.
6. Commit the export, the summary and the ledger line, whatever the verdict.
