# Release acceptance

The protocol a live end-to-end run has to follow for its export to count as a **release
acceptance**: proof that one commit, with one configuration, ran one named suite under fixed rules,
and passed. Protocol version **`release-acceptance-v1`**.

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
uv run portfolio-rag eval run --e2e --tier acceptance --rerun-of <run id> --note "<the outlier>" --output …
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

## Exit codes

| Code | Meaning for `eval run --e2e --tier acceptance` |
| --- | --- |
| `0` | `release_acceptance.verdict` is PASS |
| `4` | the run produced an artifact whose verdict is FAIL — a gate, a dirty tree, an aborted or incomplete run, missing or contradictory provenance, anything the verdict names |
| `3` | an internal defect aborted the run (a crash; the partial artifact is written) |
| `1` | the run did not start: invalid arguments, or the preflight refused it (budget, rerun rule) |

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
* the gate set is the frozen one, and the verdict is recomputed under protocol
  `release-acceptance-v1` and compared with the stored one;
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

It is **1.1.0**. `v1.0.0` and `v1.0.1` are historical tags; the state after Pakete 1–5 adds, among
others, the optional `conversation` request field, so it is a new minor release, not 1.0.1. The tag
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
4. `eval run --e2e --tier acceptance --preflight-only` — the budget forecast, no calls.
5. The live acceptance run, then `eval validate-acceptance` on its export.
6. Commit the export, the summary and the ledger line, whatever the verdict.
