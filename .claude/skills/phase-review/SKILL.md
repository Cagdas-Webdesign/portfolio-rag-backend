---
name: phase-review
description: Verify that a roadmap phase's Definition of Done is genuinely met before it is declared complete — including scope creep and work accidentally pulled in from later phases. Use when a phase is claimed to be finished. Outputs PASS or FAIL with reasons.
---

# Phase review

Verify, do not assume. Every claim in the verdict must rest on a file that was read or a command
that was run. "The roadmap says it is done" is not evidence.

## Procedure

### 1. Read the roadmap

[../../../docs/ROADMAP.md](../../../docs/ROADMAP.md). Identify the phase under review and quote its
Definition of Done — work against the written criteria, not a remembered version of them.

### 2. Check each criterion individually

For every item: **met / not met / partially met**, with the evidence.

| Kind of criterion | Evidence that counts |
| --- | --- |
| "X is implemented" | the file and the symbol, read — not inferred from a name |
| "tests pass" | the actual `pytest` output |
| "checks pass" | the actual `ruff` / `mypy` output |
| "the app starts" | a real startup and a real request |
| "documented" | the document read, and matching the code |
| "nothing costs money" | dependencies and configuration inspected |

Run the gate rather than trusting a prior claim:

```
skill: test-gate
```

### 3. Check the negative criteria

Most phases forbid more than they require. Verify the forbidden things are genuinely absent — an
unused stub still counts as implemented.

```bash
grep -rniE "langchain|llama_index|haystack" pyproject.toml src/
grep -rniE "mistral|openai|vectorize|workers_ai|boto3" src/ --include="*.py" | grep -v infrastructure/
git status --short          # uncommitted or unexpected files
```

### 4. Detect scope creep

Compare the code against the *later* phases' deliverables. Anything present that a later phase owns
is a finding, even if it works and even if it is good code — it was built without the context that
phase would have provided, and it now has to be maintained.

Also look for what nobody asked for: unused abstractions, configuration with no reader, packages
with no callers, endpoints not in the phase's deliverables.

### 5. Check honesty of documentation

* Does the README describe the actual state, or an aspirational one?
* Does OpenAPI document only what the code can produce?
* Are unimplemented parts clearly marked as planned?
* Any invented content — knowledge documents, benchmarks, metrics, feature claims? That is an
  automatic FAIL (see AGENTS.md §5).

## Output

```
Phase N — <name>

Definition of Done
  ✅ <criterion>          — <evidence>
  ❌ <criterion>          — <what is missing>
  ⚠️  <criterion>          — <what is partial>

Checks
  <the test-gate table>

Scope creep
  <items from later phases that were implemented, or "none">

Extras not in the phase scope
  <unused abstractions, unrequested features, or "none">

Verdict: PASS | FAIL
<two or three sentences of reasoning>
```

**Rules for the verdict**

* PASS requires *every* criterion met and *every* executed check green.
* A failing check is an automatic FAIL, whatever the reason.
* Scope creep is a FAIL when it adds unrequested behaviour or an unused abstraction; note it and
  recommend removal.
* Missing documentation is a FAIL when the phase's Definition of Done requires it.
* When it is close, it is FAIL — a phase declared done is a foundation the next one is built on.

On FAIL, end with the shortest list of concrete actions that would turn it into a PASS. Do not
perform them unless asked.
