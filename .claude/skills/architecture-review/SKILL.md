---
name: architecture-review
description: Review a change against this repository's architecture — module boundaries, dependency direction, provider lock-in, unnecessary abstraction, ADR compliance. Use after adding a module, moving code between layers, introducing a dependency, or any change that alters structure rather than behaviour. Analyses and reports; does not refactor.
---

# Architecture review

**Report first, do not refactor.** The output is a findings list. Apply fixes only when asked, and
then one at a time.

Read [../../../docs/ARCHITECTURE.md](../../../docs/ARCHITECTURE.md) and the ADRs in
[../../../docs/adr/](../../../docs/adr/) before judging anything. The layer table and the ADRs are
the standard — not personal preference about what good architecture looks like.

## Scope

```bash
git diff --stat            # or `git status` if nothing is committed yet
git diff -- src/
```

Review the change, not the whole repository. Note pre-existing problems separately from ones the
change introduces.

## Checks

### 1. Dependency direction

`api → application → domain ← ports ← infrastructure`, with `core` beneath everything.

```bash
grep -rn "^from portfolio_rag\|^import portfolio_rag" src/portfolio_rag/domain/ src/portfolio_rag/ports/ src/portfolio_rag/core/
```

* `core` must import no other layer.
* `domain` may import `core` only — no FastAPI, no Starlette, no provider SDK.
* `ports` may import `core` and `domain` only.
* Nothing outside `main.py` may import `infrastructure`.
* `api` must not be imported by anything except `main.py`.

### 2. Provider lock-in (ADR 0002)

```bash
grep -rniE "mistral|openai|anthropic|cohere|vectorize|workers_ai|cloudflare|boto3|d1|r2" src/ --include="*.py" | grep -v "^src/portfolio_rag/infrastructure/"
```

Any hit outside `infrastructure/` (and outside a comment or docstring explaining the plan) is a
finding. Application code must talk to ports.

### 3. Cloudflare in the core (ADR 0003)

The application must still run under plain Uvicorn and in Docker. Platform bindings, edge-runtime
globals or "this only works when deployed" assumptions below `main.py` are findings.

### 4. Abstraction without a caller

For every new protocol, base class, factory, registry, wrapper or config field, ask: **who calls
it today?** If the answer is "the next phase", it is premature — the roadmap phase that needs it
will define it better. Check for:

* packages containing only `__init__.py`
* interfaces with exactly one implementation and no second one in sight
* indirection that only forwards
* configuration nobody reads

### 5. Dependencies

```bash
git diff -- pyproject.toml uv.lock
```

* Is a new dependency justified in the change, with a reason the standard library cannot cover?
* Is it a RAG framework (LangChain, LlamaIndex, Haystack)? That is a hard no — see AGENTS.md §3.
* Does it cost money, or require an account or a credit card? Hard no — the budget is €0.
* Is it in the runtime dependencies when it is only needed for development?

### 6. Roadmap discipline

Compare against [../../../docs/ROADMAP.md](../../../docs/ROADMAP.md): does the change implement
something a later phase owns? Early work is a finding even when the code is good.

### 7. Simplification

For the changed code specifically: what would the same behaviour look like with one less layer, one
less parameter, one less concept? Propose the concrete simplification, not "consider simplifying".

### 8. ADR compliance

Does the change contradict an accepted ADR? Then either the change is wrong, or the ADR needs
superseding — say which, and why. Does it make a decision that is expensive to reverse and is not
yet recorded? Then it needs a new ADR.

## Output

Group findings by severity, most severe first. For each: what it is, where (`file:line`), which
rule or ADR it breaks, and the concrete fix.

* **Violation** — breaks a documented boundary, an ADR, or the cost constraint.
* **Risk** — allowed today, but will be expensive later. Say what makes it expensive.
* **Simplification** — same behaviour, less structure.

End with one line: does this change leave the architecture in a state the next phase can build on?
If there are no findings, say so plainly and stop.
