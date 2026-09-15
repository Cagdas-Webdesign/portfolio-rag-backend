---
name: security-review
description: Lightweight security review for this repository — secrets, environment handling, CORS, input validation, logging of sensitive data, error leakage, injection and abuse risk, unsafe defaults. Use when input handling, logging, configuration, CORS or error paths change, and before declaring a phase complete.
---

# Security review

Proportionate to what this system is: a public read-only knowledge assistant with no accounts, no
payments and no personal data store. **Do not invent enterprise requirements.** It does not need
OAuth, an audit trail or a WAF, and recommending them is noise, not diligence.

The real risks here are: leaked credentials, leaked internals in errors, user content in logs, an
over-permissive CORS policy, unbounded input, and — once RAG exists — prompt injection.

## Checks

### 1. Secrets

```bash
grep -rniE "api[_-]?key|secret|token|password|bearer |sk-[a-z0-9]" --include="*.py" --include="*.toml" --include="*.yml" --include="*.yaml" --include="*.md" --include="*.example" . | grep -v "\.venv/"
git status --short   # is a real .env about to be committed?
```

* No credential in code, tests, fixtures, comments, CI, or `.env.example`.
* `.env` is git-ignored; `.env.example` contains names and harmless placeholders only.
* No secret in the Docker image — everything comes from the runtime environment.
* Phase 1 has no credentials at all. A new one appearing is itself worth questioning.

### 2. Configuration

* Settings come from `core/config.py`; nothing reads `os.environ` directly.
* Defaults are safe when a variable is unset — a missing setting must not silently widen access.
* Production behaviour does not depend on a debug flag being off.

### 3. CORS

```bash
grep -rn "CORSMiddleware" -A 10 src/portfolio_rag/main.py
```

* No `*` in origins (configuration rejects it — verify the check still exists).
* `allow_credentials` stays `False` while there is no cookie or credential-based auth. `True`
  together with a broad origin list is the classic mistake.
* Methods and headers are enumerated, not wildcards.

### 4. Input validation

* Every client-controlled string is length-bounded and, where meaningful, pattern-bounded.
* Request models reject unknown fields.
* Anything echoed anywhere (headers, ids) is validated first — see `sanitize_request_id`, which
  exists to stop a forged `X-Request-ID` from injecting a line into the logs.

### 5. Logging

```bash
grep -rn "logger\.\|logging\." src/ --include="*.py"
```

* No chat message, prompt, retrieved passage, request body or query string is logged — not at
  `DEBUG` either.
* Logged fields are identifiers, counts, sizes, durations, status codes.
* No credential or configuration value reaches a log line.

### 6. Error leakage

```bash
uv run pytest tests/integration/test_error_contract.py -q
```

* No stack trace, exception text, path, SQL, or provider payload in a response body.
* The generic `500` message is genuinely generic.
* Response messages do not differ in a way that reveals whether something exists internally.

### 7. Injection and abuse

* Untrusted input never reaches a shell, a filesystem path, or a query without validation.
* Once retrieval exists: knowledge content is *also* untrusted input to the model — a document
  containing instructions must not be able to steer it. Flag any prompt assembly that concatenates
  retrieved text without delimitation.
* `visibility: internal` must be enforced at retrieval, not by prompt wording.

### 8. Unsafe defaults

* No debug mode, no verbose errors, no permissive fallback that is only safe locally.
* Docker runs as a non-root user.
* Dependencies are pinned by the lockfile; CI installs with `--frozen`.

## Documented for later, not now

These are Phase 6 (see `docs/ROADMAP.md`). Note them as *deferred*, not as findings, unless the
change being reviewed actually creates the exposure:

rate limiting · abuse prevention · adversarial prompt-injection testing · request size limits at
the edge · security headers · authentication for privileged operations.

What is *not* deferred, because Phase 5 built it: `visibility: internal` enforced structurally in
the retrieval query, knowledge kept in the user role and never the system role, and citations built
only from passages the backend retrieved. A regression in any of those is a finding, not a Phase 6
item.

## Output

Findings ordered by severity. Say what an attacker actually achieves — a severity without an
exploitation path is theatre.

* **Critical** — exposed secret, or unauthenticated access to something that must be protected.
* **High** — leaks internals or user content, or lets untrusted input reach somewhere dangerous.
* **Medium** — a weak default that becomes a real problem in a plausible next step.
* **Low** — hardening worth doing when the code is touched anyway.
* **Deferred** — a Phase 6 item, listed for completeness.

If nothing is found, say so. A clean review of a small, boring surface is the expected outcome.
