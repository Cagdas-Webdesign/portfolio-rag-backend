---
name: test-gate
description: Run this project's full quality gate — Ruff lint, Ruff format check, mypy, pytest, plus a FastAPI startup, /health and OpenAPI smoke test. Use before declaring any substantial change finished. Never reports success when a check fails.
---

# Test gate

Run every check. **Do not stop at the first failure** — run them all, then report. A green gate is
the only thing that counts as "done"; a partial pass is a failure.

## 1. Environment

```bash
uv sync --frozen
```

If the lockfile is out of date, `uv sync` (without `--frozen`) updates it — that is a change to
report, not a silent fix.

## 2. Static checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

`ruff format --check` reports; `uv run ruff format .` fixes. Never silence a finding with `noqa` or
`type: ignore` to make the gate pass — fix it, or report it as unresolved with the reason.

## 3. Tests

```bash
uv run pytest -q
```

All tests must pass. Skips are only acceptable when the skip reason is deliberate and stated.

## 4. Smoke test

Static checks and `TestClient` do not prove the process actually starts under a real server.

```bash
uv run uvicorn --app-dir src portfolio_rag.main:app --port 8123 &
sleep 2
curl -fsS http://127.0.0.1:8123/health
curl -fsS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8123/openapi.json
curl -fsS -i -X POST http://127.0.0.1:8123/api/v1/chat \
  -H 'Content-Type: application/json' -d '{"message":"smoke test"}' | head -1
kill %1
```

Verify:

* `/health` returns `{"status":"ok", ...}` with the current version
* `/openapi.json` returns `200` and parses
* the chat endpoint returns `200` with an `answer` and a `citations` array
* an `X-Request-ID` header is present on the responses
* startup logs contain no traceback and no warning that indicates a real problem

## 5. Docker (when the change touches the Dockerfile, dependencies, or the start command)

```bash
docker build -t portfolio-rag-assistant:gate .
docker run -d --name rag-gate -p 8124:8000 portfolio-rag-assistant:gate
sleep 3 && curl -fsS http://127.0.0.1:8124/health
docker rm -f rag-gate
```

If Docker is unavailable, say so explicitly. Do not install it, and do not report the step as
passed.

## Report

A short table — check, result, and for failures the compact reason:

| Check | Result |
| --- | --- |
| ruff check | ✅ / ❌ |
| ruff format --check | ✅ / ❌ |
| mypy | ✅ / ❌ |
| pytest | ✅ N passed / ❌ N failed |
| startup + /health + OpenAPI | ✅ / ❌ |
| docker (if run) | ✅ / ❌ / skipped |

For each failure: the file and line, one line on the cause, and the fix. Group identical failures
instead of pasting the whole output.

**The verdict is PASS only when every executed check passed.** Otherwise FAIL — say which checks
failed and stop. Do not describe work as complete on a failing gate.
