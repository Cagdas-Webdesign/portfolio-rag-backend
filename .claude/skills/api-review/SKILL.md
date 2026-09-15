---
name: api-review
description: Review FastAPI and REST changes in this repository — request/response schemas, Pydantic validation, status codes, versioning, OpenAPI honesty, the error contract, backwards compatibility, and leakage of internals. Use whenever a route, schema, status code, middleware or error handler changes.
---

# API review

The public API is a contract with an existing portfolio client. Review changes as contract changes.

## Scope

```bash
git diff -- src/portfolio_rag/api/
uv run python -c "import json;from portfolio_rag.main import app;print(json.dumps(app.openapi(),indent=2))" | head -100
```

## Checks

### 1. Schemas

* Request models set `extra="forbid"`. Silently ignoring a field the client believes in is worse
  than rejecting it.
* Every string field the client controls has a length bound; every numeric field has a range.
* Constraints match reality — a pattern the client cannot satisfy is a bug, not security.
* Field `description`s are present and true; they end up in the OpenAPI document.
* Domain types are reused rather than re-declared. Duplicate a model only when transport and domain
  genuinely diverge, and say why in the module docstring.

### 2. Status codes

* `200` only when something is actually returned. `201` for creation, `204` for no content.
* `422` for validation, `404`/`405` from routing, `503` for an upstream a client can retry past,
  `500` only for genuinely unexpected failures.
* "The knowledge base does not cover that" is a `200` with empty `citations`, not an error status.
* Never a `200` carrying an error in the body.

### 3. Versioning

* Public endpoints live under `/api/v1`. Operational endpoints (`/health`) are unversioned on
  purpose — do not "fix" that.
* A breaking change gets a new prefix. Breaking means: removing or renaming a field, tightening
  validation, changing a type, changing a status code, or changing the meaning of an existing value.
* Adding an optional field is not breaking. Adding a required request field is.

### 4. OpenAPI honesty

This is the check most likely to fail. The document must describe exactly what the code produces.

* No documented response the endpoint cannot return, and none it can return left undocumented.
  `POST /api/v1/chat` documents `200` with `ChatResponse`, plus `422`, `503` and `500`. It no longer
  documents `501` — nothing produces it.
* Every route has a `summary`, a `description` and a tag.
* Error responses reference `ErrorResponse`.
* `tests/integration/test_openapi.py` enforces much of this. If a change makes those tests fail,
  the honest fix is usually the code, not the test.

### 5. Error contract

* Every non-2xx response uses the `{"error": {...}}` envelope — no endpoint invents its own shape.
* The `code` is an existing `ErrorCode`. A new code needs an entry in `_STATUS_BY_CODE` (a unit test
  enforces this) and is a contract addition.
* Messages are written for clients and are stable. Never `str(exc)`, never a path, never a query,
  never a provider payload.
* `request_id` is present.

### 6. API vs. application logic

Route handlers validate, delegate, serialize. Retrieval decisions, prompt building, ranking and
persistence do not belong in `api/` — they belong in the application layer, called through ports.

### 7. Leakage

```bash
uv run pytest tests/integration/test_error_contract.py -q
```

* No exception text, stack trace, file path, dependency name or configuration value in any response.
* No internal identifier a client cannot use.
* Response headers add nothing that describes the internals.

### 8. Tests

Any HTTP behaviour change needs an integration test: the success path, the validation failure, and
the error shape.

## Output

Findings ordered by severity, each with `file:line` and the fix:

* **Contract break** — an existing client would break. Always report first.
* **Spec drift** — OpenAPI and code disagree.
* **Leak** — a response exposes something internal.
* **Improvement** — clarity, consistency, missing constraint.

Then run the API tests and report the result:

```bash
uv run pytest tests/integration -q
```
