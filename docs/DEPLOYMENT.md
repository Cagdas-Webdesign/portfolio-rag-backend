# Deployment

This document does two jobs, and they are kept apart on purpose:

* **Current state** — what is deployed and running today, at v1.0.0.
* **The procedure** — steps 0–8, reproducible for the next release or for rebuilding the deployment
  from scratch.

Every environment variable named below is real — copied from `portfolio_rag.core.config.Settings`,
not invented for the document. No value shown is a credential.

## Current state — v1.0.0

| | |
| --- | --- |
| Release | v1.0.0, the frozen production baseline |
| Origin | Docker container on Google Cloud Run |
| Front door | Cloudflare Worker edge gateway ([`../edge/`](../edge/README.md)) |
| Generation | Cloudflare Workers AI — `@cf/openai/gpt-oss-120b` |
| Embeddings | Mistral — `mistral-embed` |
| Vector store | Cloudflare Vectorize |
| Corpus | `knowledge/`, indexed into the production index |
| Retrieval | evaluated against the real corpus on the production path; `top_k` 7 (from 5, measured 2026-10-04 — see `evaluation/README.md`), `min_similarity` 0.250 |
| Verification | the [step 7](#7-verify) smoke test completed successfully — `/health`, a grounded answer with citations, and the controlled refusal path |

Account ids, index names, tokens, secrets and service URLs are deployment configuration and are
deliberately absent from this repository. They live in the provider dashboards, in Worker secrets and
in Google Secret Manager.

The retrieval row above comes from step 4, run on this exact path with
`evaluation/portfolio-questions.yaml`. It is a measured configuration rather than a guess — not a
claim that it is optimal: a corpus change, a model change or a re-chunk invalidates it, and the
dataset is re-run rather than assumed ([../evaluation/README.md](../evaluation/README.md)).

## The procedure

```
Portfolio front end  ──HTTPS──▶  edge (TLS + rate limit)  ──▶  FastAPI container
                                                                     │
                              ┌──────────────────────────────────────┤
                              ▼                                      ▼
                    Mistral embeddings                     Cloudflare Vectorize
                    (the question → a vector)              (public-only search)
                              │                                      │
                              └──────────────┬───────────────────────┘
                                             ▼
                                     bounded context
                                             ▼
                              Cloudflare Workers AI chat
                                  @cf/openai/gpt-oss-120b
                                             ▼
                              backend-verified citations → JSON
```

---

## 0. Before anything: validate the knowledge base

**`knowledge/` contains the authorized portfolio corpus.** Its format is documented in
[../knowledge/README.md](../knowledge/README.md). Validate it before deployment with:

```bash
uv run portfolio-rag knowledge validate
uv run portfolio-rag knowledge chunks --all
```

Set `visibility: public` on everything the assistant may answer from, and `internal` on everything
it may not. That field is the security boundary — see [SECURITY.md](SECURITY.md).

## 1. Provider accounts

Two, both on free tiers. Neither is created by this repository.

* **Mistral** — an API key, for `mistral-embed`. The chat adapter is still supported and still
  configurable, but it is not what this deployment generates with.
* **Cloudflare** — an account id, and **two separate tokens** on it: one scoped to Vectorize, one
  scoped to Workers AI. Sharing one token across both products would let a leak of either spend the
  other. The Vectorize index must **already exist**; nothing here provisions cloud resources.

## 2. Create the Vectorize index to match the embedding space

The index must be created with dimensions and a metric that match what the embedding provider
produces, because they cannot be changed afterwards and a mismatch is refused at the first write:

| Property | Value | Where it comes from |
| --- | --- | --- |
| Dimensions | **1024** | `mistral-embed`, in `infrastructure/embedding/mistral.py` |
| Metric | **cosine** | `SimilarityMetric.COSINE`, the only one implemented |

An index built at the wrong dimensionality is not repairable — it has to be recreated, and the
corpus re-indexed.

## 3. Index the corpus with production embeddings

Once, from a terminal, with the production settings in the environment. Not at deploy time and not
from the server: indexing is offline work, and the container never does it against a remote index.

```bash
export PORTFOLIO_RAG_EMBEDDING_PROVIDER=mistral
export PORTFOLIO_RAG_MISTRAL_API_KEY=…
export PORTFOLIO_RAG_VECTOR_STORE=vectorize
export PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID=…
export PORTFOLIO_RAG_CLOUDFLARE_API_TOKEN=…
export PORTFOLIO_RAG_CLOUDFLARE_VECTORIZE_INDEX=…

uv run portfolio-rag knowledge index --dry-run   # what would change; zero provider calls
uv run portfolio-rag knowledge index             # apply it
```

Re-run it after every corpus change. A second identical run is a true no-op — zero embeddings, zero
writes — and a metadata-only edit rewrites the record without paying for a vector.

**One honest limitation.** Cloudflare Vectorize cannot enumerate an index, so a run against it can
create and update records but **cannot discover stale ones**. The CLI says so rather than implying a
clean sync. In practice: deleting or renaming a document leaves its old vectors behind, and the way
to remove them is to delete and recreate the index, then re-index. This is a property of the store,
not a bug to be abstracted away.

## 4. Measure retrieval against the real corpus

**Done for v1.0.0, and repeatable.** The similarity threshold is provider-specific: a value tuned
against one embedding model means nothing under another, so it is set from a run on the model
actually in use rather than carried over. The fixture-corpus figures in
[../evaluation/README.md](../evaluation/README.md) do not transfer, and are not what this step reads.

Re-run it whenever the corpus changes materially, the embedding model changes, or the chunking
policy changes — those are the three things that invalidate a retrieval measurement.

The real-corpus dataset is `evaluation/portfolio-questions.yaml`. Run it with the production
embedding configuration:

```bash
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml
```

Then set `PORTFOLIO_RAG_RETRIEVAL_MIN_SIMILARITY` from what the sweep shows, rather than leaving a
number nobody has checked against the model actually in use. The current deployment runs at the
shipped default, `0.250`. Note what a threshold can and cannot do: it filters noise, not topic. What
keeps an unsupported answer from being published is the grounding path, and no threshold value
substitutes for it.

## 5. Run the backend

```bash
docker build -t portfolio-rag-assistant .
docker run -d -p 8000:8000 \
  -e PORTFOLIO_RAG_ENVIRONMENT=production \
  -e PORTFOLIO_RAG_ALLOWED_ORIGINS=https://your-portfolio.example \
  -e PORTFOLIO_RAG_EMBEDDING_PROVIDER=mistral \
  -e PORTFOLIO_RAG_LLM_PROVIDER=cloudflare_workers_ai \
  -e PORTFOLIO_RAG_MISTRAL_API_KEY="$MISTRAL_API_KEY" \
  -e PORTFOLIO_RAG_VECTOR_STORE=vectorize \
  -e PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID="$CF_ACCOUNT_ID" \
  -e PORTFOLIO_RAG_CLOUDFLARE_API_TOKEN="$CF_API_TOKEN" \
  -e PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN="$CF_WORKERS_AI_TOKEN" \
  -e PORTFOLIO_RAG_CLOUDFLARE_VECTORIZE_INDEX=your-index-name \
  portfolio-rag-assistant
```

To tie an image to a release-acceptance run, build it from the same clean commit and name it by that
commit — see [RELEASE_ACCEPTANCE.md](RELEASE_ACCEPTANCE.md#linking-a-build-to-the-commit).

The image carries no secrets and runs as a non-root user. The corpus ships inside it, because the
service reads `knowledge/` at startup to resolve retrieved passages back to their text.

**Production fails closed.** With `PORTFOLIO_RAG_ENVIRONMENT=production`, the composition root
refuses to build the `deterministic` embedding provider or the `deterministic` generation stub, and
the process does not start. A production deployment cannot silently serve placeholder answers.

### The full environment

| Variable | Production | Notes |
| --- | --- | --- |
| `PORTFOLIO_RAG_ENVIRONMENT` | `production` | switches on JSON logs and the fail-closed check |
| `PORTFOLIO_RAG_ALLOWED_ORIGINS` | the portfolio origin | comma-separated; `*` is rejected |
| `PORTFOLIO_RAG_EMBEDDING_PROVIDER` | `mistral` | `deterministic` refused in production |
| `PORTFOLIO_RAG_LLM_PROVIDER` | `cloudflare_workers_ai` | `mistral` is the supported alternative; `deterministic` refused in production |
| `PORTFOLIO_RAG_VECTOR_STORE` | `vectorize` | `memory` would be empty and per-process |
| `PORTFOLIO_RAG_MISTRAL_API_KEY` | **secret** | embeddings; also the chat adapter if selected |
| `PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID` | required | shared by both Cloudflare products |
| `PORTFOLIO_RAG_CLOUDFLARE_API_TOKEN` | **secret** | scoped to Vectorize only |
| `PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN` | **secret** | scoped to Workers AI only |
| `PORTFOLIO_RAG_CLOUDFLARE_VECTORIZE_INDEX` | required | must already exist |
| `PORTFOLIO_RAG_MISTRAL_EMBEDDING_MODEL` | optional | default `mistral-embed` |
| `PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_CHAT_MODEL` | optional | default `@cf/openai/gpt-oss-120b` |
| `PORTFOLIO_RAG_MISTRAL_CHAT_MODEL` | optional | default `mistral-small-latest`; only read when the Mistral chat adapter is selected |
| `PORTFOLIO_RAG_RETRIEVAL_MIN_SIMILARITY` | optional | default 0.25; production runs at that value, evaluated in step 4 |
| `PORTFOLIO_RAG_RETRIEVAL_TOP_K` | optional | default 5 |
| `PORTFOLIO_RAG_PROVIDER_TIMEOUT_SECONDS` | optional | default 30 |
| `PORTFOLIO_RAG_REQUEST_DEADLINE_SECONDS` | optional | default 60, provisional; keep below the Cloud Run timeout |
| `PORTFOLIO_RAG_KNOWLEDGE_ROOT` | optional | default `knowledge` |
| `PORTFOLIO_RAG_LOG_LEVEL` | optional | default `INFO` |

## 6. Configure the edge

**Required, not optional.** The application deliberately has no rate limiting: it cannot identify a
client correctly without knowing the proxy topology, and a limiter that is wrong looks like
protection while providing none ([ADR 0008](adr/0008-abuse-boundary-at-the-edge.md)).

The gateway that does it lives in [`edge/`](../edge/README.md). The full chain:

```
browser
  └─ Turnstile widget ........................ site key (public)
     │  X-Turnstile-Token
     ▼
  Cloudflare Worker ........................... edge/src/worker.js
     ├─ path is /api/v1/chat ........... else 404
     ├─ method is POST ................. else 405
     ├─ body ≤ 16 KiB .................. else 413
     ├─ Turnstile verified server-side . else 401 / 403
     ├─ burst 3 / 10s per IP+route ..... else 429 + Retry-After: 10
     ├─ sustained 10 / 60s ............. else 429 + Retry-After: 60
     └─ sets X-Edge-Auth (never forwarded from the client)
     ▼
  Cloud Run
     ▼
  FastAPI
     ├─ edge auth guard ................ else 403, before any provider call
     ├─ body ≤ 16 KiB .................. else 413, before the body is parsed
     ├─ message ≤ 2000 characters ...... else 422
     ├─ CORS: configured origins only
     ▼
  RAG: embedding → Vectorize → context → generation → grounding → citations
```

Each layer refuses what it can refuse cheaply. The two size limits agree on purpose: a request the
origin would reject is rejected at the edge first and never crosses the network, and the one the
edge misses is still rejected before the origin parses it.

**The origin guard is what makes the gateway matter.** Without it the Cloud Run URL is a way around
Turnstile and the rate limiter. `PORTFOLIO_RAG_EDGE_SHARED_SECRET` is checked with a constant-time
compare before routing; a request without it gets `403` and costs one string comparison. `/health`
is deliberately outside the guard so a platform probe still works.

### Manual steps — Cloudflare

None of these are done by this repository, and none of them should be scripted with a real secret in
the repository.

1. **Create a Turnstile widget** (dashboard → Turnstile). Note the **site key** (public, goes in the
   frontend) and the **secret key** (never leaves the Worker).
2. **Generate the edge shared secret.** A long random string; the same value goes to Cloudflare and
   to Google. Generate it in your own shell, e.g. `openssl rand -base64 32`.
3. **Set both as Worker secrets** — not in `wrangler.toml`, not in git:
   ```bash
   cd edge && npx wrangler secret put TURNSTILE_SECRET_KEY
   ```
   ```bash
   cd edge && npx wrangler secret put EDGE_SHARED_SECRET
   ```
4. **Set the plain variables** in `edge/wrangler.toml`: `ORIGIN_URL` (the Cloud Run base URL, no
   trailing slash) and `ALLOWED_ORIGINS` (the portfolio origin).
5. **Confirm the two rate-limit bindings** in `edge/wrangler.toml` — `CHAT_BURST_LIMITER`
   (3 per 10s) and `CHAT_SUSTAINED_LIMITER` (10 per 60s). Two bindings, because one window cannot
   express both a burst and a drip.
6. **Deploy**: `cd edge && npx wrangler deploy`.
7. **Point the portfolio at the Worker**, not at Cloud Run.

### Manual steps — Google Cloud Run

1. **Provide the same edge secret** through Secret Manager, mounted as
   `PORTFOLIO_RAG_EDGE_SHARED_SECRET`.
2. **Cap the blast radius**, which is a cost control and *not* a rate limiter — it bounds what a
   flood can spend, it does not decide who gets served:
   ```bash
   gcloud run services update portfolio-rag --max-instances=3 --concurrency=8
   ```
3. **Confirm the request timeout** (see below).

### The request timeout

Cloud Run's default request timeout is **300 seconds**. Nothing in this repository sets it — it is
service configuration, set on the Cloud Run service itself.

**What one chat request can cost in time.** One query embedding, one Vectorize query, and at most
**three calls to the generation provider**: the answer, one regeneration when the failure policy
allows it (`rag/failure_policy.py`), and the grounding check. Each provider call may be retried by
its adapter beneath the port — up to three requests on the wire, `PORTFOLIO_RAG_PROVIDER_TIMEOUT_SECONDS`
(default 30s) each. Unbounded, that adds up to several minutes; an earlier version of this section
counted two calls and 60–70 seconds, which stopped being true when the grounding check and the
regeneration were added.

**The application's own bound: `PORTFOLIO_RAG_REQUEST_DEADLINE_SECONDS`, default 60s.** Every step
of one request — retrieval, each generation, the grounding check and every transport retry beneath
them — spends from that one deadline. When it passes, the step under way is cancelled, no further
provider call is started, and the client receives a `503` in the application's error envelope
(`GENERATION_UNAVAILABLE`, or `RETRIEVAL_UNAVAILABLE` if it passed during retrieval). The value is
**provisional**: chosen to sit below the platform timeout, not measured. Re-set it from unpaced
production latencies (the `provider call` log lines record each call's elapsed time); an evaluation
run's elapsed times include pacing waits and must not be used for it.

**Recommended Cloud Run timeout: 90 seconds** — above the application deadline, so that the
application, not the platform, ends a slow request and answers it with its own envelope, and short
enough that a stuck request does not hold one of the 24 concurrency slots (`--concurrency=8`,
`--max-instances=3`) for five minutes.

```bash
gcloud run services update portfolio-rag --timeout=90
```

This is a recommendation, not a change: the value belongs to the Cloud Run service, and nothing in
this repository sets or reads it. Keep it above `PORTFOLIO_RAG_REQUEST_DEADLINE_SECONDS`.

## 7. Verify

The production smoke test. It was run against v1.0.0 and completed successfully — `/health`, a
grounded answer carrying citations, and the controlled refusal for a question the corpus does not
cover. Re-run it after every deployment: it is cheap, and it is the only check that exercises the
real providers end to end.

```bash
curl -fsS https://<origin-host>/health
curl -fsS -X POST https://<origin-host>/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"<a question your corpus answers>"}'
```

`/health` is outside the guard, so it answers unconditionally. The chat call does not: once the edge
is armed (step 6), the origin needs the gateway header and the gateway needs a fresh Turnstile
token, so run the second request through whichever of the two you are testing, with what that layer
requires. A `403` with no header is the guard working, not a failure.

Check, in order:

1. `/health` returns `{"status":"ok", …}`.
2. A question the corpus answers returns an answer **with citations** that point at real documents.
3. A question it does not cover returns `200`, the insufficient-knowledge sentence, and
   `"citations": []`.
4. A question aimed at an `internal` document returns nothing from it.
5. The response carries an `X-Request-ID` header.
6. Logs contain counts and durations — and no questions, contexts, prompts or answers.

## 8. Connect the front end

Point the existing chat UI at `POST /api/v1/chat`. The contract is in the README and in
`/openapi.json`:

```json
{ "answer": "…", "citations": [ { "document_id": "…", "title": "…", "source": "…", "section": "…" } ] }
```

`citations: []` means the knowledge base did not cover the question — render the answer, not an
error. Handle `503` as "temporarily unavailable, retry", and `422` as a client bug.

**No secret belongs in the front end.** The browser talks to this backend and to nothing else; every
credential stays server-side, which is the reason this service exists rather than the portfolio
calling providers directly.

## Rollback

The image is stateless and the index is derived data. Rolling back is redeploying the previous
image; if the corpus changed, re-run `knowledge index` from the previous revision. Nothing has to be
migrated, because nothing is stored.
