# Deployment

What the owner has to do to put this in front of a real portfolio. Nothing here has been executed:
no account exists, no Vectorize index has been provisioned, nothing is deployed.

Every environment variable named below is real — copied from `portfolio_rag.core.config.Settings`,
not invented for the document. No value shown is a credential.

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

**Do this before going live, not after.** Every retrieval number this project has was measured on
fixtures with a non-semantic embedding — see [../evaluation/README.md](../evaluation/README.md). The
similarity threshold in particular is provider-specific and its current value is a starting point,
not a measurement.

The real-corpus dataset is `evaluation/portfolio-questions.yaml`. Run it with the production
embedding configuration:

```bash
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml
```

Then set `PORTFOLIO_RAG_RETRIEVAL_MIN_SIMILARITY` from what the sweep shows, rather than leaving a
number nobody has checked against the model actually in use.

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
| `PORTFOLIO_RAG_RETRIEVAL_MIN_SIMILARITY` | set it after step 4 | default 0.25, unmeasured for this model |
| `PORTFOLIO_RAG_RETRIEVAL_TOP_K` | optional | default 5 |
| `PORTFOLIO_RAG_PROVIDER_TIMEOUT_SECONDS` | optional | default 30 |
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

Cloud Run's default request timeout is **300 seconds**. Nothing in this repository sets it, and no
value has been chosen yet.

For one RAG request that is far too long. The application's own bound is
`PORTFOLIO_RAG_PROVIDER_TIMEOUT_SECONDS`, default **30s**, applied per provider call — and a single
chat request makes at most two of them in sequence (one embedding, one generation), plus a Vectorize
query. The realistic worst case is therefore on the order of 60–70 seconds, and the common case is
two or three.

**Recommended: 90 seconds.** It leaves headroom above the worst case the application can produce,
and it stops a stuck request from occupying a concurrency slot for five minutes — which matters
precisely because `--concurrency=8` and `--max-instances=3` mean there are only 24 of them.

```bash
gcloud run services update portfolio-rag --timeout=90
```

This is a recommendation, not a change: the value is yours to set, and nothing here has set it.

## 7. Verify

```bash
curl -fsS https://your-backend.example/health
curl -fsS -X POST https://your-backend.example/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"<a question your corpus answers>"}'
```

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
