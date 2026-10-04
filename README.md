# Portfolio RAG Backend

A retrieval-augmented generation backend that answers questions about an engineer's work from a
curated, version-controlled knowledge base. The RAG mechanics — ingestion, chunking, embedding,
retrieval, context assembly, grounding and citation validation — are implemented in this repository
rather than delegated to an orchestration framework, and every external model or store is reached
through a port. FastAPI is the transport boundary; the providers behind it are configuration.

This repository is the backend only. Its production client is an existing React/Vite portfolio,
which lives in a separate repository and is developed independently of this one.

## Production status

| | |
| --- | --- |
| Release | **v1.0.0** |
| Runtime | Docker image on **Google Cloud Run**, behind a Cloudflare Worker gateway |
| Generation | **Cloudflare Workers AI** — `@cf/openai/gpt-oss-120b` |
| Embeddings | **Mistral** — `mistral-embed` (1024 dimensions, cosine) |
| Vector store | **Cloudflare Vectorize** |
| Verification | deployed and smoke-tested against the checks in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |

Three behaviours define the product surface:

* **Grounded answers.** An answer is generated only from passages retrieved for that question. There
  is no fallback to the model's own knowledge.
* **Backend-owned citations.** The model may select a source label; it can never name a document,
  path or URL and have that published. Every citation traces to a chunk that was actually retrieved.
* **Controlled unknowns.** When retrieval finds nothing above the similarity threshold, the service
  returns a fixed insufficient-knowledge sentence with an empty citation list, a normal `200`, and
  no language-model call at all.

This is not a claim to be hallucination-free, and no number it produces is a confidence. The
architecture narrows the room a model has to invent and makes what it publishes checkable. What is
enforced structurally and what remains a risk is written down in [docs/SECURITY.md](docs/SECURITY.md).

## Architecture at a glance

```
React portfolio / future clients        developer / CI
        │  HTTPS + JSON                        │  CLI
        ▼                                      ▼
┌───────────────────────────────┐   ┌────────────────────────────┐
│ api/     transport: routing,  │   │ cli.py  knowledge validate │
│          schemas, middleware, │   │         knowledge chunks   │
│          error envelope       │   │         knowledge index    │
│                               │   │         query retrieve     │
├───────────────────────────────┤   │         query answer       │
│ rag/     query side:          │   │         eval run           │
│   query boundary · retrieval  │   └─────────────┬──────────────┘
│   context · prompt · grounding│                 │
│   citations · orchestration   │   ┌─────────────▼──────────────┐
├───────────────────────────────┤   │ ingestion/  corpus side:   │
│ application/  indexing:       │◀──┤   discovery · frontmatter  │
│   desired state · plan ·      │   │   validation · normalize   │
│   convergent sync             │   │   provenance · chunking ·  │
│                               │   │   embedding representation │
├───────────────────────────────┴───┴────────────────────────────┤
│ domain/     knowledge · embeddings · retrieval vocabulary      │
│ ports/      LLMProvider · EmbeddingProvider · VectorStore ·    │
│             ChunkResolver                                      │
├────────────────────────────────────────────────────────────────┤
│ infrastructure/  embeddings:  deterministic · Mistral          │
│                  generation:  deterministic · Mistral ·        │
│                               Cloudflare Workers AI            │
│                  stores:      in-memory · Cloudflare Vectorize │
└────────────────────────────────────────────────────────────────┘
       core/  config · errors · logging · request context
     composition.py  the one place settings become adapters
```

Dependencies point inwards. `domain` and `ports` know nothing about HTTP, FastAPI or any provider
SDK, and neither does `rag`.

**Ingestion is the corpus side; `rag` is the query side.** Everything that turns source material into
indexable units — parsing, validation, chunking, embedding preparation — belongs to `ingestion`.
Nothing outside it opens a knowledge file or parses YAML, and within it only the loader touches the
filesystem: the chunker is a pure function from a document to chunks. Everything that happens once a
question arrives belongs to `rag`, works through ports, and names no vendor.

**`application` orchestrates; `infrastructure` implements.** Indexing composes an embedding provider,
a vector store and a change plan without knowing which adapters it was handed. Exactly one module,
`composition.py`, turns settings into concrete adapters, which is why swapping a provider is a
configuration change rather than a refactor.

Full write-up: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Request flow

```
user question
   ▼
POST /api/v1/chat                     edge guard + 16 KiB body limit, before routing
   ▼
validation                            message length, shape, unknown fields rejected
   ▼
query embedding                       mistral-embed; embedding space recorded with the vector
   ▼
vector retrieval                      Vectorize, visibility=public built into the query
   ▼                                  below-threshold matches dropped
bounded context                       ranked, deduplicated, token-budgeted, labelled,
   ▼                                  never truncated mid-passage
LLM generation                        Workers AI, grounding instructions as a system message
   ▼
grounding check                       an answer with no verifiable source is not published
   ▼
citation validation                   labels resolved against chunks retrieved for this request;
   ▼                                  unknown labels are dropped and counted
JSON response                         answer + citations + X-Request-ID
```

Retrieval that finds nothing above the threshold short-circuits at step four: the generation
provider is never called, and the response is the insufficient-knowledge answer.

## Engineering roadmap

The system was built in six deliberate phases, each with its own definition of done, so that later
work extended earlier boundaries instead of restarting them. Objectives and deliverables per phase
are in [docs/ROADMAP.md](docs/ROADMAP.md).

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Foundation & Architecture — a typed, tested FastAPI application with the boundaries the RAG phases build into | complete |
| 2 | Reliable Knowledge Ingestion — Markdown and frontmatter into validated, traceable documents | complete |
| 3 | Structure-Aware Chunking & Chunk Provenance — deterministic retrieval units that keep their structure and origin | complete |
| 4 | Embeddings & Vector Indexing — versioned embeddings, incremental and provider-agnostic synchronization | complete |
| 5 | Retrieval & Grounded RAG — public-only retrieval, bounded context, grounding and backend-owned citations | complete |
| 6 | Final Engineering & Production Readiness — evaluation, adversarial testing, security audit, container and deployment | complete |

Phases 1–5 built the product. **Phase 6 added no features**: it measured the pipeline, attacked it,
audited it and made it deployable — the work that closed out v1.0.0. There is no Phase 7; further
behavioural change is a new, explicit version rather than an open-ended phase.

## Key engineering properties

* **Ports and adapters.** `LLMProvider`, `EmbeddingProvider`, `VectorStore` and `ChunkResolver` are
  the only way out of the core. Provider and store are independent — any supported combination
  works, and `composition.py` is the single place any of them is chosen
  ([ADR 0002](docs/adr/0002-provider-agnostic-core.md)).
* **Structured knowledge ingestion.** Markdown with YAML frontmatter, one topic per file, reviewable
  in a pull request. Every document declares `schema_version`; a document written against an unknown
  version is refused rather than half-understood. YAML is parsed only through a `SafeLoader`
  subclass, and duplicate keys are rejected
  ([ADR 0004](docs/adr/0004-versioned-markdown-knowledge-format.md)).
* **Deterministic chunking and provenance.** Documents are cut along their own Markdown structure —
  headings become a `heading_path`, blocks are packed whole, code fences stay intact. The same
  corpus produces the same chunks, ids and SHA-256 fingerprints on any machine, with no model in the
  loop ([ADR 0005](docs/adr/0005-deterministic-structure-aware-chunking.md)).
* **Public/internal knowledge boundary.** Each document declares a `visibility`. `public` is built
  into the query the retrieval service sends — it is not a parameter, a policy field or a flag a
  caller can pass, forget or invert. Chunks are re-checked against the corpus after resolution, so a
  document reclassified to `internal` stops being retrievable before the index is rebuilt.
* **Incremental indexing.** An embedding fingerprint decides re-embedding, so a metadata-only change
  rewrites the record around the vector it already has and costs no provider call. A second run over
  an unchanged corpus is a true no-op ([ADR 0006](docs/adr/0006-versioned-embeddings-and-incremental-indexing.md)).
* **Embedding-space identity.** Query and index spaces are compared by `EmbeddingSpec` equality.
  Equal dimensionality is not compatibility, for indexing or for a query vector.
* **Grounded generation and backend-owned citations.** Instructions are a system message; passages
  and the question are a user message. No configuration and no document text can move a passage into
  the instruction role ([ADR 0007](docs/adr/0007-grounded-retrieval-and-backend-owned-citations.md)).
* **Bounded provider failure handling.** The Mistral and Workers AI adapters retry a bounded number
  of transport attempts (3 by default) on retryable statuses, honour a capped `Retry-After`, and
  fail closed to `503` otherwise. A malformed provider reply produces no answer, and its raw text
  never reaches the client. Every outbound call is bounded by
  `PORTFOLIO_RAG_PROVIDER_TIMEOUT_SECONDS` (default 30s).
* **Fail-closed production configuration.** With `PORTFOLIO_RAG_ENVIRONMENT=production`, the
  composition root refuses to build either development stand-in and the process does not start; a
  production app that requires the edge guard without a secret refuses to start too. CORS wildcards
  are rejected in every environment. A CI step asserts the first of these against the built image.

## Tech stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.13 (pinned; a minor upgrade is adopted deliberately, not by a resolver) |
| Web framework | FastAPI (ASGI), served by Uvicorn |
| Validation / settings | Pydantic v2, pydantic-settings |
| Embeddings | Mistral `mistral-embed` — 1024 dimensions, cosine |
| Vector store | Cloudflare Vectorize |
| Generation | Cloudflare Workers AI `@cf/openai/gpt-oss-120b` |
| HTTP client | `httpx2` — one async HTTP library for every adapter and for the test client |
| Frontmatter | PyYAML, through a `SafeLoader` subclass only |
| Markdown structure | markdown-it-py (parsing only, never rendering) |
| CLI | `argparse` (standard library) |
| Packaging | uv + `pyproject.toml`, Hatchling backend |
| Lint & format | Ruff (including `flake8-bandit` and `flake8-annotations`) |
| Type checking | mypy, `strict`, over `src` and `tests` |
| Tests | pytest + Starlette `TestClient`; `node --test` for the edge worker |
| CI | GitHub Actions |
| Container | Docker, `python:3.13-slim`, multi-stage, non-root |
| Hosting | Google Cloud Run |
| Edge gateway | Cloudflare Worker + Turnstile (`edge/`) |

Seven runtime dependencies and four development dependencies, unchanged since the first phase.
Retrieval, context building, prompting, grounding and citations added none.

A Mistral chat adapter (`mistral-small-latest`) is implemented and remains selectable, but it is not
what this deployment generates with.

## HTTP API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe. Unversioned on purpose, and outside the edge guard so a platform probe still works. No dependency checks. |
| `POST` | `/api/v1/chat` | Ask a question; get a grounded answer with citations. |

```bash
curl -s -X POST localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Which HTTP framework does the service use?"}'
```

```json
{
  "answer": "The service uses FastAPI for its HTTP API.",
  "citations": [
    {
      "document_id": "http-stack",
      "title": "HTTP Stack",
      "source": "docs/http-stack.md",
      "section": "Web framework"
    }
  ],
  "conversation_id": null
}
```

When the knowledge base does not cover the question the response is still `200`, with the
insufficient-knowledge sentence and `citations: []` — that empty list is how a client detects the
case. The sentence is written by the backend rather than the model, and comes back in the language
of the question (German or English; English for anything else).

**No conversation state.** `conversation_id` is echoed back and nothing more. A client may send the
turns immediately before the question in an optional `conversation` list (at most 6 turns, roles
`user` and `assistant`, each at most 2000 characters; the oldest whole turns beyond a 3000-character
budget are left out). They are used only so the model can read a follow-up such as "and how is it
deployed?": retrieval still searches for the question as asked, the grounding check never sees them,
and they are never a source or a citation. Nothing is stored between requests.

**What a response never contains:** similarity scores, source labels, the assembled context, the
prompt, the embedding space, vector ids, fingerprints, timings, or provider and model names. All of
that exists — in the CLI, in the logs, in tests — and none of it is a public promise.

**Error contract.** Every non-2xx response uses one envelope:

```json
{ "error": { "code": "VALIDATION_ERROR", "message": "…", "request_id": "…", "details": [] } }
```

| Status | Code | When |
| --- | --- | --- |
| `403` | `FORBIDDEN` | the request did not come through the edge gateway |
| `413` | `PAYLOAD_TOO_LARGE` | the body exceeds 16 KiB; it is never parsed |
| `422` | `VALIDATION_ERROR` | the payload or the question is unusable |
| `503` | `RETRIEVAL_UNAVAILABLE` | the embedding provider or vector store could not be reached |
| `503` | `GENERATION_UNAVAILABLE` | the generation provider failed or answered unusably |
| `500` | `INTERNAL_ERROR` | anything else; nothing about it is disclosed |

`code` is stable and machine-readable — branch on it, not on `message`. Messages never carry
internal details, provider payloads or stack traces; those are logged, never returned. Every request
gets an id, returned in `X-Request-ID` and referenced in error bodies; a well-formed client-supplied
id is reused.

OpenAPI is served at `/openapi.json` with Swagger UI at `/docs`, and documents only what the code
can actually produce.

## Reliability and security

Full detail, including residual risks, is in [docs/SECURITY.md](docs/SECURITY.md). The summary:

* **Abuse boundary at the edge.** The application has no rate limiter on purpose: it cannot identify
  a client correctly without knowing the proxy topology, and a limiter that is wrong looks like
  protection while providing none ([ADR 0008](docs/adr/0008-abuse-boundary-at-the-edge.md)). The
  Cloudflare Worker in [`edge/`](edge/README.md) verifies a Turnstile token server-side and applies
  a burst limit (3 / 10s) and a sustained limit (10 / 60s) per IP and route before forwarding.
* **Origin guard.** `POST /api/v1/chat` refuses any request that did not come through the gateway.
  The header is compared in constant time before routing, body parsing or dependency resolution, so
  a rejected request costs one string comparison and no provider call. It authenticates the
  *gateway*, not a user — there are no accounts here.
* **Input bounds**, in the order a request meets them: 16 KiB body (before parsing), 2000 characters
  per message (HTTP schema), 4000 characters per query (the pipeline's own bound, which also applies
  to the CLI). Empty, whitespace-only and invisible-only messages are rejected before any provider
  call; unknown fields are rejected rather than ignored; a rejected value is never echoed back.
* **Nothing is scrubbed.** Angle brackets, braces, backticks and quotes survive, because
  `How is <T> serialized?` is a legitimate question here and the message never reaches a shell, a
  path, a template or a query.
* **Logging.** Never logged at any level: the question, the context, the prompt, the answer,
  embedding vectors, credentials, `Authorization` headers, provider response bodies, internal
  knowledge. Provider error messages are composed locally from a status code and the operation, so a
  provider that returns a secret in an error body cannot put it in a log line.
* **Prompt injection is not solved** — nothing solves it. What is demonstrated, against a model
  scripted to have fallen for a hostile knowledge document completely, is that such a document
  cannot become a system message, change the retrieval policy, authorise a citation, or reach
  internal knowledge — and that an answer it induced is not published, because none of its claimed
  sources verify.
* **Secrets** come from the environment or a secret store. Nothing is baked into the image, and the
  container carries no credential.

## Evaluation and testing

Retrieval quality is measured rather than asserted:

```bash
uv run portfolio-rag eval run                    # retrieval + grounding, configured providers
uv run portfolio-rag eval run --retrieval-only   # no generation provider is called at all
uv run portfolio-rag eval run --generation-delay-seconds 8   # pace a run inside a rate limit
```

The runner reports hit@1/@3/@5, MRR, a similarity-threshold sweep and **every failing question by
name** — there is no single score that could hide one. `evaluation/` holds two datasets: 24 questions
in 8 categories against a neutral fixture corpus, and a larger set against the real corpus. Ground
truth is a document id plus a section heading, never a chunk id, so the dataset survives a
re-chunking.

Read [evaluation/README.md](evaluation/README.md) before quoting any number from it: the recorded
measurements were taken against the fixture corpus, and retrieval parameters are provider-specific.
`min_similarity` measured under one embedding model says nothing under another, which is why the
threshold is changed only from a measurement and never to make a benchmark look better.

The automated gate is what CI runs on every push and pull request:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

CI adds a second job: build the image, start it on the offline stack, and assert that `/health` is
`ok`, that `/openapi.json` describes `/api/v1/chat`, and that the endpoint answers a real request —
then assert that the same image *refuses to start* in production with a development stand-in
configured.

Tests are organised by what they protect: `tests/unit/` (models, config, errors, ports, ingestion,
chunking, retrieval, adapters), `tests/integration/` (HTTP behaviour, indexing, the full RAG flow,
adversarial and hostile input, the provider failure matrix, composition, the CLI),
`tests/evaluation/` (the dataset as a regression guard, with real metrics), `tests/contracts/`
(behavioural contracts every adapter must satisfy) and `tests/live/` (opt-in, against real
providers, skipped without credentials). At v1.0.0 the suite collects 1659 tests; a standard run
passes 1654 and skips the 5 live ones. Nothing in CI needs a credential, a network or a cent.

## Local development

Requires [uv](https://docs.astral.sh/uv/), which provisions the pinned interpreter itself — no
system Python of a particular version is needed.

```bash
uv sync                  # dev tooling is a PEP 735 group, installed by default
cp .env.example .env     # optional; the defaults are local-development defaults
uv run uvicorn --app-dir src portfolio_rag.main:app --reload --port 8000
```

The defaults are entirely offline — a deterministic embedding provider, a deterministic generation
stub and an in-memory vector store — so everything below runs with no account, no key and no cost:

| Setting | Default | Alternatives |
| --- | --- | --- |
| `PORTFOLIO_RAG_EMBEDDING_PROVIDER` | `deterministic` | `mistral` |
| `PORTFOLIO_RAG_LLM_PROVIDER` | `deterministic` | `cloudflare_workers_ai`, `mistral` |
| `PORTFOLIO_RAG_VECTOR_STORE` | `memory` | `vectorize` |

Both stand-ins are honest about being stand-ins. The embedding provider derives stable vectors from
SHA-256 — real infrastructure for tests, **not a semantic model**, so similarity between two of its
vectors means nothing about meaning and nothing clears the default threshold. The generation stub
produces no language at all; it reports which sources were retrieved and says plainly what it is.
Neither can be selected in production.

### CLI

Ingestion, indexing and inspection are developer workflows, not HTTP endpoints. They are reached
through the `portfolio-rag` console script, and all accept `--root` to point at a different
knowledge base:

```bash
uv run portfolio-rag knowledge validate            # ingest the corpus, report every problem found
uv run portfolio-rag knowledge inspect <id>        # metadata, provenance, hash — not the content
uv run portfolio-rag knowledge chunks <id>         # where a document is cut (--show-content, --all)
uv run portfolio-rag knowledge embedding <id>      # exactly what would be embedded, and its identity
uv run portfolio-rag knowledge index --dry-run     # what indexing would change; zero provider calls
uv run portfolio-rag knowledge index               # apply the plan (--rebuild re-embeds everything)
uv run portfolio-rag query retrieve "…"            # everything up to, and not including, the model
uv run portfolio-rag query answer "…"              # the whole pipeline (--show-retrieval, --show-context)
```

Exit codes: `0` success, `1` invalid knowledge base or document not found, `2` usage error, `3`
unexpected internal failure — a crash is never reported as a clean run, and no tracebacks are
printed.

Two design notes worth knowing. `validate` reports everything it can find rather than stopping at
the first broken document, while the programmatic entry point `load_knowledge_base` raises instead of
returning a corpus that quietly lost documents. And the query commands exist because "the answer is
wrong" is usually a retrieval problem: every match reports its **similarity**, which is a cosine
score that orders results — never a confidence and never a percentage. `--show-retrieval` and
`--show-context` are developer tooling; the HTTP API returns none of it and none of it is logged.

If the console script cannot import the package, run it from the source tree:
`PYTHONPATH=src uv run python -m portfolio_rag knowledge validate`.

| Task | Command |
| --- | --- |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Type check | `uv run mypy` |
| Tests | `uv run pytest` |
| Edge worker tests | `node --test edge/test` |

### Container

```bash
docker build -t portfolio-rag-assistant .
docker run --rm -p 8000:8000 \
  -e PORTFOLIO_RAG_ENVIRONMENT=local \
  -e PORTFOLIO_RAG_ALLOWED_ORIGINS=http://localhost:5173 \
  portfolio-rag-assistant
```

The image defaults to `PORTFOLIO_RAG_ENVIRONMENT=production`, so a local run has to say `local`
explicitly — a production container refuses to start on development stand-ins. It runs as a
non-root user, installs from the lockfile, carries no secrets, and reads `PORT` from the environment
so a platform that injects one (Cloud Run injects 8080) is served correctly. The corpus ships inside
it, because retrieved passages are resolved back to their text locally rather than stored in the
vector store. There is no Compose file: there is one runtime component.

The full production procedure — provider accounts, creating the Vectorize index at a matching
dimensionality, indexing the corpus offline, measuring retrieval against the real embedding model,
Cloud Run settings, and arming the edge gateway — is in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repository structure

```
src/portfolio_rag/
  main.py               ASGI application factory — the HTTP entry point
  cli.py                developer CLI — the terminal entry point
  composition.py        settings → adapters, in one place
  api/                  routes, schemas, error handlers, request-id and gate middleware
  ingestion/            discovery · frontmatter · metadata · normalization · provenance
    chunking/           structure · packing · policy · fingerprint · statistics
    embedding.py        the embedding representation and its fingerprint
  application/indexing/ desired state, index plan, convergent synchronization
  rag/                  query · policy · retrieval · context · prompt · generation ·
                        citations · language · service · tokens
  evaluation/           dataset model, retrieval metrics, evaluation runners, pacing
  infrastructure/       embedding/ · llm/ · vector_store/ · knowledge/ adapters
  core/                 config, error taxonomy, logging, request context
  domain/               KnowledgeDocument, KnowledgeChunk, RetrievedChunk, SourceCitation
  ports/                LLMProvider, EmbeddingProvider, VectorStore, ChunkResolver
knowledge/              the corpus, its format standard and a template
evaluation/             the question sets, the fixture corpus, and what was measured
edge/                   the Cloudflare Worker gateway and its tests
tests/                  unit · integration · evaluation · contracts · live · fixtures
docs/                   ARCHITECTURE.md, ROADMAP.md, SECURITY.md, DEPLOYMENT.md, adr/
```

The knowledge format — required frontmatter fields, id rules, discovery rules — is specified in
[knowledge/README.md](knowledge/README.md) and enforced by `portfolio-rag knowledge validate`.

## Architecture decisions

Decisions that would be expensive to reverse are recorded in [docs/adr/](docs/adr/), one file per
decision, written when the decision was made rather than rationalised afterwards:

| ADR | Decision |
| --- | --- |
| [0001](docs/adr/0001-modular-monolith.md) | Modular monolith instead of microservices |
| [0002](docs/adr/0002-provider-agnostic-core.md) | External AI and storage systems are reached through ports |
| [0003](docs/adr/0003-fastapi-portable-runtime.md) | FastAPI/ASGI is the portable core; Cloudflare is a deployment target |
| [0004](docs/adr/0004-versioned-markdown-knowledge-format.md) | Knowledge lives in versioned Markdown with YAML frontmatter |
| [0005](docs/adr/0005-deterministic-structure-aware-chunking.md) | Deterministic, structure-aware chunking |
| [0006](docs/adr/0006-versioned-embeddings-and-incremental-indexing.md) | Versioned embedding representation and incremental indexing |
| [0007](docs/adr/0007-grounded-retrieval-and-backend-owned-citations.md) | Grounded retrieval, public by construction, backend-owned citations |
| [0008](docs/adr/0008-abuse-boundary-at-the-edge.md) | The public abuse boundary lives at the deployment edge |

The trail matters more than the snapshot: superseding beats editing, so an ADR that stops being true
is marked superseded rather than quietly rewritten.

## Deliberate non-goals

Complexity is not added to make the architecture look larger. Each of these was considered and
declined, with the reasoning in [docs/ROADMAP.md](docs/ROADMAP.md):

* **No orchestration framework** — LangChain, LlamaIndex and Haystack are not used. Chunking,
  retrieval, ranking, grounding and evaluation are where answer quality is won or lost, and they are
  written here so they can be measured and changed.
* **No microservices.** One deployable with real module boundaries. Extraction stays possible;
  nothing is distributed without a reason.
* **No conversation memory, chat history or database.** The assistant answers one question at a time
  from a fixed corpus. A client may send the last few turns to make a follow-up readable; the
  service stores nothing, and those turns are never evidence.
* **No agents or tool use.** There is one job: answer from the corpus, or say it cannot.
* **No re-ranking, hybrid/BM25 search or query rewriting.** Real techniques, but evaluation found
  the limiting factor to be threshold calibration under a specific embedding model rather than
  ranking, so none of them is justified yet.
* **No abstraction without a caller** — no package, interface or configuration field exists for a
  future that has not arrived.
* **No world-knowledge fallback, and no audit trail.** If the corpus does not support an answer, the
  system says so. Nothing records who asked what, deliberately: there is no user to attribute a
  question to, and storing questions would create a privacy obligation the system does not need.

## Development workflow

Engineering rules that apply to every contributor are in [AGENTS.md](AGENTS.md);
[CLAUDE.md](CLAUDE.md) is a short orientation file, and `.claude/skills/` holds repository-specific
review checklists (architecture, API, security, test gate).

The repository was developed with human direction and AI-assisted coding and review. Every change
went through the same gate as any other: Ruff, mypy `strict`, the test suite, and the ADR trail for
anything structural.

## License

MIT.
