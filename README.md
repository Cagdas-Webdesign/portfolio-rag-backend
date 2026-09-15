# portfolio-rag-assistant

A provider-agnostic backend platform for retrieval-augmented knowledge assistants.

Its first client will be an existing React/Vite portfolio site whose chat UI currently talks to a
Cloudflare Worker. This service is being built to replace and extend that Worker — without the
portfolio itself changing in the meantime.

> **Status: complete and unreleased.** All six phases are done — the pipeline is built, measured,
> adversarially tested and documented for deployment. `POST /api/v1/chat` answers a question from
> public knowledge with citations the backend verified itself, or says honestly that it cannot.
> Everything runs offline with no account and no key.
>
> **It is not deployed.** `knowledge/` contains the authorized portfolio corpus. Retrieval against
> the production embedding model still needs to be measured before deployment; the procedure is in
> [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## What it does

* **Semantic retrieval** over a corpus of versioned Markdown documents.
* **Public-only retrieval**, enforced structurally: a public request cannot express a search for
  `internal` documents, so it cannot accidentally perform one.
* **Bounded context building** — ranked, deduplicated, token-budgeted, never truncated mid-passage.
* **Provider-agnostic generation** through a narrow port; Cloudflare Workers AI and Mistral are
  two adapters behind it, not a dependency.
* **Grounded answers**: generated only from passages retrieved for that question, with no fallback
  to the model's own knowledge when the corpus is silent.
* **Backend-owned citations**: the model may select a source label, never name a source. Every
  citation traces to a chunk that was actually retrieved.
* **An honest "I don't know"** when nothing supports an answer — with no provider call, no invented
  facts, and a normal `200`.
* **End-to-end offline tests**: ingestion to answer, no credentials, no network, no cost.

It does not claim to be hallucination-free, and no number in it is presented as a confidence. The
architecture reduces the room a model has to invent and makes what it says checkable; that is a
different claim, and the honest one. What is enforced structurally, and what remains a risk, is
written down in [docs/SECURITY.md](docs/SECURITY.md).

Explicit non-goals: microservices, an orchestration framework (LangChain, LlamaIndex, Haystack),
paid infrastructure, conversation memory, agents, and abstractions without a caller.

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
│ rag/     query side:          │   └─────────────┬──────────────┘
│   query boundary · retrieval  │                 │
│   context · prompt · grounding│   ┌─────────────▼──────────────┐
│   citations · orchestration   │   │ ingestion/  corpus side:   │
├───────────────────────────────┤   │   discovery · frontmatter  │
│ application/  indexing:       │◀──┤   validation · normalize   │
│   desired state · plan ·      │   │   provenance · chunking ·  │
│   convergent sync             │   │   embedding representation │
├───────────────────────────────┴───┴────────────────────────────┤
│ domain/     knowledge · embeddings · retrieval vocabulary      │
│ ports/      LLMProvider · EmbeddingProvider · VectorStore ·    │
│             ChunkResolver                                      │
├────────────────────────────────────────────────────────────────┤
│ infrastructure/  deterministic + Mistral embeddings            │
│                  deterministic stub + Mistral + Workers AI     │
│                  generation                                    │
│                  in-memory + Cloudflare Vectorize stores       │
└────────────────────────────────────────────────────────────────┘
       core/  config · errors · logging · request context
     composition.py  the one place settings become adapters
```

Dependencies point inwards. `domain` and `ports` know nothing about HTTP, about FastAPI or about
any provider SDK, and neither does `rag`.

**Ingestion is the corpus side; `rag` is the query side.** Everything that turns source material
into indexable units — parsing, validation, chunking, embedding preparation — belongs to
`ingestion`. Nothing outside it opens a knowledge file or parses YAML, and within it only the loader
touches the filesystem: the chunker is a pure function from a document to chunks. Everything that
happens once a question arrives — the input boundary, retrieval, context, prompting, grounding,
citations — belongs to `rag`, works through ports, and touches no file and no vendor.

**`application` orchestrates; `infrastructure` implements.** Indexing composes an embedding
provider, a vector store and a change plan without knowing which adapters it was handed. Exactly one
module — `composition.py` — turns settings into concrete adapters, which is why swapping a provider
is a configuration change rather than a refactor.

Full write-up: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Decisions and their reasoning: [docs/adr/](docs/adr/).
Security boundaries and residual risks: [docs/SECURITY.md](docs/SECURITY.md).
Putting it in production: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Tech stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.13 |
| Web framework | FastAPI (ASGI) |
| Validation / models | Pydantic v2, pydantic-settings |
| Server | Uvicorn |
| Frontmatter | PyYAML (`SafeLoader` subclass only) |
| HTTP client | httpx (Mistral, Workers AI and Vectorize adapters) |
| Markdown structure | markdown-it-py (parsing only, never rendering) |
| CLI | `argparse` (standard library) |
| Packaging / envs | uv + `pyproject.toml` (Hatchling backend) |
| Lint & format | Ruff |
| Type checking | mypy (`strict`) |
| Tests | pytest + Starlette `TestClient` |
| CI | GitHub Actions |
| Container | Docker (`python:3.13-slim`, non-root) |

Seven runtime dependencies, four development dependencies — unchanged since Phase 1. Retrieval, context building, prompting, grounding and citations added none.

## Local setup

Requires [uv](https://docs.astral.sh/uv/). uv provisions the Python interpreter itself — the version
is pinned in `.python-version` — so no system Python of a particular version is needed.

```bash
uv sync
```

Development tooling is a PEP 735 dependency group, so `uv sync` installs it by default and the
container image excludes it with `--no-dev`.

Configuration is optional — the defaults are local-development defaults:

```bash
cp .env.example .env
```

## Development

Run the API with auto-reload:

```bash
uv run uvicorn --app-dir src portfolio_rag.main:app --reload --port 8000
```

`--app-dir src` imports the package straight from the source tree, so development does not depend on
an editable install being wired up correctly. Tests do the same via pytest's `pythonpath` setting.
The container image installs the package properly and needs neither.

| Task | Command |
| --- | --- |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Format check (CI) | `uv run ruff format --check .` |
| Type check | `uv run mypy` |
| Tests | `uv run pytest` |
| Everything CI runs | `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest` |

## CLI

Ingestion is a developer workflow, not an HTTP endpoint. It is reached through the `portfolio-rag`
console script:

```bash
uv run portfolio-rag knowledge validate                  # ingest and report problems
uv run portfolio-rag knowledge inspect api-integrations  # what ingestion made of one document
uv run portfolio-rag knowledge chunks api-integrations   # where that document is cut
uv run portfolio-rag knowledge chunks --all              # corpus-wide chunk statistics
uv run portfolio-rag knowledge embedding api-integrations  # what would be embedded, and its identity
uv run portfolio-rag knowledge index --dry-run          # what indexing would change
uv run portfolio-rag knowledge index                    # make it so
```

All accept `--root` to point at a different knowledge base. Exit codes: `0` success, `1` invalid
knowledge base or document not found, `2` usage error, `3` unexpected internal failure — a crash is
never reported as a clean run. No tracebacks are printed.

```console
$ portfolio-rag knowledge validate --root tests/fixtures/knowledge/invalid
Knowledge root: tests/fixtures/knowledge/invalid

✓ 00-claims-an-id.md
✗ 01-missing-frontmatter.md
    MISSING_FRONTMATTER  Document does not start with a `---` frontmatter block.
✗ 03-unsupported-schema-version.md
    UNSUPPORTED_SCHEMA_VERSION  schema_version: 99 is not supported; this build reads version 1
✗ 04-invalid-metadata.md
    INVALID_METADATA  language: String should match pattern '^[a-z]{2}$'
✗ 06-duplicate.md
    DUPLICATE_DOCUMENT_ID  id: `contested-id` first claimed by 00-claims-an-id.md

7 documents
1 valid
6 errors
```

One broken document does not hide the others: a batch run reports everything it can find. The
programmatic entry point, `load_knowledge_base`, is stricter — it raises rather than return a corpus
that quietly lost documents.

`inspect` reports what the pipeline made of one document — metadata, provenance, content hash and
size — deliberately not the content itself:

```console
$ portfolio-rag knowledge inspect alpha --root tests/fixtures/knowledge/valid
ID             alpha
Title          Test Document Alpha
Schema         1
Type           reference
Language       en
Version        1
Updated        2026-08-07
Visibility     public
Trust          verified
Source         tests/fixtures/knowledge/valid/alpha.md
Path           alpha.md
Hash           81b8c029dd6be11205cae87d38645740f1cf0188eaabb4ae064f88633967ce5c
Characters     234
Topics         testing, ingestion
Technologies   Python, Markdown
```

`chunks` shows where a document is cut into retrieval units, and why — the boundaries, the heading
path each unit belongs to, its fingerprint, and how much text is repeated from the previous chunk:

```console
$ portfolio-rag knowledge chunks levels
Document: Heading Levels
ID: levels

Policy
  strategy       markdown-structure-v1
  target chars   1200
  max chars      1800
  overlap chars  150

Chunk 0000
  ID             levels--0000
  Heading        —
  Characters     28
  Fingerprint    0bb548cc8277d47f17582f43ecc0dc3b5fbdef5126d88092b6ad7c32c6e5a63d
  Overlap        0

Chunk 0001
  ID             levels--0001
  Heading        Backend > APIs
  Characters     35
  Fingerprint    f6665828d0ad9036065131b76a4ff4d684a54af8b5fe0fca8ca9d8cf00d21f22
  Overlap        0

2 chunks
```

Add `--show-content` to print each chunk's text and check the boundaries by eye. The size budget can
be overridden per run — `--target-chars`, `--max-chars`, `--overlap-chars` — to compare boundaries
without changing anything on disk. `--all` reports counts and sizes across the corpus; it refuses to
run on a knowledge base that has unusable documents, because a partial corpus summarised as a whole
one would be a lie with numbers on it.

The output deliberately contains no quality score. Structural chunk statistics do not establish
retrieval quality; that is measured separately against questions with known answers.

### Embedding and indexing

`embedding` shows exactly what would be sent to an embedding provider, and the identity that decides
whether it has to be sent at all. It makes no request and needs no credentials:

```console
$ portfolio-rag knowledge embedding guide
Embedding space
  provider       deterministic
  model          sha256-derived-v1
  dimensions     256
  representation embedding-text-v1

Chunk 0000
  Chunk ID       guide--0000
  Document       guide
  Heading        Backend > APIs
  Representation embedding-text-v1
  Dimensions     256
  Fingerprint    dacdbeacdff81f1a669b1980b1029dd68561e14ba516d87523cdf49e9d3649b6
  Characters     70

    │ Integration Guide
    │
    │ Backend > APIs
    │
    │ REST endpoints are documented here.
```

Add `--show-text` for the full representation rather than a preview.

`index --dry-run` reads the index and works out what would change, without embedding, writing or
deleting anything:

```console
$ portfolio-rag knowledge index --dry-run
Index plan
  create              3
  re-embed            0
  metadata only       0
  unchanged           0
  delete              0
  embeddings required 3

Dry run: nothing was embedded, written or deleted.
```

Without `--dry-run` it applies the plan and reports what it did. The distinction between
**re-embed** and **metadata only** is the one that matters: changing a chunk's text costs an
embedding, while flipping its `visibility` rewrites the record around the vector it already has. A
second run over an unchanged corpus is a true no-op — zero provider calls, zero writes.

`--rebuild` re-embeds everything instead of reusing what the index holds. It is the deliberate
escape hatch when the embedding space itself has changed.

### Asking questions

The query side has its own commands, and they exist because "the answer is wrong" is usually a
retrieval problem rather than a model problem — and a retrieval step you cannot see is one you
cannot fix.

`query retrieve` runs everything up to, and not including, the language model. The run below uses
the test fixture corpus and the offline defaults:

```console
$ portfolio-rag query retrieve "Which HTTP framework does the service use?" \
    --root tests/fixtures/knowledge/rag --top-k 1 --min-similarity -1
Question: Which HTTP framework does the service use?

Embedding space
  provider       deterministic
  model          sha256-derived-v1
  dimensions     256
  representation embedding-text-v1

Retrieval policy
  top k          1
  min similarity -1.000
  visibility     public (enforced)

Stack
  generation model context-echo-v1
  corpus chunks    7

Retrieval
  matches returned 1
  below threshold  0
  unresolved       0
  withheld         0
  retrieved        1

Match 1
  Similarity     0.0546
  Chunk ID       storage-layer--0001
  Document       Storage Layer
  Heading        Storage Layer > Corpus format
  Source         tests/fixtures/knowledge/rag/storage-layer.md
  Visibility     public

    │ Knowledge documents are stored as Markdown files.

1 passage
```

**That match is wrong, and the output shows exactly why.** The similarity is 0.05 — noise — because
the default embedding provider derives vectors from SHA-256 and understands nothing; the threshold
had to be disabled to see any result at all. With a real embedding provider the same command ranks
the passage that mentions FastAPI first. Printing the run as it actually is beats printing a
plausible one.

Every match reports its **similarity** — never a "confidence" and never a percentage. It is a cosine
score: it orders results, it does not say how likely an answer is to be correct.

`query answer` runs the whole pipeline and prints what came out of it — retrieval, context, the
model, and the citations the backend verified:

```console
$ portfolio-rag query answer "Which HTTP framework does the service use?" \
    --root tests/fixtures/knowledge/rag --top-k 2 --min-similarity -1
Answer
  outcome            answered
  retrieved          2
  context sources    2
  citations          2
  unknown labels     0
  retrieval seconds  0.0003
  generation seconds 0.0000
  total seconds      0.0003

    │ This is a development stub, not a generated answer. The configured generation
    │ provider is `deterministic`, which does not produce language. It reports the
    │ knowledge sources that were retrieved for this question so that the pipeline
    │ can be inspected end to end.

Sources
  [1] Storage Layer > Corpus format  (tests/fixtures/knowledge/rag/storage-layer.md)
  [2] HTTP Stack > HTTP Stack  (tests/fixtures/knowledge/rag/http-stack.md)
```

The answer text is the development stub saying what it is. Configure
`PORTFOLIO_RAG_LLM_PROVIDER=cloudflare_workers_ai` (or the supported `mistral` alternative), and the
same pipeline — same retrieval, same context, same citation validation — produces a real answer over
the same sources.

`--show-retrieval` adds the ranked passages, and `--show-context` prints the exact context the model
was given, labels and all. Both are developer tooling: the HTTP API returns none of it, and none of
it is logged.

### Measuring retrieval

The question "is retrieval any good?" has a reproducible answer rather than an opinion:

```bash
uv run portfolio-rag eval run                  # retrieval + grounding, configured providers
uv run portfolio-rag eval run --retrieval-only # no generation provider is called at all
uv run portfolio-rag eval run --generation-delay-seconds 8   # stay inside a provider's rate limit
```

It reports hit@1/@3/@5, MRR, threshold calibration and **every failing question by name** — there is
no single score that could hide one. `evaluation/` holds 24 questions in 8 categories with
hand-checkable ground truth, and `evaluation/README.md` records what the measurements settled,
including why the similarity threshold was left exactly where it was.

Read that file before quoting a number from it: the corpus is fixtures, and the offline embedding
provider has no semantics.

### Providers and stores

The defaults are the offline ones, so everything above runs with no account and no cost:

| Setting | Default | Alternative |
| --- | --- | --- |
| `PORTFOLIO_RAG_EMBEDDING_PROVIDER` | `deterministic` | `mistral` |
| `PORTFOLIO_RAG_LLM_PROVIDER` | `deterministic` | `cloudflare_workers_ai`, `mistral` |
| `PORTFOLIO_RAG_VECTOR_STORE` | `memory` | `vectorize` |

The two `deterministic` adapters are development stand-ins, and both are honest about it. The
embedding provider derives stable vectors from SHA-256: real infrastructure for tests, **not a
semantic model** — similarity between two of its vectors means nothing about meaning, which is why
the examples above were produced with a provider that does. The generation stub produces no language
at all; it reports which sources were retrieved and says plainly that it is a stub.

**Neither can be selected in production.** `PORTFOLIO_RAG_ENVIRONMENT=production` with either
`deterministic` adapter fails at startup, loudly, rather than serving a public page with a
placeholder. Real providers need their credentials, which come from the environment and are never
written to disk or baked into the image.

> If the console script cannot find the package (an editable install that did not register), run it
> from the source tree instead: `PYTHONPATH=src uv run python -m portfolio_rag knowledge validate`.

## API

| Method | Path | Status | Description |
| --- | --- | --- | --- |
| `GET` | `/health` | ✅ implemented | Liveness probe. Unversioned on purpose. |
| `POST` | `/api/v1/chat` | ✅ implemented | Ask a question; get a grounded answer with citations. |

```bash
curl -s localhost:8000/health
# {"status":"ok","service":"portfolio-rag-assistant","version":"0.1.0"}

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

Every citation refers to a document that was actually retrieved for that question. A source the
model merely named is never published — see
[ADR 0007](docs/adr/0007-grounded-retrieval-and-backend-owned-citations.md).

**When the knowledge base does not cover the question**, the response is still `200`:

```json
{
  "answer": "I don't have enough information in the available knowledge base to answer that.",
  "citations": [],
  "conversation_id": null
}
```

That is the honest outcome, not an error, and `citations: []` is how a client detects it. Nothing is
invented, and no language model is called when retrieval found nothing to ground an answer in.

The sentence is written by the backend, not by the model, and it comes back in the language of the
question — German or English, English for anything else. Grounded answers already follow the
question's language because the prompt says so; the refusal used to be the one reply that did not.

**No conversation state.** `conversation_id` is echoed back and nothing more; each question is
answered on its own, from the corpus.

**What a response never contains:** similarity scores, source labels, the assembled context, the
prompt, the embedding space, vector ids, fingerprints, timings, or the provider and model names.
All of that exists — in the CLI, in the logs, in tests — and none of it is a public promise.

**Error contract.** Every non-2xx response — validation failures and unexpected errors included —
uses one envelope:

```json
{ "error": { "code": "VALIDATION_ERROR", "message": "…", "request_id": "…", "details": [] } }
```

| Status | Code | When |
| --- | --- | --- |
| `422` | `VALIDATION_ERROR` | the payload or the question is unusable |
| `503` | `RETRIEVAL_UNAVAILABLE` | the embedding provider or vector store could not be reached |
| `503` | `GENERATION_UNAVAILABLE` | the generation provider failed or answered unusably |
| `500` | `INTERNAL_ERROR` | anything else; nothing about it is disclosed |

`code` is stable and machine-readable; branch on it, not on `message`. Messages never contain
internal details, provider payloads or stack traces — those are logged, never returned.

**Request correlation.** Every request gets an id, returned in the `X-Request-ID` header and
referenced in error bodies. A client-supplied `X-Request-ID` is reused when it is well-formed.

### OpenAPI

* Swagger UI: <http://localhost:8000/docs>
* Schema: <http://localhost:8000/openapi.json>

The document is kept honest: nothing is described that is not implemented, and everything that is
implemented is described. The chat endpoint documents its `200` with the real `ChatResponse` schema
and the three failures it can produce — and, since Phase 5, no `501`.

## Knowledge base

`knowledge/` holds the source documents the assistant is grounded in: Markdown with YAML
frontmatter, one topic per file, reviewable in a pull request. The format, the required frontmatter
fields, the id rules and the discovery rules are specified in
[knowledge/README.md](knowledge/README.md) and **enforced** by `portfolio-rag knowledge validate`.

Every document declares `schema_version: 1`; a document written against a version this build does
not know is refused rather than half-understood ([ADR 0004](docs/adr/0004-versioned-markdown-knowledge-format.md)).
Ingestion is deterministic: the same corpus produces the same documents and the same chunks, in the
same order, with the same SHA-256 fingerprints, on any machine.

Documents are then cut into retrieval units along their own Markdown structure — headings become a
`heading_path`, blocks are packed whole, code fences stay intact, and every chunk carries its
document's metadata and provenance so nothing downstream has to reopen the file
([ADR 0005](docs/adr/0005-deterministic-structure-aware-chunking.md)).

Each document declares a `visibility`, and it is load-bearing: `public` documents are the only ones
a chat request can retrieve, and that is enforced in the shape of the query rather than by a filter
somebody could forget.

The directory ships with a template only; no content has been invented.

## Docker

Run the image with the offline stack, which needs no credentials:

```bash
docker build -t portfolio-rag-assistant .
docker run --rm -p 8000:8000 \
  -e PORTFOLIO_RAG_ENVIRONMENT=local \
  -e PORTFOLIO_RAG_ALLOWED_ORIGINS=http://localhost:5173 \
  portfolio-rag-assistant
curl -s localhost:8000/health
```

A production run has to name real providers — the container refuses to start on development
stand-ins, which is the point:

```bash
docker run --rm -p 8000:8000 \
  -e PORTFOLIO_RAG_ENVIRONMENT=production \
  -e PORTFOLIO_RAG_ALLOWED_ORIGINS=https://your-portfolio.example \
  -e PORTFOLIO_RAG_EMBEDDING_PROVIDER=mistral \
  -e PORTFOLIO_RAG_LLM_PROVIDER=mistral \
  -e PORTFOLIO_RAG_MISTRAL_API_KEY="$MISTRAL_API_KEY" \
  portfolio-rag-assistant
```

The image runs as a non-root user, contains no secrets and installs dependencies from the lockfile.
Its entrypoint is the production start command:

```bash
uvicorn portfolio_rag.main:app --host 0.0.0.0 --port 8000
```

There is no Docker Compose file — there is only one runtime component.

## Project structure

```
src/portfolio_rag/
  main.py               ASGI application factory — the HTTP entry point
  cli.py                developer CLI — the terminal entry point
  api/
    router.py           composition of the HTTP surface
    error_handlers.py   exceptions → the public error envelope
    routes/             health.py, chat.py
    schemas/            request/response models
    middleware/         request id + access logging
  ingestion/
    loader.py           the public entry point: collect / load a knowledge base
    discovery.py        which files are documents, in a stable order
    frontmatter.py      split the YAML block, parse it safely
    metadata.py         schema version gate + field validation
    normalization.py    UTF-8, BOM, line endings, Unicode NFC
    provenance.py       canonical form and SHA-256 document hash
    errors.py           ingestion error taxonomy
    embedding.py        the embedding representation + its fingerprint
    chunking/
      chunker.py        chunk_document / chunk_knowledge_base — pure functions
      structure.py      Markdown headings and blocks, sliced from the source
      packing.py        block packing, oversized fallback, overlap
      policy.py         the character budget + strategy version
      fingerprint.py    chunk id and SHA-256 chunk fingerprint
      statistics.py     counts and sizes for developer inspection
  application/
    indexing/           desired state, index plan, convergent synchronization
  evaluation/           dataset model, retrieval metrics, evaluation runners
  rag/
    query.py            the input boundary + the query representation
    policy.py           top-k, similarity threshold, context budget
    retrieval.py        public-only search, threshold, chunk resolution
    context.py          bounded, labelled, deterministic context building
    prompt.py           the grounded prompt + its version
    generation.py       parsing the structured answer contract
    citations.py        backend-owned citation validation
    service.py          the orchestrator: question in, grounded answer out
    tokens.py           the documented token estimate
    errors.py           query-side error taxonomy
  infrastructure/
    embedding/          deterministic (offline) and Mistral adapters
    llm/                deterministic stub (offline), Mistral and Workers AI chat adapters
    vector_store/       in-memory reference and Cloudflare Vectorize adapters
    knowledge/          in-process corpus snapshot behind the ChunkResolver port
  composition.py        settings → adapters, in one place
  core/                 config, error taxonomy, logging, request context
  domain/               KnowledgeDocument, KnowledgeChunk, RetrievedChunk, SourceCitation
  ports/                LLMProvider, EmbeddingProvider, VectorStore, ChunkResolver
knowledge/              knowledge document standard + template
tests/unit/             models, config, errors, ports, ingestion, chunking, retrieval, adapters
tests/integration/      HTTP behaviour, indexing, the full RAG flow, adversarial input, the failure
                        matrix, hostile input, composition, the CLI
tests/evaluation/       the dataset as a regression guard, with real metrics
tests/contracts/        reusable behavioural contracts every adapter must satisfy
tests/doubles.py        scripted and lexical test adapters, for branches a real provider cannot take
tests/live/             opt-in tests against real providers; skipped without credentials
tests/fixtures/         small neutral knowledge bases: valid, broken on purpose, and query-side
evaluation/             the question set, its fixture corpus, and what was measured
docs/                   ARCHITECTURE.md, ROADMAP.md, SECURITY.md, DEPLOYMENT.md, adr/
.claude/skills/         repository-specific review workflows
```

## Architecture principles

1. **Modular monolith.** One deployable, real module boundaries. Extraction stays possible; nothing
   is distributed without a reason ([ADR 0001](docs/adr/0001-modular-monolith.md)).
2. **Provider-agnostic core.** External AI and storage systems are reached through ports; no SDK is
   imported by the core ([ADR 0002](docs/adr/0002-provider-agnostic-core.md)).
3. **Portable runtime.** FastAPI/ASGI is the application boundary. Cloudflare is a possible
   deployment target, not a dependency ([ADR 0003](docs/adr/0003-fastapi-portable-runtime.md)).
4. **Own the RAG mechanics.** Chunking, retrieval, ranking, grounding and evaluation are written
   here, not delegated to a framework, because that is where answer quality is won or lost.
5. **Versioned, git-native knowledge.** The corpus is reviewable Markdown; the index is derived data
   that can always be rebuilt from it ([ADR 0004](docs/adr/0004-versioned-markdown-knowledge-format.md)).
6. **Deterministic corpus processing.** The same documents produce the same chunks, ids and
   fingerprints on any machine, with no model in the loop
   ([ADR 0005](docs/adr/0005-deterministic-structure-aware-chunking.md)).
7. **Grounded, and only grounded.** An answer comes from retrieved public passages or it does not
   come at all; citations are built by the backend from what was actually retrieved
   ([ADR 0007](docs/adr/0007-grounded-retrieval-and-backend-owned-citations.md)).
8. **Honest surface.** Documentation and OpenAPI describe what exists today. Nothing here claims to
   be hallucination-free, and no score is presented as a confidence.

## Roadmap

| Phase | Focus | Status |
| --- | --- | --- |
| 1 | Foundation & Architecture | ✅ complete |
| 2 | Reliable Knowledge Ingestion | ✅ complete |
| 3 | Structure-Aware Chunking & Chunk Provenance | ✅ complete |
| 4 | Embeddings & Vector Indexing | ✅ complete |
| 5 | Retrieval & Grounded RAG | ✅ complete |
| 6 | Final Engineering & Production Readiness | ✅ complete |

Six phases, five of which built the product. Phase 6 added no features: it measured the pipeline,
attacked it, audited it and wrote down how to deploy it. There is no Phase 7.

Objectives, definitions of done, and what is deliberately *not* being built —  conversation memory,
agents, re-ranking, hybrid search — are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Contributing agents

This repository is worked on by both humans and coding agents. Engineering rules that apply to
everyone live in [AGENTS.md](AGENTS.md); [CLAUDE.md](CLAUDE.md) is the short orientation file for
Claude Code.

## License

MIT.
