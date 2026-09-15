# CLAUDE.md

Orientation for Claude Code. The engineering rules live in **[AGENTS.md](AGENTS.md)** — read that
file, it is not repeated here.

## What this is

A provider-agnostic RAG backend platform. This repository is the **backend only**. Its client is an
existing React/Vite portfolio, which is a **separate repository and must not be touched** — no
change here may edit, generate or assume frontend source.

**All six phases are complete, and v1.0.0 is the frozen production baseline.** The pipeline is
built, measured (`evaluation/`), adversarially tested and audited.

**It is deployed and running.** A Docker container on Google Cloud Run, behind a Cloudflare Worker
edge gateway, with production smoke tests completed successfully
([docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)):

| Concern | Production |
| --- | --- |
| Generation | Cloudflare Workers AI — `@cf/openai/gpt-oss-120b` |
| Embeddings | Mistral — `mistral-embed` |
| Vector store | Cloudflare Vectorize |

**There is no Phase 7.** Functional development is finished, and nothing gets added under the
heading of an implied next phase. Do not add retrieval techniques, providers, endpoints or
infrastructure without being asked. Any behavioural change to a released system is an explicit,
versioned change — proposed, scoped and agreed first, never slipped in as continuation of the
roadmap. If a change is only "this could be prettier", record it rather than making it.

## Commands

```bash
uv sync                       # install (dev tooling included by default)
uv run uvicorn --app-dir src portfolio_rag.main:app --reload --port 8000
uv run portfolio-rag knowledge validate      # ingest the knowledge base, report problems
uv run portfolio-rag knowledge inspect <id>  # what ingestion made of one document
uv run portfolio-rag knowledge chunks <id>   # where that document is cut (--show-content, --all)
uv run portfolio-rag knowledge embedding <id>  # what would be embedded (--show-text)
uv run portfolio-rag knowledge index --dry-run # what indexing would change
uv run portfolio-rag query retrieve "…"      # what a question finds (--top-k, --min-similarity)
uv run portfolio-rag query answer "…"        # the whole pipeline (--show-retrieval, --show-context)
uv run portfolio-rag eval run                # measure retrieval + grounding (--retrieval-only)
uv run portfolio-rag eval run --generation-delay-seconds 8  # pace a live run inside a rate limit
uv run pytest                 # all fast, no network, no credentials
uv run ruff check .
uv run ruff format .
uv run mypy                   # strict
```

If the console script cannot import the package, run it from the source tree:
`PYTHONPATH=src uv run python -m portfolio_rag knowledge validate`.

The full gate, which is also what CI runs:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

## Where things are

| Path | Contents |
| --- | --- |
| `src/portfolio_rag/main.py` | ASGI app factory — the HTTP wiring point |
| `src/portfolio_rag/cli.py` | developer CLI — the terminal entry point |
| `src/portfolio_rag/api/` | routes, schemas, middleware, error handlers |
| `src/portfolio_rag/ingestion/` | knowledge pipeline; `loader.py` is the public entry point |
| `src/portfolio_rag/ingestion/chunking/` | document → chunks; `chunker.py` is the public entry point |
| `src/portfolio_rag/ingestion/embedding.py` | the embedding representation and its fingerprint |
| `src/portfolio_rag/application/indexing/` | desired state, index plan, convergent synchronization |
| `src/portfolio_rag/rag/` | the query side: query, policy, retrieval, context, prompt, generation, citations, service |
| `src/portfolio_rag/evaluation/` | dataset model, retrieval metrics, evaluation runners |
| `evaluation/` | the question set, its fixture corpus, and what the measurements settled |
| `src/portfolio_rag/infrastructure/` | adapters: deterministic/Mistral embeddings and generation, memory/Vectorize stores, corpus snapshot |
| `src/portfolio_rag/composition.py` | the only place settings become concrete adapters |
| `src/portfolio_rag/core/` | config, error taxonomy, logging, request context |
| `src/portfolio_rag/domain/` | knowledge, embedding and retrieval models |
| `src/portfolio_rag/ports/` | `LLMProvider`, `EmbeddingProvider`, `VectorStore`, `ChunkResolver`, port errors |
| `tests/unit/`, `tests/integration/` | fast unit tests; HTTP, indexing, the full RAG flow, adversarial input, the failure matrix, CLI |
| `tests/evaluation/` | the dataset as a regression guard, with real metrics |
| `tests/contracts/` | reusable behavioural contracts every adapter must satisfy |
| `tests/doubles.py` | scripted/lexical test adapters, for branches a real provider cannot take |
| `tests/live/` | opt-in tests against real providers; skipped without credentials |
| `tests/fixtures/knowledge/` | neutral fixture corpora: `valid/`, `invalid/`, `rag/` |
| `knowledge/` | knowledge document standard + template (no content) |
| `docs/`, `docs/adr/` | architecture, roadmap, decision records |

## Read before changing architecture

1. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — layers, dependency direction, what exists today
2. [docs/adr/](docs/adr/) — 0001 modular monolith, 0002 provider-agnostic core, 0003 portable
   runtime, 0004 versioned Markdown knowledge format, 0005 structure-aware chunking,
   0006 versioned embeddings + incremental indexing, 0007 grounded retrieval + backend-owned
   citations, 0008 abuse boundary at the deployment edge
3. [docs/ROADMAP.md](docs/ROADMAP.md) — six phases, and what is deliberately not being built
4. [docs/SECURITY.md](docs/SECURITY.md) — which boundaries are structural, and what remains a risk
5. [AGENTS.md](AGENTS.md) — the rules
6. [knowledge/README.md](knowledge/README.md) — before touching anything about the document format

## The rules that get broken most often

1. No provider SDK outside `infrastructure`; no Cloudflare specifics in the application core.
2. No package, abstraction or config field without a caller.
3. Never log chat messages, prompts, retrieved passages or request bodies.
4. Never invent knowledge content, benchmarks or metrics.
5. OpenAPI documents only what the code can actually produce.
6. **Only `ingestion/loader.py` opens a knowledge file, parses YAML or handles encodings.** The
   chunker takes a `KnowledgeDocument`, never a path, and stays a pure function.
7. YAML is parsed only through a `SafeLoader` subclass — never a non-safe loader. Duplicate keys are
   rejected.
8. Ingestion normalizes encoding, line endings and Unicode composition. It never rewrites prose.
9. **`chunk.content` is source text.** The embedding representation and the context representation
   are composed elsewhere, versioned separately, and never baked into the corpus.
10. Chunk boundaries are deterministic and never exceed `max_chars`. An oversized code fence is an
   error, not a cut.
11. **An embedding provider only ever sees the representation**: title, heading path, content. No
   visibility, no trust level, no paths, no fingerprints, no ids.
12. Embedding spaces are compared by `EmbeddingSpec` equality. Equal dimensionality is *not*
   compatibility — for indexing *and* for a query vector.
13. The embedding fingerprint decides re-embedding. A metadata-only change must never cost a
   provider call.
14. **Public retrieval is structural.** `visibility=public` is built into the query the retrieval
   service sends. It is never a parameter, a policy field or a flag a caller can pass, forget or
   invert.
15. **Citation truth belongs to the backend.** A model may select a source label; it may never name
   a document, path or URL and have that published. Unknown labels are dropped, never rendered.
16. **Weak retrieval short-circuits.** Nothing above the threshold means an honest answer and no
   provider call. A substantive answer with no verifiable source is not published either.
17. No world-knowledge fallback. If the corpus does not support an answer, the system says so.
18. Similarity is a similarity — never a confidence, never a percentage, in code, CLI or docs.
19. No provider logic in `rag/`. It works against ports and knows no vendor's name.
20. Provider and vector store are independent — any combination must work, and `composition.py` is
   the only place any of them is chosen.
21. Development stand-ins (`deterministic` embedding and LLM adapters) must never be selectable in
   a production environment.
22. Standard tests and CI never need a credential or a network. Live tests are opt-in and skipped.
23. Retrieval parameters are **provider-specific**. `min_similarity` measured against one embedding
   model means nothing under another — change it only with a measurement from `eval run`, never to
   make a benchmark look better.
24. Rate limiting belongs at the deployment edge, not in this process ([ADR 0008](docs/adr/0008-abuse-boundary-at-the-edge.md)).

## Project skills

In `.claude/skills/`, to be used when the situation matches:

| Skill | Use when |
| --- | --- |
| `architecture-review` | structure, boundaries or dependencies changed |
| `api-review` | routes, schemas, status codes or the error contract changed |
| `security-review` | input handling, logging, CORS, configuration, retrieval or error paths changed |
| `test-gate` | before declaring any substantial change finished |
| `phase-review` | a roadmap phase is claimed to be complete |

A dedicated `rag-review` skill was considered twice and not added: the four above cover the
boundaries, and the recurring RAG question — "is retrieval any good?" — is now answered by running
`portfolio-rag eval run`, which is a command rather than a checklist.
