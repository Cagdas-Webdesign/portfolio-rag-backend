# Engineering rules

Applies to everyone working in this repository — humans, Claude Code, Codex, anything else.
These are the rules that are expensive to discover by breaking them.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing structure, and
[docs/ROADMAP.md](docs/ROADMAP.md) before adding a feature.

---

## 1. Architecture

* **Respect the layer boundaries.** Dependencies point inwards:
  `api → application → domain ← ports ← infrastructure`, with `core` beneath all of them.
  `domain` and `ports` may use Pydantic for validated data models, but never a web framework, a
  provider SDK, a platform binding or a database driver. `core` imports no other layer.
  See "What 'no framework in the core' means" in `docs/ARCHITECTURE.md`.
* **A module needs a caller.** Do not create packages, base classes, interfaces or config fields
  "for later". An empty package is decoration, not architecture — create it in the phase that fills
  it.
* **Business logic does not live in route handlers.** Routes validate, delegate and serialize.
* **Change the architecture deliberately.** A structural change means updating
  `docs/ARCHITECTURE.md` in the same change, and writing an ADR (`docs/adr/`) when the decision
  would be expensive to reverse. Contradicting an existing ADR means superseding it, not ignoring
  it.

## 2. Providers and the cloud

* **Never import a provider SDK outside `infrastructure`.** No Mistral, OpenAI, Workers AI,
  Vectorize, D1 or R2 call in `api`, `application`, `rag`, `domain`, `ports` or `core`.
  See [ADR 0002](docs/adr/0002-provider-agnostic-core.md).
* **`composition.py` is the only place a setting becomes an adapter.** Provider and store selection
  happens once, explicitly. No dynamic imports from a configuration string, no DI container.
* **An embedding provider receives finished text and composes nothing.** What gets embedded is
  decided in `ingestion/embedding.py` for a chunk and in `rag/query.py` for a question, is versioned
  in both cases, and contains no internal metadata. That is a privacy boundary, not a formatting
  preference. See [ADR 0006](docs/adr/0006-versioned-embeddings-and-incremental-indexing.md).
* **A generation provider receives the grounding instructions, the selected public passages and the
  question.** Nothing else leaves this process: not the corpus, not internal documents, not
  fingerprints, source paths, chunk ids or vectors.
* **Embedding spaces are compared by identity, never by dimensionality.** Two models with the same
  output length produce vectors that are not comparable.
* **Keep the RAG core free of Cloudflare.** Cloudflare is a deployment target reached through
  adapters. The application must keep running under plain Uvicorn and in Docker.
  See [ADR 0003](docs/adr/0003-fastapi-portable-runtime.md).
* **Ports stay narrow, and are added when they have a caller.**
* **`rag/` names no vendor.** The query side works against ports only. A provider-shaped type
  anywhere in it — a Mistral response model, an HTTP status, a `MistralRAGService` — is a bug.
* **Development stand-ins never run in production.** The deterministic embedding provider and the
  deterministic generation stub are refused by the composition root when the environment is
  `production`. A public page answering with a stub is worse than a page that fails to start.

## 3. Dependencies

* **The budget is €0.** No paid API, no paid infrastructure, no service that requires a credit card,
  no paid SaaS. This is a technical requirement, not a preference.
* **Standard tests and CI run offline.** No credential, no network, no cost — the deterministic
  embedding provider and the in-memory vector store exist so this stays true. Tests against real
  services are opt-in and skip when their credentials are absent; they never fail a build.
* **Adding a dependency requires a reason** that is written down in the change: what problem it
  solves and why the standard library or existing dependencies do not.
* **No RAG frameworks.** LangChain, LlamaIndex, Haystack and equivalents are out. Chunking,
  retrieval, ranking, context building, grounding, citations and evaluation are implemented here on
  purpose — that is where answer quality is decided. Narrow libraries that solve one well-defined
  problem are fine.
* **No infrastructure without a use case.** No Redis, no Kafka, no Kubernetes, no message queue, no
  event system.

## 4. Retrieval, grounding and citations

Phase 5 rules. They are here rather than in a module docstring because breaking one is a security
or honesty failure, not a style one. See
[ADR 0007](docs/adr/0007-grounded-retrieval-and-backend-owned-citations.md).

* **Public retrieval is enforced by construction.** `visibility=public` is part of the query the
  retrieval service builds. Never a parameter, a policy field, a keyword argument or a flag. A
  caller must not be able to request internal knowledge, correctly or by accident.
* **Citation truth belongs to the backend.** A model may select a source label. It may never name a
  document, a section, a path or a URL and have that published. Every citation is built from a
  chunk that was actually retrieved; unknown labels are dropped, never rendered.
* **Weak retrieval short-circuits generation.** Nothing above the threshold means an honest answer
  and no provider call — a model handed no evidence and asked a question is being invited to invent
  one. A substantive answer citing nothing verifiable is not published either.
* **No world-knowledge fallback.** When the corpus does not support an answer, the system says so.
  The authority in this system is the retrieved corpus.
* **Knowledge is data, never instruction.** Instructions are a system message; retrieved passages
  and the question are a user message. No configuration or document content may move a passage into
  the instruction role.
* **Similarity is a similarity.** Never a confidence, never a probability, never a percentage — in
  code, in the CLI, in logs or in documentation.
* **Nothing is silently truncated.** A passage fits the context budget whole or is left out and
  counted.
* **Never claim the system is hallucination-free.** It is grounded, which is a description of a
  process and reduces the room a model has to invent. It is not a correctness guarantee.
* **Retrieval parameters are provider-specific and are changed only with a measurement.**
  `min_similarity` and `top_k` mean different things under different embedding models. Change one
  because `portfolio-rag eval run` showed a problem, never to make a benchmark greener — and record
  the sweep that justified it.
* **Rate limiting belongs at the deployment edge.** An in-process limiter cannot identify a client
  correctly without knowing the proxy topology, and one that is wrong looks like protection while
  providing none. See [ADR 0008](docs/adr/0008-abuse-boundary-at-the-edge.md).

## 5. Security and privacy

* **No secrets in the repository.** Not in code, not in tests, not in `.env.example`, not in a
  comment, not in a fixture. Configuration comes from the environment.
* **Validate all external input** with Pydantic, including bounds. Unknown fields are rejected, not
  ignored.
* **Never log chat messages, prompts, retrieved passages, answers, query strings or request
  bodies.** Log identifiers, counts, sizes and durations. When a phase genuinely needs real queries
  (evaluation), make that an explicit, documented, opt-in data flow.
* **The public API returns answers and citations.** Not scores, labels, context, prompts, vector
  ids, fingerprints, embedding spaces, timings or provider names. The CLI and the logs are where
  developer diagnostics live.
* **Error responses expose nothing internal** — no exception text, no stack traces, no paths, no
  provider payloads. Everything goes through the shared error envelope.
* **CORS stays explicit.** No wildcards. The configuration validator rejects them.

## 6. Knowledge content

* **Never invent knowledge documents.** No fabricated skills, projects, experience, dates,
  employers or claims. The knowledge base states facts about a real person; those come from that
  person only.
* **Never invent numbers.** No benchmarks, latencies, accuracy figures or user counts unless they
  were actually measured, with the method recorded.
* Templates and format examples are fine, and must be obviously non-content.

## 7. Working method

* **Prefer small, reviewable changes** with a clear reason over large rewrites.
* **Test new core logic.** Anything in `domain`, `ports`, `application`, `rag` or `core` gets unit
  tests; anything that changes HTTP behaviour gets an integration test. Do not write tests whose
  only purpose is raising a coverage number — a golden RAG flow, a leakage test and a citation
  integrity test are worth more than fifty assertions about getters.
* **Run the checks before and after a substantial change:**
  ```bash
  uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
  ```
* **Do not break the public API without being asked.** `/health` and `/api/v1/*` are contracts. A
  breaking change means a new version prefix.
* **Keep OpenAPI honest.** Never document a response the code cannot produce. If a schema exists but
  is not returned yet, keep it out of the spec.
* **Documentation describes what exists.** Planned work is marked as planned, in the roadmap.

## 8. What not to do without being asked

* No deployment, and no provisioning of cloud resources (no Vectorize index, D1 database, R2 bucket,
  Worker or hosted service).
* No commits, no pushes, no remotes, no releases.
* No changes to the existing portfolio application — it is a separate repository, and integrating
  it is not part of the current plan.
* No implementing the next phase because the current one finished early. Finish the phase, report,
  stop.

## 9. Priorities

When two of these conflict, the earlier one wins:

1. Correct and honest over impressive.
2. Answer quality over feature count — a better answer beats another endpoint.
3. Simple over clever.
4. Tested over assumed.
5. Portable over vendor-locked.
6. Measured over asserted.
7. An honest refusal over a confident invention.
