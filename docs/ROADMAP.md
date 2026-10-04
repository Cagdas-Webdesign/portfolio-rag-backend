# Roadmap

Six phases: five that build the product, and one that gets it ready to be seen.
Each has one objective, a short list of deliverables, and a definition of done that is checkable
rather than aspirational.

Two rules govern the sequence:

* **No phase starts before the previous one is done.** Phases build on each other; a half-finished
  retrieval layer makes the generation phase debug the wrong thing.
* **Nothing from a later phase is implemented early.** Speculative work is the main way an
  architecture rots before it is used.

**All six phases are complete, and v1.0.0 is deployed.** Phase 6 closed the plan: the pipeline is
measured, adversarially tested, hardened where measurement justified it, and taken to production —
a Docker container on Google Cloud Run behind a Cloudflare Worker edge gateway
([DEPLOYMENT.md](DEPLOYMENT.md)). What follows is not a seventh phase: further behavioural change is
an explicit, scoped, versioned change rather than a continuation of this plan.

Cost constraint across all phases: €0. Any phase that would require paid infrastructure gets a free
alternative or is redesigned.

> **This plan replaced an earlier eleven-phase one.** Retrieval, context building, generation,
> grounding and citations were separate phases in that version; they are one feature phase here,
> because they are one pipeline and splitting them would have meant shipping four halves of an
> answer. Persistence, conversation state and a standalone evaluation phase are gone entirely —
> see "Deliberately not in the plan" below. There is no Phase 7.

---

## Phase 1 — Foundation & Architecture ✅ complete

**Objective.** A runnable, typed, tested FastAPI application whose boundaries the later RAG phases
can be built into without an architectural restart.

**Deliverables** — all delivered

* Python project with `uv`, `pyproject.toml`, a lockfile, Ruff, mypy (strict), pytest
* FastAPI application: `GET /health`, `POST /api/v1/chat` as a validating `501` stub
* Consistent error envelope with machine-readable codes; no internal detail in responses
* Request id (accepted, validated, generated, logged, returned) and structured logging
* Typed configuration from the environment; controlled CORS
* Domain models and provider ports (`LLMProvider`, `EmbeddingProvider`, `VectorStore`)
* Knowledge document standard (Markdown + frontmatter), documented, no content
* `ARCHITECTURE.md`, ADRs 0001–0003, `AGENTS.md`, `CLAUDE.md`, README
* Dockerfile, GitHub Actions CI

**Definition of done** — all met

* `ruff check`, `ruff format --check`, `mypy` and `pytest` all pass ✅
* The app starts under Uvicorn; `/health` answers; `/openapi.json` reflects exactly what exists ✅
* The Docker image builds and answers `/health` ✅
* No secrets, no paid infrastructure, no provisioned cloud resources, no LLM call ✅

---

## Phase 2 — Reliable Knowledge Ingestion ✅ complete

**Objective.** Turn `knowledge/**/*.md` into validated `KnowledgeDocument` objects — deterministic,
traceable, and strict about what it accepts.

**Deliverables** — all delivered

* Recursive, deterministic source discovery with documented exclusion rules
* Frontmatter extraction and YAML parsing, `safe_load` only
* Schema version 1, declared per document and enforced ([ADR 0004](adr/0004-versioned-markdown-knowledge-format.md))
* Strict metadata validation: types, enums, patterns, bounds, no unknown keys
* Normalization: UTF-8, BOM, CRLF/LF, Unicode NFC — never the prose itself
* Provenance: relative source path and a canonical SHA-256 over metadata and body
* Duplicate document id detection across the whole base
* `collect_knowledge_base` (batch, collects issues) and `load_knowledge_base` (strict, raises)
* `portfolio-rag knowledge validate` and `portfolio-rag knowledge inspect`, with real exit codes

**Definition of done** — all met

* A directory of documents parses into validated objects; a malformed document fails loudly and
  names the file and field ✅
* Ingesting the same input twice produces identical output, including hashes ✅
* Tests cover valid documents, every failure mode, and unicode/CRLF/BOM edge cases ✅
* Hostile YAML cannot execute code ✅
* No embedding, no chunking, no storage, no HTTP surface ✅

**Deliberately deferred.** Non-Markdown sources and semantic duplicate detection.

---

## Phase 3 — Structure-Aware Chunking & Chunk Provenance ✅ complete

**Objective.** Split documents into retrieval units that make sense on their own, deterministically,
without losing structure, meaning or provenance.

Chunking belongs to `portfolio_rag.ingestion`, not to `rag`: it is a property of the corpus, decided
once when a document is ingested, not re-decided on every query. Its input is a `KnowledgeDocument`
— never a file path.

**Deliverables** — all delivered

* Markdown structure analysis via markdown-it-py: headings and blocks, chunks sliced from the source
  by token line ranges (never rendered)
* Heading hierarchy as a `heading_path`, with a documented stack rule for level jumps
* Block-aware packing: paragraphs, lists, quotes and fences packed whole while the budget allows
* `ChunkingPolicy` — provider-neutral character budget, validated, immutable
* Fallback ladder for oversized blocks: lines → sentences → whitespace → hard cut
* Atomic code fences; `UNSPLITTABLE_BLOCK` rather than broken Markdown
* Controlled overlap inside a split section only, snapped to clean boundaries
* Stable identity: `document-id--0000` plus a SHA-256 chunk fingerprint
* Strategy version `markdown-structure-v1`, recorded on every chunk
* `chunk_document` and `chunk_knowledge_base`, both pure
* `portfolio-rag knowledge chunks` with `--show-content`, `--all` and policy overrides

**Definition of done** — all met

* Chunks respect headings and block boundaries; no chunk exceeds `max_chars` ✅
* The same document and policy produce the same chunks, ids and fingerprints every run ✅
* No source content is silently lost — asserted across every document shape and policy ✅
* Golden boundary tests pin exact chunk contents, heading paths, ordinals and ids ✅
* Chunking touches no filesystem, no network, no provider ✅

**Deliberately deferred.** Character offsets, token-aware chunk sizing, and any second chunking
strategy — one implementation, no registry, until a second one earns it.

---

## Phase 4 — Embeddings & Vector Indexing ✅ complete

**Objective.** Turn retrieval-ready chunks into versioned embeddings and synchronize them safely
into a compatible vector index — incrementally, provider-agnostically, and testably offline.

**Deliverables** — all delivered

* Versioned embedding representation `embedding-text-v1`: document title, heading path, chunk
  content. Separate from `chunk.content`, and deliberately free of internal metadata
* `EmbeddingSpec` — provider, model, dimensions, representation version — as the identity of an
  embedding space; equal dimensionality is explicitly *not* compatibility
* Embedding fingerprint: the third and last level of the change-detection model
* `EmbeddingProvider` port with batch embedding and stable input/result mapping
* `DeterministicEmbeddingProvider` — offline, free, reproducible across processes, not a mock
* `MistralEmbeddingProvider` — real HTTP adapter with batching, bounded retries, timeouts and full
  response validation
* `VectorStore` port: upsert, fetch, fetch states, list state, delete, query — with enumeration
  modelled as a capability
* `InMemoryVectorStore` — the strict reference implementation, plus a reusable contract suite
* `CloudflareVectorizeStore` — real REST adapter against a pre-existing index
* `application/indexing`: desired state, index plan, convergent synchronization
* `portfolio-rag knowledge embedding` and `portfolio-rag knowledge index [--dry-run] [--rebuild]`

**Definition of done** — all met

* A second identical run is a true no-op: zero embeddings, zero writes, zero deletes ✅
* A metadata-only change updates the record without calling the provider ✅
* A content, title or heading change re-embeds exactly the affected chunks ✅
* Removed chunks leave no stale vectors behind ✅
* Incompatible embedding spaces cannot be mixed, and dimension mismatches fail before any write ✅
* The whole pipeline runs and is tested with no credentials, no network and no cost ✅
* Live Mistral and Vectorize tests are opt-in and skipped by default ✅

**Deliberately deferred.** A second embedding representation — version 2 is for when evaluation has
something to say about version 1.

---

## Phase 5 — Retrieval & Grounded RAG ✅ complete

**Objective.** A public user asks a question and gets a useful answer, grounded only in retrievable
public knowledge, with traceable sources.

This is the last feature phase. Retrieval, context building, generation, grounding and citations are
one pipeline, and they were built as one.

**Deliverables** — all delivered

* Query input boundary: bounded length, non-empty after normalization, Unicode-safe, one limit
  shared by the HTTP schema and the application
* Deterministic, versioned query representation `query-text-v1` — the normalized question, with no
  prefix, keyword stuffing, rewriting or expansion
* Query embedding through the existing `EmbeddingProvider`; no second port, no fabricated chunk
* `PublicRetrievalService`: `visibility=public` built into the query it sends, with no parameter,
  policy field or override that can turn it off
* Embedding-space compatibility checked by identity before any search
* `RetrievalPolicy` — `top_k` and a minimum similarity, starting values at the time, both central
* `ChunkResolver` port and an in-process corpus snapshot, so a match becomes real text without
  duplicating the corpus into vector metadata
* Deterministic ordering: similarity descending, chunk id as the tie-break
* Insufficient-retrieval short circuit: nothing found means an honest answer and no provider call
* `GroundedContext` — bounded, deduplicated, labelled `S1…Sn`, deterministic, never truncated
* Token-aware context budget over instructions, context, question and an output reserve
* Grounded prompt `grounded-answer-v3` with structural role separation and a JSON answer contract
* `LLMProvider` port with a provider-neutral request, response and JSON-output capability
* `MistralChatProvider` — real HTTP adapter with retries, timeouts and full response validation
* `DeterministicLLMProvider` — an offline development stub that generates nothing and says so
* Backend-owned citation validation: unknown labels dropped, duplicates collapsed, every citation
  built from a chunk that was actually retrieved
* `GroundedAnswerService`, and `POST /api/v1/chat` answering through it
* `portfolio-rag query retrieve` and `portfolio-rag query answer`
* [ADR 0007](adr/0007-grounded-retrieval-and-backend-owned-citations.md)

**Definition of done** — all met

* A question the corpus supports is answered with citations that resolve to real documents ✅
* `internal` documents never appear in results, context, prompts, answers or logs for a public
  request — covered by tests at the service, pipeline and HTTP levels ✅
* A question the corpus does not cover produces an honest refusal with no citations and no provider
  call — not an invention, and not a server error ✅
* An answer that cites nothing the backend can verify is never published ✅
* A source label a model invented never becomes a citation ✅
* Retrieval is inspectable: scores, ranks, headings and provenance are visible from the CLI ✅
* The whole pipeline — ingestion to answer — runs and is tested with no credentials, no network
  and no cost ✅
* Provider and store failures produce the standard error envelope, never a stack trace, a provider
  payload or a partial answer ✅
* `POST /api/v1/chat` returns `200` with `ChatResponse`; the `501` is gone and OpenAPI says so ✅

**Deliberately deferred.** Re-ranking, hybrid and lexical search, query rewriting and expansion,
HyDE and multi-query retrieval. Each is a plausible improvement to a pipeline whose quality has not
been measured yet; adding one now would be a fix for a problem nobody has observed.

---

## Phase 6 — Final Engineering & Production Readiness ✅ complete

**Objective.** Make what exists trustworthy enough to put in front of the public internet, and
finished enough to show.

**Not a feature phase.** Phase 6 adds no capability to the product. If something here turns into a
new feature, it belongs in a later plan, not in this one.

**Deliverables** — all delivered

* `evaluation/`: 24 questions in 8 categories with hand-checkable ground truth, a neutral fixture
  corpus, and `portfolio-rag eval run`
* `portfolio_rag.evaluation`: hit@1/@3/@5, MRR, threshold calibration, per-question failures —
  no aggregate score that could hide one
* A measured baseline **before** any tuning, and a threshold and `top_k` sweep
* Adversarial tests for hostile user messages *and* hostile knowledge documents, followed through
  every stage to the HTTP response
* A canary in an `internal` fixture, asserted absent from retrieval, context, prompt, answer,
  citations and logs — at four thresholds including none
* The failure matrix as executable tests: every stage, every fault, the status and code a client sees
* Context budget and cost-aware behaviour measured with real counts
* Secret, logging, dependency and repository-hygiene audits
* A fresh-environment gate: the full suite from the lockfile alone, in a clean container
* [SECURITY.md](SECURITY.md), [DEPLOYMENT.md](DEPLOYMENT.md),
  [ADR 0008](adr/0008-abuse-boundary-at-the-edge.md)

**Definition of done** — all met

* Retrieval and grounding are measured, reproducible, and separated rather than averaged ✅
* Tuning happened only where data justified it — and the data justified **no parameter change**,
  which is recorded with the sweep that showed why ✅
* Prompt injection is tested from both directions, and the structural boundaries hold even against
  a model scripted to have obeyed the injection ✅
* No request of any shape reached `internal` content — 0 leaks at every threshold tested ✅
* Error responses leak nothing; every logged field was enumerated and reviewed ✅
* The abuse boundary is decided, justified and documented rather than faked in-process ✅
* Running costs are still €0, and the offline suite still needs no credential ✅

**Deliberately not done.** No re-ranking, hybrid search or query rewriting: the evaluation measured
a threshold problem, not a ranking problem, and inventing a second retrieval stage for it would
have been a fix for an unobserved cause. No tokenizer dependency: peak context utilisation was
12.3%, so the approximation would have to be wrong by eightfold to matter. No in-process rate
limiter: it cannot be written correctly without knowing the proxy topology.

---

## Deliberately not in the plan

Named here so that "we should also build…" has an answer that is already written down.

| Not building | Why |
| --- | --- |
| Conversation memory, chat history, a conversation database | The assistant answers one question at a time from a fixed corpus. A client may send the last few turns with a question so a follow-up can be read (`rag/conversation.py`); they are never evidence and nothing is stored. Server-side multi-turn state is a product decision nobody has made, and a database is a lot of machinery to add on a guess. |
| User accounts, authentication | Nothing here is per-user. There is nothing to protect that a public corpus does not already publish. |
| An agent framework, tool calling, MCP runtime, web search | The authority in this system is the retrieved corpus. Letting a model reach past it would change what the product *is*. |
| Re-ranking, hybrid/BM25 search, query rewriting, HyDE | Real techniques. Phase 6 measured retrieval and found the limiting factor to be threshold calibration under a specific embedding model, not ranking — so none of them is justified yet. |
| Streaming responses | A latency improvement for answers that are short. Worth revisiting if the numbers say so. |
| Redis, Kafka, queues, Kubernetes, background workers | No use case. See `AGENTS.md`: no infrastructure without one. |
| A RAG framework (LangChain, LlamaIndex, Haystack) | Chunking, retrieval, context and grounding are where answer quality is decided. They are written here on purpose. |
