# Architecture

Status of this document: describes the architecture as it is, and marks precisely what exists today.
Everything not marked **implemented** is a plan, not a promise.

---

## 1. System context

```
┌──────────────────────┐        ┌──────────────────────┐
│ React/Vite portfolio │        │ future clients       │
│ (existing chat UI)   │        │ (CLI, other sites)   │
└──────────┬───────────┘        └──────────┬───────────┘
           │  HTTPS + JSON, public API only │
           └───────────────┬────────────────┘
                           ▼
              ┌────────────────────────────┐
              │  FastAPI application       │  ← the deployable unit
              │  ┌──────────────────────┐  │
              │  │ api (transport)      │  │
              │  ├──────────────────────┤  │
              │  │ rag (query side)     │  │
              │  │ application (corpus) │  │
              │  ├──────────────────────┤  │
              │  │ domain · ports       │  │
              │  ├──────────────────────┤  │
              │  │ infrastructure       │  │
              │  └──────────────────────┘  │
              └───────┬─────────┬──────────┘
                      │         │
          ┌───────────▼──┐   ┌──▼─────────────┐
          │ LLM provider │   │ vector store   │
          │ (stub,       │   │ (in-memory or  │
          │  Workers AI  │   │  Vectorize)    │
          │  or Mistral) │   │                │
          └──────────────┘   └────────────────┘
```

**The client never reaches past the API.** It has no address for the vector store, the embedding
provider, the LLM provider, a database or object storage — and never gets one. Everything a client
can do is expressed in the public REST API, which means credentials, retrieval policy and
grounding rules stay on the server where they can be enforced.

**Today** this service is deployed, and the portfolio is its client — reaching it through the
Cloudflare Worker edge gateway rather than directly. The portfolio remains a separate repository:
no frontend source, frontend architecture or frontend roadmap belongs here.

## 2. Layers and their responsibilities

Dependencies point inwards, without exception:

```
api ──▶ application ──▶ domain ◀── ports ◀── infrastructure
 │           │                                    ▲
 └───────────┴────────────── core ────────────────┘
```

| Layer | Owns | Must not |
| --- | --- | --- |
| `api` | HTTP: routing, status codes, headers, request/response schemas, the error envelope, CORS, correlation ids | contain business rules; be imported by anything but `main` |
| `application` | corpus-side use cases: orchestrating several ports. Today: indexing | know about HTTP or about a concrete adapter |
| `rag` | the **query side**: query boundary, retrieval, context building, prompting, grounding, citations, and the orchestrator that joins them | perform I/O directly — it works through ports; produce chunks (that is ingestion's job) |
| `domain` | the shared vocabulary: `KnowledgeDocument`, `KnowledgeChunk`, `RetrievedChunk`, `SourceCitation` | import a web framework, a provider SDK, or any other layer except `core` |
| `ports` | the contracts: `LLMProvider`, `EmbeddingProvider`, `VectorStore` | contain implementations |
| `infrastructure` | adapters implementing ports: deterministic + Mistral embeddings, in-memory + Vectorize stores | be imported by anything except the composition root |
| `core` | configuration, error taxonomy, logging, request context | import any other layer |
| `ingestion` | the **corpus side**, offline: parsing → normalization → validation → chunking → metadata enrichment → embedding preparation | be reachable from a request handler; open a file outside `loader` |
| `evaluation` | measuring the query side against a dataset with known answers: metrics, runners, per-question failures, run pacing | be part of a request path; judge answer *quality*, which it deliberately does not claim to |
| `cli` | the developer entry point: runs ingestion, prints reports, sets exit codes | contain pipeline logic of its own |
| `main` | the composition root: builds the app, wires adapters into use cases | contain logic of its own |

`evaluation` is the one module that sits across the corpus/query split rather than on one side of
it: it drives `rag` and borrows `ingestion`'s strict YAML loader for its dataset file, because a
second YAML loader is how a non-safe one eventually gets used. It is reached from the CLI only —
nothing in `api` or `main` imports it, and no request path can. `evaluation/acceptance.py` judges a
finished end-to-end export as a release acceptance — from the JSON alone, so the CLI writing the
verdict and the validator re-checking it apply one rule (see [RELEASE_ACCEPTANCE.md](RELEASE_ACCEPTANCE.md)).

**Pacing lives here, not in an adapter.** A dataset run is a burst of generations, which is what a
rate-limited account refuses first; a live request is one generation with somebody waiting for it.
`evaluation/pacing.py` is a decorator over `LLMProvider` that spaces calls out and waits out a
provider's `Retry-After`, wired in by the CLI when `eval run --generation-delay-seconds` asks for
it. Adapters keep their own short, per-request retry budget unchanged and report the wait a
provider asked for; deciding to take a wait measured in tens of seconds belongs to something that
can afford to. Because it decorates the port, a retry re-issues one provider call inside one
question — retrieval is not repeated and no metric counts a question twice.

**Ingestion owns chunking.** The split between `ingestion` and `rag` is corpus side vs. query side,
not "offline vs. clever". Everything that turns source material into indexable units — parsing,
normalization, validation, chunking, metadata enrichment, preparing text for embedding — belongs to
`ingestion`. Everything that happens once a question arrives — query processing, retrieval, ranking,
context building, grounding — belongs to `rag`. Chunking is a property of the corpus, and putting it
on the query side would mean re-deciding it on every request.

**Two packages orchestrate, and the split is corpus versus query.** `application` owns work that
happens once per document: indexing composes a representation, a provider, a store and a change
plan. `rag` owns work that happens once per question. Neither imports the other. They meet in
exactly one place — `composition.py` — where the corpus is loaded, chunked and handed to the query
side as the source of the text behind a retrieved id.

### What "no framework in the core" means

`domain` and `ports` use **Pydantic** for validated data models, deliberately: validation at the
boundary is the point of those models, and re-implementing it by hand would be worse code, not purer
architecture. Pydantic is a data-modelling library with no I/O, no network and no vendor behind it.

What they must never import is anything that ties the core to a *delivery mechanism* or a *vendor*:

* the FastAPI/Starlette HTTP layer
* provider SDKs — Mistral, OpenAI, Anthropic, Workers AI, …
* Cloudflare SDKs and platform bindings
* database drivers and storage SDKs — D1, R2, boto3, SQLAlchemy, …

The test is simple: could this module still be used unchanged if the service stopped being an HTTP
API and stopped using its current providers? For `domain` and `ports`, the answer has to stay yes.

### Why the dependency direction matters

The rule that `domain` and `ports` import nothing outward is what makes the two big swaps cheap:
replacing a provider and replacing a deployment target. If retrieval code imported a vendor SDK,
both swaps would become rewrites.

## 3. Runtime flow

What happens to a question, end to end.

```
 client request
   → HTTPS / CORS                    ✅ implemented
   → request id + access log         ✅ implemented   (api/middleware)
   → schema validation               ✅ implemented   (api/schemas)
   → rate limiting / abuse           ◻ at the edge   (ADR 0008 — deliberately not in-process)
   → query boundary                  ✅ implemented   (rag/query.py — bounded, normalized)
   → query representation            ✅ implemented   (`query-text-v1`, deterministic)
   → query embedding                 ✅ implemented   (the corpus's own EmbeddingProvider)
   → space compatibility check       ✅ implemented   (by identity, not dimensionality)
   → public vector search            ✅ implemented   (visibility=public, structurally)
   → similarity threshold            ✅ implemented   (weak matches are not evidence)
   → chunk resolution                ✅ implemented   (ids → real corpus text)
   → [nothing found] ────────────────────────────────▶ honest answer, no provider call
   → context building                ✅ implemented   (bounded, labelled S1…Sn, deterministic)
   → earlier turns (optional)        ✅ implemented   (rag/conversation.py — bounded, prompt only,
                                                       never embedded, never evidence or a citation)
   → grounded prompt                 ✅ implemented   (`grounded-answer-v3`, roles separated)
   → generation                      ✅ implemented   (LLMProvider port)
   → answer contract parsing         ✅ implemented   (structured JSON, not regex)
   → citation validation             ✅ implemented   (backend-owned, unknown labels dropped)
   → [nothing verifiable] ───────────────────────────▶ honest answer, no unproven claims
   → ChatResponse                    ✅ 200
```

Both dead ends are successes. A knowledge base that does not cover a question, and a model that
cannot ground an answer in what it was given, produce the same fixed statement with no citations and
a `200`. Neither is a server error, and neither is allowed to become a plausible invention. There is
no fallback to the model's own knowledge: the authority in this system is the retrieved corpus.

The statement is fixed **per language** (`rag/language.py`): German or English, chosen from the
question, English when nothing indicates otherwise. That module picks between two sentences the
backend wrote and does nothing else — it cannot reach a provider, change what is retrieved, or open
a third path out of the pipeline above.

See [ADR 0007](adr/0007-grounded-retrieval-and-backend-owned-citations.md).

## 4. Ingestion flow

Ingestion is offline and separate from the request path — a slow, expensive, occasionally failing
pipeline must never sit inside an HTTP handler. It has no HTTP surface at all: it is reached through
the CLI and, later, through code that runs before serving.

```
knowledge/**/*.md
   → discovery        which files are documents at all; stable ordering  ✅ implemented
   → reading          bytes off disk, once per file                      ✅ implemented
   → decoding         UTF-8, BOM, CRLF/LF, Unicode NFC                   ✅ implemented
   → frontmatter      split the YAML block from the Markdown body        ✅ implemented
   → validation       schema version, then every authored field          ✅ implemented
   → normalization    trim file artifacts, never touch the prose         ✅ implemented
   → provenance       relative path + canonical SHA-256                  ✅ implemented
   → conflicts        duplicate document ids across the whole base       ✅ implemented
   → KnowledgeDocument[]                                                 ✅ implemented
   → structure        headings and blocks, from a real Markdown parser   ✅ implemented
   → chunking         block-aware packing, size bounding, overlap        ✅ implemented
   → chunk identity   readable id + SHA-256 fingerprint                  ✅ implemented
   → metadata enrich  metadata and provenance carried onto every chunk   ✅ implemented
   → KnowledgeChunk[]                                                    ✅ implemented
   → representation   title + heading path + content, versioned          ✅ implemented
   → embedding id     SHA-256 over representation + embedding space      ✅ implemented
   → index plan       create / re-embed / metadata / unchanged / delete  ✅ implemented
   → embeddings       EmbeddingProvider port, only for what changed      ✅ implemented
   → vector index     VectorStore upsert, then delete                    ✅ implemented
```

### The ingestion boundary

`portfolio_rag.ingestion.loader` is the **only** module that opens a knowledge file, parses YAML,
worries about encodings or resolves a path. Everything downstream receives `KnowledgeDocument`
objects and takes for granted that they validated, are non-empty, carry provenance and have unique
ids.

That boundary is what keeps the chunker honest, and it held: `chunk_document` is a pure function
from a document to chunks, with no filesystem, no `yaml`, no BOM handling and no duplicate-id logic
in it. The two halves are composed at the edge — the CLI calls the loader, then the chunker — and
neither calls the other. If a later module ever needs to reach past the loader, the loader is what
should change.

Two entry points, because two callers want different things:

| | `collect_knowledge_base` | `load_knowledge_base` |
| --- | --- | --- |
| On a bad document | records an issue, continues | raises `KnowledgeBaseError` |
| Returns | every document *and* every problem | documents only |
| For | `knowledge validate` — fixing five files needs five messages | programs — a corpus quietly missing three documents produces confidently incomplete answers |

### From documents to retrieval units

Chunking is the second half of ingestion, and it is where retrieval quality is decided long before
any embedding exists: retrieval never returns a document, it returns a chunk.

```
KnowledgeDocument
  → structure analysis   markdown-it-py parses; chunks are sliced from the source
                         by token line ranges, so content is the author's Markdown
  → heading hierarchy    ('Backend', 'APIs', 'Authentication') — a path, not a label
  → block packing        whole paragraphs, lists, quotes and fences, while budget allows
  → size bounding        soft target, hard maximum, fallback ladder for oversized blocks
  → controlled overlap   inside a split section only
  → identity             `document-id--0000` and a SHA-256 fingerprint
  → KnowledgeChunk[]
```

Four rules govern it ([ADR 0005](adr/0005-deterministic-structure-aware-chunking.md)):

* **A heading is a hard boundary.** Two headings mean the author considered the material separate,
  so blocks from two sections are never packed together to fill a budget. Short sections stay short.
* **The hard ceiling is not negotiable.** `max_chars` is never exceeded — not by packing, not by
  overlap. A block that cannot be divided without producing broken Markdown (an oversized code
  fence) raises `UNSPLITTABLE_BLOCK` instead of being cut.
* **Content is source text.** The heading path, title and metadata sit *beside* the chunk as data,
  never spliced into it. Whether an embedding representation prepends them is a Phase 4 decision,
  and baking it in now would freeze that decision into the corpus.
* **Identity is deterministic.** The id says where a unit sits; the fingerprint says what it
  contains. Neither depends on time, path or machine.

The two hashes work at different levels, deliberately. The **document hash** (Phase 2) covers a
document's metadata and body — it is the signal to re-chunk. The **chunk fingerprint** covers the
strategy, the boundary-relevant policy, the document id, the heading path and the chunk text — it is
the signal to re-embed. The document hash is carried on every chunk for lineage but is not part of
the fingerprint: fixing a typo in section one must not invalidate section three's embedding.

### Design constraints

* **Deterministic.** Same input, same output — same order, same content, same hashes, on any
  machine. Discovery sorts by relative path; nothing volatile enters a document's canonical form,
  and chunking adds nothing volatile either.
* **Strict per document, tolerant per batch.** One malformed file is an error, not a warning; it
  does not stop the other files from being checked.
* **Provenance always.** Every document knows its path and carries a SHA-256 over its normalized
  metadata *and* body. Metadata counts: changing `visibility` changes how retrieval may use the
  document, so it has to count as a change. The path deliberately does not — moving a file should
  not force a re-embed.
* **Content is never rewritten.** Normalization fixes encoding, line endings and Unicode
  composition. It does not touch a single character of the author's prose.
* **Re-runnable.** Index synchronization is an upsert, not an append. A second identical run is a
  true no-op, and document and embedding fingerprints limit work to changed records.
* **Inspectable.** `knowledge inspect` shows what the pipeline made of one document and
  `knowledge chunks` shows where it cut, because "the answer is bad" is usually a corpus problem,
  not a model problem — and a boundary you cannot see is a boundary you cannot fix.
* **Versioned at every derived boundary.** Documents declare `schema_version`; unknown versions
  are refused rather than guessed at
  ([ADR 0004](adr/0004-versioned-markdown-knowledge-format.md)). Chunks record a
  `strategy_version`, so units produced by different algorithms are recognisably not comparable
  ([ADR 0005](adr/0005-deterministic-structure-aware-chunking.md)). The index records the embedding
  space that produced each vector, because vectors from two models are not comparable either.

## 4b. From chunks to a vector index

The last stage of the corpus side, and the first that talks to something outside the process.

```
KnowledgeChunk
  → embedding representation   title + heading path + content  (`embedding-text-v1`)
  → embedding fingerprint      SHA-256 over that text *and* the embedding space
  → desired state              ids, fingerprints and metadata the index should hold
  → index plan                 create / re-embed / metadata-only / unchanged / delete
  → EmbeddingProvider          called only for what actually changed
  → VectorRecord               vector + space + three fingerprints + retrieval metadata
  → VectorStore                upserts first, then deletes
```

### The representation is not the content

`chunk.content` stays source text. What a model sees is composed separately, versioned separately,
and deliberately narrow: **document title, heading path, chunk content**. Internal metadata —
visibility, trust level, source paths, fingerprints, ids, licences, dates — is structured data that
retrieval filters on, and is not embedded.

That separation is also the **privacy boundary**. A real external provider receives exactly the
composed text and nothing else: no source paths, no trust flags, no internal classifications, no
identifiers. One function decides what leaves this process, and it is short enough to read in full.
The deterministic local provider sends nothing anywhere.

### Embedding spaces

An `EmbeddingSpec` is provider + model + dimensions + representation version, and two vectors are
comparable only when all four match. **Equal dimensionality is not compatibility**: two models that
both emit 1024 numbers produce vectors that mean nothing to each other, and an index that mixes them
still answers, still looks healthy, and is wrong with no error anywhere. Both the store and the
indexing service refuse the mix.

### Three fingerprints, three questions

| Fingerprint | Question it answers | Covers |
| --- | --- | --- |
| document (Phase 2) | has this source document changed? | normalized metadata + body |
| chunk (Phase 3) | has this structural retrieval unit changed? | strategy, policy, document id, heading path, content |
| embedding (Phase 4) | does this unit need a new vector *in this space*? | representation text + embedding space |

Deliberately not nested. Fixing a typo in section one changes the document fingerprint and section
one's chunk fingerprint, and leaves section three's embedding fingerprint untouched — so section
three is not re-embedded. A metadata change that never reaches the representation costs a record
rewrite, not a provider call.

### Incremental and convergent

Every run compares desired against actual and produces a plan; a second identical run is a true
no-op, with zero provider calls and zero writes. The index *is* the reuse mechanism — a record
carrying the desired embedding fingerprint already holds the vector — so there is no second cache to
disagree with it.

There is no transaction across a provider and a store, and none is claimed. Upserts happen before
deletes, so a mid-run failure leaves an index that is stale rather than incomplete; the run reports
the failure rather than a partial success, and the next run converges.

One capability is modelled rather than assumed: some stores can fetch by id but cannot enumerate.
Cloudflare Vectorize is one of them, so `supports_enumeration` is part of the port. Against such a
store, planning can still decide creates and updates but cannot discover stale records — and the
CLI says so instead of quietly skipping the check.

### Deliberately not done: character offsets

A chunk could in principle record where in the body it came from. It does not, and that was a
decision rather than an omission: a chunk is assembled from several blocks that are not contiguous
in the source (headings are removed, blocks are rejoined), and overlap repeats text that already
belongs to the previous chunk. Honest offsets would mean a list of ranges per chunk, and nothing
would consume it. Citations need the source path and the heading path, both of which every chunk
already carries. Character-exact highlighting would need them; nothing asks for it, so they can be
derived later against a real requirement.

## 4c. From a question to a grounded answer

The query side, `portfolio_rag.rag`. Where the corpus side runs once per document, this runs once
per question — which is why every step is bounded, deterministic and cheap to explain.

```
question
  → query boundary        non-empty, bounded, NFC, whitespace collapsed
  → representation        the normalized question  (`query-text-v1`)
  → query embedding       the same EmbeddingProvider the corpus used
  → space check           EmbeddingSpec equality, before any search
  → vector search         top-k, filters={"visibility": "public"}
  → threshold             below `min_similarity` is not evidence
  → chunk resolution      ChunkResolver: ids back into real corpus text
  → RetrievedChunk[]      similarity descending, chunk id as tie-break
  → GroundedContext       labelled S1…Sn, deduplicated, budgeted  (`grounded-context-v1`)
  → GenerationRequest     system instructions | knowledge + question  (`grounded-answer-v3`)
  → LLMProvider           through the port; no vendor named on this side
  → answer contract       {"answer": …, "sources": ["S1"]}, parsed strictly
  → citation validation   only labels this context actually contained
  → GroundedAnswer
```

### Public by construction

`PublicRetrievalService` puts `visibility=public` into the query it sends. Not a default, not an
argument, not a policy field — there is no way to express a request for internal knowledge, which is
the difference between a boundary and a filter somebody can forget.

Two further checks belong to the same idea. Resolved chunks are re-checked against the corpus,
because the index filtered on metadata written when the record was indexed and the corpus is the
newer of the two. And the query's embedding space is compared with the index's **by identity**:
equal dimensionality is not compatibility, and an index that mixes spaces answers confidently and
wrongly with no error anywhere.

### The context is not the chunk, and the labels are ours

`chunk.content` stays source text. What a model sees is composed in `rag/context.py`, versioned
there, and pinned by golden tests — the same separation the embedding representation has.

Source labels (`S1`, `S2`, …) are minted per answer and mean nothing outside it. That is what makes
citations checkable: a model can select a label, but it cannot name a document, a path or a URL and
have that believed. Every published citation is built from a `RetrievedChunk` that this request
actually retrieved, and traces back to a chunk id, a document fingerprint and a file in git.

### Bounded, and never truncated

The token budget covers instructions, context, question and an output reserve — not just the
passages, which would overspend by exactly the size of the prompt around them. Token counts are a
documented, deliberately pessimistic **estimate** (`rag/tokens.py`), not a tokenizer: the budget
they feed is itself a conservative ceiling, and paying a per-model dependency to compute a precise
input to an approximate decision is not a good trade.

A passage either fits whole or is left out and counted. Nothing is silently cut: half a paragraph of
evidence reads exactly like whole evidence, to a model and to a reader.

### Two ways to have no answer, both successful

Retrieval finding nothing above the threshold short-circuits before any provider call. A model that
answers but cites nothing verifiable has its answer replaced. Both produce the same fixed statement,
no citations, and `200`. Neither is a server error; neither is allowed to become an invention.

### Where the chunk text comes from

A vector index returns ids, scores and filter metadata — not text
([ADR 0006](adr/0006-versioned-embeddings-and-incremental-indexing.md) chose that deliberately). The
`ChunkResolver` port turns matches back into passages, and its implementation is an in-process
snapshot of `knowledge/`: the same Markdown the index was built from, loaded once at startup.

No second copy of the corpus, and therefore nothing to drift. A chunk id the corpus no longer
contains means the index is out of step with the source tree — which retrieval reports and counts
rather than papering over.

## 5. Storage responsibilities

Two jobs that exist, and one that does not. Conflating them is the mistake this section exists to
prevent.

| | Vector store | Corpus snapshot | Object store |
| --- | --- | --- | --- |
| **Question it answers** | "Which passages resemble this query?" | "What does chunk `x--0003` actually say?" | "Where is the original file?" |
| **Holds** | embeddings + denormalized chunk metadata | the chunks of `knowledge/`, in memory | PDFs, DOCX, uploads |
| **Access pattern** | approximate nearest neighbour + filters | lookup by id | read/write blobs by key |
| **Rebuildable?** | Yes — derived from `knowledge/` | Yes — it *is* `knowledge/` | No — it is the original |
| **Port** | `VectorStore` | `ChunkResolver` | `DocumentStore` (only if needed) |
| **First adapter** | in-memory, Cloudflare Vectorize | `InMemoryChunkResolver` | — |
| **State** | implemented | implemented | not defined |

Because the vector index is *derived*, losing it is an inconvenience, not data loss — it can be
rebuilt from Markdown in git. That property is worth protecting: it is what keeps the vector store
swappable.

**There is no relational store, and none is planned.** A `ConversationStore` would exist to hold
conversation state, and this assistant deliberately keeps none: every question is answered from the
corpus, and the few earlier turns a client may send with it are a request field, not stored state. `DocumentStore` stays undefined for the same reason it always has — no caller.
A port invented before its first caller usually has to be changed by that caller anyway.

## 6. Cross-cutting concerns

**Configuration.** One typed `Settings` object, read from the environment once, validated at
startup. Nothing reads `os.environ` directly. The CORS wildcard is rejected by validation rather
than left to reviewer discipline. An application built with explicit settings uses *those* settings
everywhere — routes, lifespan and logging alike — so one instance can never run on two
configurations.

**Composition.** Exactly one module turns settings into adapters: `portfolio_rag.composition`. That
is why swapping an embedding provider, a generation provider or a vector store is a configuration
change rather than a refactor, and why nothing above that module has to know Mistral or Cloudflare
exist. No dependency-injection framework, and no dynamic imports from a configuration string — a
mistyped setting is a validation error, not an arbitrary module load.

It is also where a dangerous configuration is refused. The deterministic embedding provider and the
deterministic generation stub are development stand-ins: one produces vectors with no meaning, the
other produces no language at all. In a production environment the composition root refuses to build
either, so a deployment that would answer strangers with a stub fails to start instead. Loud beats
plausible.

**Errors.** One envelope for every failure, with a stable machine-readable `code`. Messages are
written for clients; exception text never becomes a response body. Tracebacks go to the log,
correlated by request id. The mapping from code to HTTP status lives in `api`, so `core` stays
transport-agnostic.

Ingestion has its own error taxonomy (`IngestionErrorCode`) rather than extending `AppError`, and
deliberately so: those errors are developer-facing and name files and fields, which is exactly what
a client-facing error may never do. Nothing routes an ingestion error to an HTTP response.

**Observability.** Every request carries an id — reused from `X-Request-ID` when the caller supplies
a well-formed one, generated otherwise, and validated against a strict pattern before it is allowed
anywhere near a log line. Logs are structured (JSON outside local development) and carry the id
automatically via a context variable, so nothing has to thread it through function signatures.
An evaluation run binds one run id to the same variable, so its log lines and its export name the
same run.

Every call to the generation provider — each generation attempt and each grounding check, successful
or not — leaves one `ProviderCallRecord` (`rag/telemetry.py`): step, attempt, model, requested
response format, service-observed elapsed time (transport retries and evaluation pacing included —
not provider latency), result (`parsed` / `unusable_reply` / `provider_error`), the failure's
category and detail, the provider's finish reason and token usage as reported (`null` when not
reported, never estimated), and the reply's size in characters and in non-whitespace characters. It
is built in `rag/service.py`, the only place that calls the provider; it is logged as one
`provider call` line, carried on the answer (and on a `GenerationUnavailableError`), and exported
per question and in aggregate by the end-to-end evaluation. It records and decides nothing: no
routing, retry or refusal reads it.

**Failure policy.** Every way a call to the generation provider can fail is one
`GenerationFailureCategory`, and `rag/failure_policy.py` holds one row per category: who detects it
(adapter, answer/verdict contract, or the service's provider boundary), whether the adapter retries
it beneath the port, whether the answer may be generated once more, and what the request ends as.
The service reads its regeneration decision from that table and nowhere else; the adapters' retry
behaviour is what the table describes, and fault-injection tests prove each row
(`tests/unit/test_failure_policy.py`). Every row ends as a technical error — `503`, fail closed, an
availability finding — because a provider failure says nothing about the knowledge base. That
includes a grounding check that gave no readable verdict; only a readable `not_supported` is a
refusal (`NOT_GROUNDED`, `200`). Precedence is fixed: no reply → the adapter's facts; a parseable
reply is never a failure; a rejected reply the provider cut off at the limit is `output_truncated`,
any other rejected reply `unparseable_output`. Anything an adapter raises outside the port's
contract is `unclassified` at the one awaited provider call, with its cause chained; a defect
anywhere else is not normalised. At most three provider calls per request (two generations, one
check), each with the adapter's bounded transport retries, all inside one request deadline
(`request_deadline_seconds`, provisional 60s) enforced with `asyncio.timeout`; cancellation is never
caught. Failures keep their root cause: step, category, rule or provider kind, status, attempts,
`Retry-After`, finish reason, token usage and the provider-call records.

**Privacy.** The access log records shape, not content: method, path, status, duration. Chat
messages, prompts, retrieved passages, query strings and request bodies are not logged. Answering a
question logs an outcome, counts, durations, the embedding space and the generation model — never
the question, the context, the prompt or the answer. Every `extra=` field logged anywhere in `src/`
was enumerated and reviewed in Phase 6; the complete set is counts, durations, identifiers,
statuses, model and version names — the provider-call record included, which has no field that
could hold text. The evaluation harness reads a dataset committed to the
repository and never a real user's question, which is why it needed no opt-in data flow.

**What leaves this process.** Two boundaries, both narrow and both worth stating plainly. The
embedding provider receives the query representation — the normalized question, and nothing else.
The generation provider receives the grounding instructions, the selected *public* passages under
their labels, and the question. Neither receives the corpus, internal documents, fingerprints,
source paths, chunk ids, vectors, settings or credentials belonging to anything else. The developer
CLI shows more than the API does, deliberately; it still never prints a credential or an
`Authorization` header, and none of its output is logged.

Ingestion is the exception that proves the rule: it prints document paths and field names, because
it is a developer tool operating on repository files rather than on user input. It still never
prints document *content* — `knowledge inspect` reports a character count and a hash, not the text.

Indexing logs counts, durations and the embedding space it worked in. It never logs vectors, API
keys, `Authorization` headers or raw provider response bodies — provider error messages are composed
locally from a status code and the operation, never echoed from upstream.

## 7. Portability

**FastAPI/ASGI is the application boundary.** Above it is infrastructure that can be swapped;
below it is portable Python that runs anywhere ASGI runs — Uvicorn locally, Uvicorn in Docker, any
ASGI host in production.

Cloudflare is an attractive *deployment* target (Workers, Vectorize, D1, R2, a generous free tier
and the portfolio already lives there), and parts of it are in use: Vectorize as the vector store,
Workers AI for generation, a Worker as the edge gateway. It is still not an architectural
dependency — the origin itself runs as a plain ASGI container on Cloud Run. The rules that keep it
that way:

* Provider-specific code lives in `infrastructure`, behind the ports — a `VectorizeVectorStore` is
  one adapter among possible others.
* No Cloudflare binding, SDK or runtime assumption may appear in `api`, `application`, `rag`,
  `domain`, `ports` or `core`.
* Deployment-shaped constraints (bundle size, cold starts, request limits) were evaluated when
  deployment was actually on the table, not designed around speculatively.
* Docker remains a supported target permanently, which is the practical test that the core has not
  quietly acquired a platform dependency.

What is provisioned: a Vectorize index, Workers AI, the Worker gateway, and the Cloud Run service
that runs the container. No D1 database and no R2 bucket — neither is used by this service. See
[ADR 0003](adr/0003-fastapi-portable-runtime.md), whose Context records what was true when the
decision was made.

## 8. What exists today

| Component | State |
| --- | --- |
| FastAPI app, ASGI, Uvicorn, Docker | implemented |
| `GET /health` | implemented |
| `POST /api/v1/chat` | implemented — grounded answer with citations |
| Error envelope + handlers (404/405/422/500/503) | implemented |
| Request id + structured logging | implemented |
| Typed configuration, CORS | implemented |
| Domain models, provider ports | implemented and in use |
| Knowledge document standard, schema version 1 | specified and enforced |
| Knowledge ingestion: discovery → frontmatter → validation → normalization → provenance → conflicts | implemented |
| Structure-aware chunking: heading hierarchy, block packing, overlap, chunk identity | implemented |
| Versioned embedding representation + embedding-space identity + embedding fingerprint | implemented |
| Embedding providers: deterministic (local, offline) and Mistral | implemented |
| Vector stores: in-memory reference and Cloudflare Vectorize | implemented |
| Incremental index planning and convergent synchronization | implemented |
| Query boundary, versioned query representation, query embedding | implemented |
| Public-only retrieval, similarity threshold, deterministic ordering | implemented |
| Chunk resolution from the corpus snapshot | implemented |
| Bounded, labelled, deterministic context building | implemented |
| Grounded prompt with separated roles + structured answer contract | implemented |
| Generation providers: deterministic stub (local, offline), Mistral and Cloudflare Workers AI | implemented |
| Backend-owned citation validation | implemented |
| `portfolio-rag knowledge validate` / `inspect` / `chunks` / `embedding` / `index` | implemented |
| `portfolio-rag query retrieve` / `answer` | implemented |
| Evaluation harness, dataset, metrics, `portfolio-rag eval run` | implemented |
| Adversarial and failure-matrix test suites | implemented |
| Rate limiting / abuse protection | deliberately at the deployment edge ([ADR 0008](adr/0008-abuse-boundary-at-the-edge.md)) |
| Conversation state, persistence, agents, tool calling | deliberately not planned |
| Production deployment: Docker container on Google Cloud Run, behind the Cloudflare Worker edge gateway | running at v1.0.0 |

## Related documents

* [ROADMAP.md](ROADMAP.md) — phases, deliverables, definitions of done
* [adr/0001-modular-monolith.md](adr/0001-modular-monolith.md)
* [adr/0002-provider-agnostic-core.md](adr/0002-provider-agnostic-core.md)
* [adr/0003-fastapi-portable-runtime.md](adr/0003-fastapi-portable-runtime.md)
* [adr/0004-versioned-markdown-knowledge-format.md](adr/0004-versioned-markdown-knowledge-format.md)
* [adr/0005-deterministic-structure-aware-chunking.md](adr/0005-deterministic-structure-aware-chunking.md)
* [adr/0006-versioned-embeddings-and-incremental-indexing.md](adr/0006-versioned-embeddings-and-incremental-indexing.md)
* [adr/0007-grounded-retrieval-and-backend-owned-citations.md](adr/0007-grounded-retrieval-and-backend-owned-citations.md)
* [adr/0008-abuse-boundary-at-the-edge.md](adr/0008-abuse-boundary-at-the-edge.md)
* [SECURITY.md](SECURITY.md) — what is enforced structurally, and what remains a risk
* [DEPLOYMENT.md](DEPLOYMENT.md) — the current deployment, and the procedure that reproduces it
* [../evaluation/README.md](../evaluation/README.md) — how retrieval and grounding were measured
* [../knowledge/README.md](../knowledge/README.md) — knowledge document standard
* [../AGENTS.md](../AGENTS.md) — engineering rules for humans and agents
