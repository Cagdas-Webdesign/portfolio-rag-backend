# ADR 0007 — Grounded retrieval and backend-owned citations

* **Status:** accepted
* **Date:** 2026-08-11
* **Phase:** 5 — Retrieval & Grounded RAG
* **Supersedes:** nothing. Builds on [0002](0002-provider-agnostic-core.md) (provider-agnostic core)
  and [0006](0006-versioned-embeddings-and-incremental-indexing.md) (embedding spaces, fingerprints).

## Context

Phases 1–4 built the corpus side: documents become validated, chunked, versioned, embedded records
in a vector index. Nothing consumed any of it. A question could not be turned into a search, no
model was called, and `POST /api/v1/chat` returned `501`.

Phase 5 is the query side, and it is where this project stops being infrastructure and starts being
a product with an opinion. The decisions below are the ones that would be expensive to reverse:
they shape what an answer is allowed to be, what a citation means, and what a user is told when the
knowledge base does not know.

The constraints they were made under: €0, no credentials in CI, provider-swappable, and honest
about its own limits — this is a public-facing assistant answering questions about a real person,
where a confident invention is worse than a refusal.

## Decision

### 1. Vector retrieval only, in v1

One search: embed the question with the corpus's own embedding provider, ask the index for the
`top_k` nearest public records, drop what scores below a threshold. No re-ranking, no lexical or
hybrid search, no query rewriting, no expansion, no HyDE, no multi-query.

Every one of those is a real technique that plausibly helps. None of them has been measured against
this corpus, because nothing has: the evaluation set arrives in Phase 6. Adding a re-ranker now
would mean carrying a second model, a second failure mode and a second cost centre to fix a problem
nobody has observed. **Measured need over feature accumulation.**

### 2. Public visibility is structural, not a filter

`PublicRetrievalService` puts `visibility=public` into the query it sends to the store. There is no
parameter for it, no policy field, no keyword argument, no override.

The alternative — retrieve everything, filter afterwards — is one forgotten line away from a leak,
and the leak is silent and public. Here, a caller cannot ask for internal knowledge, correctly or
incorrectly, because the request cannot be expressed. Two further checks belong to the same idea:
resolved chunks are re-checked against the corpus (which is newer than the index), and the query's
embedding space is compared with the index's by identity rather than dimensionality.

**Public by construction over filter-after-retrieval.**

### 3. A similarity threshold, and it is a similarity

A match below `min_similarity` is not evidence. Without a floor, a question the corpus knows nothing
about still returns five passages — the five least irrelevant ones — and a model handed irrelevant
passages will find something to say.

The number is a cosine similarity, and it is called that everywhere: in the policy, in the CLI, in
this document. It is not a confidence, not a probability, and it is never rendered as a percentage.
A "92% confident" label on a cosine score is a number that means nothing pretending to be evidence.

### 4. Insufficient retrieval short-circuits before the provider

Nothing above the threshold means the answer is a fixed, honest statement that the knowledge base
does not cover the question, with no citations and HTTP `200`.

No provider call: it costs nothing, takes no time, and — the real reason — a model given no evidence
and asked a question is being invited to invent one. The short circuit removes the invitation.

`200` rather than an error code: "I don't know" is a correct answer to a question, not a server
fault. Errors are reserved for things a client can act on.

### 5. Bounded, labelled, deterministic context

Retrieved passages are rendered into one context string: source label, document title, heading path,
content. Deduplicated by exact identity only — Phase 3's controlled overlap must survive — and
bounded by a token budget that accounts for instructions, the question and an output reserve, not
just the passages.

A passage either fits whole or is left out and counted. Nothing is truncated: half a paragraph of
evidence reads exactly like whole evidence to a model, and to a reader of the answer.

The context representation is separate from `chunk.content`, exactly as the embedding representation
is ([ADR 0006](0006-versioned-embeddings-and-incremental-indexing.md)). It is versioned
(`grounded-context-v1`) and pinned by golden tests.

### 6. Generation is grounded-only, with roles separated structurally

Instructions are a system message. Retrieved knowledge and the question are a user message, in
labelled sections, with the model told that knowledge is data. No document content can reach the
instruction role, because it is never in that message.

This is the architectural half of prompt-injection defence, not the whole of one — a determined
instruction inside a passage can still influence a model, and adversarial hardening is Phase 6.
What is settled here is that the boundary exists, is visible in the code, and is asserted by a test.

The prompt is versioned (`grounded-answer-v3`) so an evaluation run can name what it measured.

### 7. The model selects sources; the backend owns them

The model is asked for structured output — `{"answer": ..., "sources": ["S1"]}` — and asking for
JSON is the difference between a field lookup and regular expressions over prose. Labels are minted
by the context builder, are local to one answer, and carry no external meaning: a model that can
only say `S2` cannot invent a source.

Every claimed label is checked against the context this request actually built. Unknown labels are
dropped and logged; duplicates collapse; passages from one document and section become one citation.
Every published citation is built from a `RetrievedChunk` — a document title, path or URL that a
model produced is text, not provenance.

**Backend-owned citations over model-invented sources.**

### 8. An answer with no verifiable source is not published

If a model produces a substantive answer but cites nothing the backend can verify, the answer is
replaced by the same honest statement as case 4, with no citations, and the event is logged.

A grounded answer without grounding is not a successful grounded answer. Publishing it would mean
the citation mechanism is decorative — present when convenient, absent when it mattered.

### 9. No fallback to the model's own knowledge

When the corpus is silent, the system says so. It does not answer from training data.

The authority in this system is the retrieved corpus. An assistant that quietly switches to general
knowledge is a different product, and an unfalsifiable one: a reader cannot tell which sentences
came from the documents and which did not.

### 10. Grounding is a process, and is described as one

Every answer is generated from retrieved public passages and carries only verified sources. That
narrows the room a model has to invent and makes what it says checkable against files in git.

It is not a correctness guarantee. The documentation says "grounded", never "hallucination-free",
and no number in this system is presented as a confidence.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| **Answer directly from the model, no retrieval** | Fast, cheap, and unable to say anything specific about a real person. Also unfalsifiable: nothing to check an answer against. |
| **Retrieve, but let the model use world knowledge to fill gaps** | The most tempting option, and the one that quietly ruins the product: an answer becomes a mixture of sourced and unsourced claims with no way to tell them apart. |
| **Filter `internal` documents after retrieval** | One forgotten call away from a public leak, and the failure is silent. The structural version costs nothing extra. |
| **Free-form citations: let the model name documents and URLs** | Simple to prompt for, impossible to trust. A model naming a plausible file produces a citation that looks exactly like a real one. |
| **Answer whatever retrieval returns, however weak** | Guarantees an answer to every question, and guarantees an invented one to every unanswerable question. |
| **Call the provider even when retrieval found nothing** | A cost and a latency spent on an invitation to hallucinate. |
| **Truncate a passage that does not fit the budget** | Silent truncation is undetectable downstream: a half-sentence of evidence reads like whole evidence. Skipping and counting is honest. |
| **Re-ranking or hybrid search now** | A fix for an unmeasured problem, carrying a second model and a second failure mode. Phase 6 evaluation decides whether the problem exists. |
| **A RAG framework** | Chunking, retrieval, context and grounding are where answer quality is decided. Delegating them would delegate the interesting part and keep the maintenance. |
| **Store chunk text in vector metadata** | Would make resolution trivial and put a second copy of the corpus in a place that can drift from `knowledge/`, plus press against provider metadata limits. A resolver against the source of truth has neither problem. |
| **An exact tokenizer for the context budget** | A real dependency, pinned per model, wrong when the model changes, feeding a budget that is itself a conservative guess. A documented, deliberately pessimistic estimate is honest and costs nothing. |

## Consequences

**Good.**

* A public request cannot retrieve internal knowledge, and that is a property of the code rather
  than of reviewer attention.
* Every citation traces to a chunk id, a document fingerprint and a file in git.
* An unanswerable question produces a refusal instead of an invention, without spending a request.
* The whole pipeline — corpus to answer — runs offline, in CI, at no cost.
* Mistral can be swapped for another provider without touching retrieval, context or citations.

**Costs, accepted.**

* Retrieval quality is unmeasured. The threshold and `top_k` are starting values, and this document
  says so rather than implying otherwise.
* The token budget is an estimate, deliberately pessimistic, and leaves some context unused.
* An answer the corpus supports but the model cites badly is refused rather than published. Failing
  closed is the right default; Phase 6 has the data to say how often it happens.
* No conversation state: a follow-up question is answered on its own. That is a product decision,
  recorded in the roadmap. *Later amended:* a client may send the last few turns with a question
  (`rag/conversation.py`). They reach only the generation prompt, as a section named as not being a
  source; retrieval, the grounding check and citation validation are unchanged, and nothing is stored.
* The offline development stack answers with a stub that generates no language. It is unmistakably
  labelled, and the composition root refuses to build it in production.
