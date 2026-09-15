# ADR 0006 — Versioned embedding representation and incremental vector indexing

* **Status:** accepted
* **Date:** 2026-08-09
* **Phase:** 4

One ADR, not two. "What text becomes a vector" and "when does that vector have
to be made again" are the same question asked twice: the answer to the first
*is* the mechanism for the second.

## Context

Phase 3 ended with chunks. Turning them into a searchable index raises three
decisions that are expensive to reverse once vectors exist.

**What gets embedded.** A chunk from the middle of a document is often
unintelligible alone — "Ich verwende es für die Anbindung externer Systeme" is
a good answer to a question nobody can tell it is answering. But padding the
text with metadata pollutes the vector and hands an external provider
information it has no reason to see.

**What makes two vectors comparable.** Not dimensionality: two models that both
emit 1024 numbers produce vectors that mean nothing to each other. An index
that quietly mixes them still returns results, still looks healthy, and is
wrong in a way no error surfaces.

**What has to be recomputed, and when.** Embedding is the only metered
operation in this pipeline. Re-embedding an unchanged corpus is money and time
spent to produce numbers that already exist — and with a €0 budget, "just
re-embed everything" is not a shrug, it is the difference between a workflow
that runs and one that does not.

## Decision

**A versioned embedding representation.** `chunk.content` stays source text.
A separate, explicit rule composes what is embedded: **document title, heading
path, chunk content**, blank-line separated, identified as
`embedding-text-v1`. Metadata — visibility, trust level, source paths,
fingerprints, ids, licences, dates, topics, technologies — is deliberately not
embedded. It is structured data that retrieval filters on.

**Embedding space as an identity.** `EmbeddingSpec` is provider + model +
dimensions + representation version, and compatibility is equality over all
four. A store refuses a record from a different space; the indexing service
refuses to extend an index that already holds one.

**A third fingerprint level.** The system now has three, each answering one
question:

| Fingerprint | Question | Covers |
| --- | --- | --- |
| document | has this source document changed? | metadata + body |
| chunk | has this structural retrieval unit changed? | strategy, policy, document id, heading path, content |
| embedding | does this unit need a new vector *in this space*? | representation text + embedding space |

They are deliberately not nested. A typo fixed in section one changes the
document fingerprint and section one's chunk fingerprint — and leaves section
three's embedding fingerprint alone, so section three is not re-embedded.

**Incremental, convergent synchronization.** Each run compares desired state
against actual state and produces a plan: create, re-embed, metadata-only
update, unchanged, delete. A second identical run is a true no-op — zero
provider calls, zero writes. The index itself is the reuse mechanism; there is
no separate embedding cache to disagree with it.

**Metadata-only updates cost nothing.** Flipping `visibility` changes what
retrieval may do with a chunk but not what the chunk says, so the stored vector
is fetched and re-upserted with corrected metadata. No provider call.

**Upserts before deletes.** Two remote systems are not a transaction. Writing
first means a mid-run failure leaves an index that is stale, not incomplete —
and the next run converges.

## Alternatives considered

**Embedding the chunk content alone.** Simplest, and loses the context that
makes a mid-document chunk interpretable. Rejected: the heading path is already
structured, already free, and already the thing a human would read first.

**Embedding all metadata.** Tempting because more signal sounds better.
Rejected on two grounds: an embedding of prose diluted with enum values and
identifiers is worse at matching prose, and it would send internal
classifications to a third party for no retrieval benefit. If evaluation later
shows topics help, that is version 2 — with evidence.

**Always re-embedding the whole corpus.** Honest and trivially correct.
Rejected: it makes every trivial edit cost a full pass over a metered API, and
it removes the incentive to keep the fingerprints meaningful.

**Provider-specific indexing logic.** Writing directly against Vectorize would
be less code today. Rejected: it would put a vendor in the application layer,
which is exactly what [ADR 0002](0002-provider-agnostic-core.md) exists to
prevent, and it would make the pipeline untestable without an account.

**A Cloudflare-first core.** Same objection, and worse: the corpus side would
inherit a platform's limitations as if they were design. Instead the port is
neutral, the in-memory store is the reference implementation, and Vectorize is
one adapter whose real constraint — no enumeration — is modelled as a
capability rather than assumed away.

## Consequences

**Good**

* The whole pipeline runs offline, free and deterministically: a local provider
  derived from SHA-256 and an in-memory store, both real implementations.
* Editing one paragraph costs one embedding. Flipping a metadata flag costs
  none.
* Incompatible vector spaces cannot be mixed by accident.
* What is sent to an external provider is one function, easy to read and to
  audit.
* Swapping either the provider or the store is a settings change.

**Bad / accepted**

* A character-based representation is not tuned to any model's tokenizer; if a
  chunk turns out to exceed a model's input window, that surfaces at Phase 4
  boundaries rather than being prevented by construction.
* Changing the representation version invalidates every vector. That is
  correct, and it makes the version a decision rather than a refactor.
* Vectorize cannot enumerate its contents, so stale-record detection is
  unavailable against it. The CLI says so; the alternative was pretending.
* Convergence, not atomicity: a failed run can leave an index partially
  updated. It is never reported as success, and the next run finishes the job.

**Revisit when** evaluation exists (Phase 6) and can compare representation
variants on real questions, or when a chosen model's token window turns out to
constrain chunk size.
