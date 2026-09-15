# ADR 0002 — External AI and storage systems are reached through ports

* **Status:** accepted
* **Date:** 2026-08-07
* **Phase:** 1

## Context

A RAG system depends on an embedding model, a vector store and a language model. All three are
volatile in a way normal dependencies are not: pricing changes, free tiers appear and disappear,
models are deprecated, and quality differences only become visible once real queries are running.

The intended first provider is Mistral AI, on its free tier, chosen for cost rather than after
evaluation. Cloudflare Vectorize is the likely first vector store for the same reason. Both choices
should be reversible — and "reversible" has to be designed in before the first adapter exists, not
after five modules have imported an SDK.

Rejected alternative: a RAG framework (LangChain, LlamaIndex, Haystack). Those provide provider
abstraction, but at the cost of owning the retrieval mechanics — chunking, ranking, context
assembly, grounding — which is exactly where answer quality is decided and exactly what this
project wants to understand and control. The abstraction is wanted; the framework is not.

## Decision

Define the boundary as small `typing.Protocol` contracts in `portfolio_rag.ports`:

* `LLMProvider` — structured generation request → structured generation response
* `EmbeddingProvider` — text → vectors, plus the model id and dimensionality needed to detect a
  mismatch between index and query
* `VectorStore` — `upsert` / `search` / `delete` over knowledge vectors

Rules:

1. Application, RAG, domain and API code depends on ports, never on an SDK.
2. Adapters live in `portfolio_rag.infrastructure` and are wired in at the composition root.
3. Ports exchange domain types, so no provider's data model leaks into the core.
4. A port is added when it has a caller. `ConversationStore` and `DocumentStore` are named in the
   architecture but deliberately not defined yet.
5. Ports stay narrow. Features nobody consumes (streaming, tool calling, hybrid search) are added
   when a phase needs them — a wide port makes every adapter expensive.

Structural typing means an adapter subclasses nothing and registers nowhere: matching the signature
is the whole contract, and a test fake is a plain class.

## Consequences

**Good**

* Swapping a provider is writing one adapter and changing one wiring line.
* The core is testable without network access, credentials or cost — which is what makes evaluation
  affordable.
* No provider SDK is a build dependency of the core, so the dependency list stays small.
* Comparing two providers on the same corpus becomes a normal experiment rather than a refactor.

**Bad / accepted**

* A thin translation layer per adapter, and provider-specific capabilities have to be either
  normalized into the port or deliberately left out.
* The ports are designed before any real adapter exists, so the first adapter will probably reveal a
  wrong assumption. Changing a port at that point is the intended outcome, not a failure — it is
  cheap while the ports have one caller each.

**Revisit when** a provider capability turns out to be decisive for answer quality and cannot be
expressed through a port. The response is to widen the port deliberately, not to bypass it.
