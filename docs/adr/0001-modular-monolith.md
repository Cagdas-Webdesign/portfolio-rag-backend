# ADR 0001 — Modular monolith instead of microservices

* **Status:** accepted
* **Date:** 2026-08-07
* **Phase:** 1

## Context

The system will eventually contain an HTTP API, an ingestion pipeline, a retrieval engine, LLM
integration and evaluation tooling. Those are separable concerns, and splitting them into services
is a common instinct.

The situation they have to fit into: one developer, one client application, no traffic, and a hard
budget of €0. Ingestion is offline and infrequent; the request path is a single question-answering
flow. There is no team boundary and no scaling pressure to encode in a deployment topology.

## Decision

Build one deployable application with enforced internal module boundaries: `api`, `application`,
`rag`, `domain`, `ports`, `infrastructure`, `ingestion`, `core`. Dependencies point inwards, and the
domain and the ports import nothing outward.

Modules are separated by dependency rules, not by network calls.

## Consequences

**Good**

* One process to run, test, debug and deploy. A single `uv run pytest` covers the whole system, and
  a stack trace crosses the whole request path.
* No distributed-systems tax: no inter-service contracts to version, no partial failures, no
  tracing infrastructure needed to answer "what happened to this request?".
* Free to run. Multiple services would mean multiple deployment units on a €0 budget.
* Boundaries are still real. The dependency rules are the part that makes later extraction possible;
  the deployment topology is not.

**Bad / accepted**

* Everything scales together. Irrelevant at current load, and ingestion — the one genuinely
  different workload — runs offline anyway.
* Boundaries are conventions rather than network walls, so they can be violated by accident. This is
  mitigated by the layer table in `ARCHITECTURE.md`, by review (`.claude/skills/architecture-review`)
  and by keeping `domain`/`ports` free of framework imports, which makes a violation obvious.

**Revisit when** a component develops genuinely different scaling or availability needs (a heavy
ingestion workload, a long-running evaluation harness), or when more than one team owns parts of the
system. Extraction then means moving a package and putting a transport in front of an existing port
— which is why the ports exist.
