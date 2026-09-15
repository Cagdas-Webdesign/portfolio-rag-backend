# ADR 0003 — FastAPI/ASGI is the portable core; Cloudflare is a deployment target

* **Status:** accepted
* **Date:** 2026-08-07
* **Phase:** 1

## Context

The portfolio front end already runs on Cloudflare and talks to a Cloudflare Worker, and
Cloudflare's free tier covers everything this project needs: Workers for compute, Vectorize for
vectors, D1 for relational data, R2 for objects. Deploying there later is the obvious path.

The trap is equally obvious: build against Workers bindings and the "application" becomes a
Cloudflare artifact. It could then only be tested where those bindings exist, and moving it — or
running it locally — would mean rewriting it.

There is a second reason to keep the boundary strict: the deployment decision is not due yet.
Nothing is deployed yet, and by the time something is, the Python-on-Workers story, the bundle-size
constraints and the free-tier terms may all look different.

## Decision

The ASGI application (`portfolio_rag.main:app`) is the boundary of the portable system.

* Everything below it is plain Python: no platform bindings, no runtime-specific globals, no
  assumptions about the host beyond "async Python".
* Everything above it — Uvicorn, Docker, a future Worker or serverless wrapper — is infrastructure
  and replaceable.
* Cloudflare services are reached through the ports of [ADR 0002](0002-provider-agnostic-core.md).
  `VectorizeVectorStore` will be an adapter, not a foundation.
* Docker stays a supported target permanently. It is the practical test that no platform dependency
  has crept in: if the image runs and `/health` answers, the core is still portable.
* Deployment integration is treated as an adapter problem when it arrives, not as an architecture
  problem now.

Phase 1 provisions nothing: no Vectorize index, no D1 database, no R2 bucket, no Worker, no
deployment, no account resources of any kind.

## Consequences

**Good**

* The application runs identically on a laptop, in CI and in a container — so tests exercise the
  real thing.
* Cloudflare remains a choice rather than a commitment; so does moving away from it.
* Local development needs no cloud account, no credentials and no network, which keeps the cost at
  €0 and the feedback loop fast.

**Bad / accepted**

* Platform-native conveniences (direct bindings, edge KV access, platform-integrated auth) are not
  used directly; they arrive through an adapter, which costs a small amount of indirection.
* If the eventual target turns out to be Python Workers, an adapter layer will be needed to bridge
  ASGI to the Worker request model, and constraints such as bundle size will have to be dealt with
  then. That work is bounded and sits above the boundary.

**Revisit when** deployment is actually being implemented (Phase 6) and the target's real
constraints are known. The decision to revisit is about *how to host the ASGI app*, not about
whether the core may depend on a platform — that answer stays no.
