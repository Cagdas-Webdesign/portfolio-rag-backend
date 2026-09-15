# ADR 0008 — The public abuse boundary lives at the deployment edge

* **Status:** accepted
* **Date:** 2026-08-11
* **Phase:** 6 — Final Engineering & Production Readiness
* **Relates to:** [0003](0003-fastapi-portable-runtime.md) (the runtime is portable, the deployment
  target is not chosen), [0007](0007-grounded-retrieval-and-backend-owned-citations.md) (what one
  request is allowed to cost)

## Context

`POST /api/v1/chat` is meant to be called by a public web page. There is no login, no account and
no session, and none is planned — the corpus is public knowledge, so there is nothing to
authenticate anybody *for*.

What there is, is a bill. Every request that clears the similarity threshold costs one embedding
call and one generation call against a free-tier account. The €0 constraint is a hard requirement of
this project, which makes request *rate* the one genuinely unbounded quantity in the system.

So Phase 6 had to answer: does this application limit its own traffic, or does the deployment?

## Decision

**The application does not rate limit. The deployment edge does.** The application's job is to make
each individual request cheap and bounded; bounding how many arrive is the responsibility of
whatever terminates TLS in front of it, and that is documented as a deployment requirement rather
than left implied.

### Why not in-process

An in-process limiter has to identify a client, and neither available way of doing that survives
contact with a real deployment:

* **Keyed on the socket peer.** Correct only when the container is exposed directly to the
  internet. Behind any reverse proxy, CDN or PaaS router — which is every realistic target for this
  service — every visitor arrives from the same handful of addresses and shares one bucket. The
  first busy minute rate-limits the entire site.
* **Keyed on `X-Forwarded-For`.** Correct only when a trusted proxy sets it and the application
  knows how many hops to skip. Exposed directly, the header is attacker-controlled, so the limiter
  can be bypassed with one line of curl — and worse, it can be *aimed*: a forged header lets an
  attacker exhaust somebody else's bucket.

Choosing between those at build time is impossible, because the deployment target is deliberately
undecided ([ADR 0003](0003-fastapi-portable-runtime.md)). Shipping either would mean shipping a
mechanism that is wrong in a configuration nobody has ruled out, while *looking* like protection —
and a security control that is believed and does not work is worse than a documented gap.

### What the application does bound

Per-request cost is capped, and this is where the application's responsibility genuinely lies:

| Bound | Where | Effect |
| --- | --- | --- |
| 16 KiB request body | `api/middleware/gate.py` | refused before the body is parsed |
| 2000 characters per message | the HTTP schema — stricter than the pipeline's own bound | one embedding input, bounded |
| 4000 characters per query | `rag/query.py` — also the CLI and the evaluation harness | the pipeline's own ceiling |
| Requests that skipped the gateway | `api/middleware/gate.py` | `403` before any provider call |
| Invisible-only messages rejected | `rag/query.py` | no provider call for empty input |
| `top_k` ≤ 50, default 5 | `rag/policy.py` | bounded retrieval work |
| Context budget, default 6000 tokens | `rag/policy.py` | bounded prompt size |
| `max_output_tokens`, default 800 | the same reserve, so they cannot disagree | bounded generation |
| Weak retrieval short-circuits | `rag/service.py` | **zero** generation calls when nothing was found |
| Provider timeouts, default 30s | `core/config.py` | bounded time per request |
| ≤ 3 attempts, no retry on 401/403 | both Mistral adapters | bounded amplification |

Measured on the evaluation set: an answered question costs exactly one embedding call and one
generation call; an unanswerable one costs one embedding call and nothing else.

### What the deployment must provide

Documented in [DEPLOYMENT.md](../DEPLOYMENT.md) as a required step, not a suggestion: a rate limit
on `POST /api/v1/chat`, keyed on client IP, configured at the edge — Cloudflare, the PaaS router, or
nginx. The edge is the only place that knows the real client address, and every plausible target
offers this without extra cost.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| **In-process limiter on the socket peer** | Silently wrong behind a proxy: one shared bucket for every visitor. |
| **In-process limiter on `X-Forwarded-For`** | Spoofable when exposed directly, and forgeable to exhaust another client's quota. |
| **A limiter with a `trusted_proxies` setting** | Correct, and it moves the decision to a configuration value that is wrong by default and that nobody revisits. The edge already knows the answer. |
| **Redis-backed distributed limiting** | Infrastructure with no other user, for a single-container deployment. Ruled out by AGENTS §3. |
| **A global in-flight cap instead** | Bounds concurrency, not spend: a patient serial script drains the same budget more slowly. It would look like a limit while not being one. |
| **API keys for the chat endpoint** | An auth system for a public page, and the key would ship in the front-end bundle where anybody can read it. |

## Consequences

**Good.** No mechanism that is wrong in a plausible deployment. Per-request cost is bounded and
measured. The requirement is written down where somebody deploying will read it.

**Costs, accepted.** A deployment that skips the edge configuration has no rate limit at all — the
gap is real, and naming it is the point of this ADR. The risk is a drained free-tier quota and an
assistant that answers `503` until it refills, not a data exposure: retrieval stays public-only and
citations stay backend-owned however much traffic arrives.

**Revisit when** the deployment target is fixed *and* it turns out not to offer edge rate limiting.
Then an in-process limiter can be written against a known proxy configuration, which is the only
condition under which it can be written correctly.
