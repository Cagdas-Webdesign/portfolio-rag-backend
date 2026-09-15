# Edge gateway

A single Cloudflare Worker in front of the RAG origin. It exists because every
request that reaches the origin costs an embedding call, a Vectorize query and
a generation call, all on free-tier accounts — so the cheapest place to refuse
a request is before any of that.

```
browser → Turnstile token → Worker → size → challenge → rate limit → origin
```

## What it does, in order

| Step | Refusal |
| --- | --- |
| Path is `/api/v1/chat` | `404` |
| Method is `POST` (or `OPTIONS` preflight) | `405` |
| Body ≤ 16 KiB | `413` |
| `X-Turnstile-Token` present | `401` |
| Turnstile verifies server-side | `403` |
| Burst limit: 3 / 10s per IP+route | `429` + `Retry-After: 10` |
| Sustained limit: 10 / 60s per IP+route | `429` + `Retry-After: 60` |
| → forward, setting `X-Edge-Auth` | |

`fetch` to the origin is reached from one place in `src/worker.js`, after every
check above. A request that fails any of them never crosses the network.

## What the frontend has to send

The portfolio frontend is a separate repository and is **not** changed by this.
The frontend must:

1. Render the Turnstile widget with the **site key** (public, safe to ship).
2. On submit, take the token the widget produces.
3. `POST` to the Worker's `/api/v1/chat` with:
   - `Content-Type: application/json`
   - `X-Turnstile-Token: <token>`
   - body `{"message": "...", "conversation_id": null}`
4. Request a fresh token per submission — Turnstile tokens are single-use and
   short-lived. A reused token verifies as a failure and comes back `403`.
5. Handle `429` by honouring `Retry-After` rather than retrying immediately.

The frontend never sees or sends `X-Edge-Auth`. That header is set by the
Worker and would be stripped if a client supplied one.

## Local checks

```bash
node --test edge/test
```

No network, no account, no secrets: the tests stub `fetch` and the rate-limit
bindings.

## Deploying

See [../docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) — the manual steps, the
secrets to set, and the Cloud Run settings that go with them.
