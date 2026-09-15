/**
 * The public gateway in front of the RAG backend.
 *
 * Everything expensive lives behind this Worker: an embedding call, a
 * Vectorize query and a generation call, all on free-tier accounts. So the
 * job here is to answer "should this request cost anything?" as cheaply as
 * possible, and to forward only what survives.
 *
 *   method → size → Turnstile → burst limit → sustained limit → origin
 *
 * The order is deliberate. Each check is cheaper than the one after it, and
 * the two that talk to the network — Turnstile and the origin — are last.
 * A request that fails any of them is answered here; `fetch` to Cloud Run is
 * reached from exactly one place in this file, after every check has passed.
 *
 * **The origin secret is set here and only here.** A header a client supplied
 * is dropped before the request is rebuilt, so a browser cannot present one
 * and skip this file by calling Cloud Run directly.
 *
 * No framework, no router, no dependency. One route, one method, six checks.
 */

/** The only path this gateway forwards. Anything else is a 404 here. */
const CHAT_PATH = "/api/v1/chat";

/** Matches the origin's own body guard, so a request refused there is
 *  refused here first and never crosses the network. */
const MAX_BODY_BYTES = 16 * 1024;

/** Header the origin checks. Named the same on both sides on purpose. */
const EDGE_AUTH_HEADER = "X-Edge-Auth";

/** Header the browser sends its Turnstile token in. */
const TURNSTILE_HEADER = "X-Turnstile-Token";

const TURNSTILE_VERIFY_URL =
  "https://challenges.cloudflare.com/turnstile/v0/siteverify";

/** Advertised on a 429. A number a client can act on beats a bare status. */
const BURST_WINDOW_SECONDS = 10;
const SUSTAINED_WINDOW_SECONDS = 60;

export default {
  /**
   * @param {Request} request
   * @param {Env} env
   * @returns {Promise<Response>}
   */
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname !== CHAT_PATH) {
      return refuse(404, "not_found", "The requested resource does not exist.");
    }

    if (request.method === "OPTIONS") {
      return preflight(request, env);
    }

    if (request.method !== "POST") {
      return refuse(
        405,
        "method_not_allowed",
        "This method is not allowed for the requested resource.",
        { Allow: "POST, OPTIONS" },
      );
    }

    // Read the body once, bounded. Everything below needs it, and reading it
    // here is what makes "too large" a decision rather than a surprise.
    const body = await readBounded(request, MAX_BODY_BYTES);
    if (body === null) {
      return refuse(413, "payload_too_large", "The request body is too large.");
    }

    const token = request.headers.get(TURNSTILE_HEADER);
    if (!token) {
      return refuse(
        401,
        "challenge_required",
        "A completed challenge is required for this endpoint.",
      );
    }

    const verified = await verifyTurnstile(env, token, clientIp(request));
    if (!verified) {
      return refuse(
        403,
        "challenge_failed",
        "The challenge could not be verified.",
      );
    }

    const limited = await checkRateLimits(env, request, url);
    if (limited !== null) {
      return refuse(
        429,
        "rate_limited",
        "Too many requests. Please wait a moment and try again.",
        { "Retry-After": String(limited) },
      );
    }

    return forward(request, env, body);
  },
};

/**
 * Read at most `limit` bytes. Returns null when the body is larger, without
 * buffering the rest of it.
 *
 * `Content-Length` is checked first because it lets an oversized request be
 * refused before a byte is read — but it is a claim, and a chunked request
 * carries none, so the stream is counted too.
 *
 * @param {Request} request
 * @param {number} limit
 * @returns {Promise<Uint8Array | null>}
 */
async function readBounded(request, limit) {
  const declared = request.headers.get("content-length");
  if (declared !== null) {
    const length = Number(declared);
    if (Number.isFinite(length) && length > limit) return null;
  }

  if (request.body === null) return new Uint8Array(0);

  const reader = request.body.getReader();
  const chunks = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > limit) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }

  const body = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    body.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return body;
}

/**
 * Verify a Turnstile token with Cloudflare, server-side.
 *
 * The secret never leaves the Worker and is never logged. A verification that
 * errors is treated as a failure: a challenge that cannot be checked has not
 * been passed, and failing open here would make the whole gateway optional.
 *
 * @param {Env} env
 * @param {string} token
 * @param {string | null} ip
 * @returns {Promise<boolean>}
 */
async function verifyTurnstile(env, token, ip) {
  const form = new FormData();
  form.append("secret", env.TURNSTILE_SECRET_KEY);
  form.append("response", token);
  if (ip) form.append("remoteip", ip);

  try {
    const response = await fetch(TURNSTILE_VERIFY_URL, {
      method: "POST",
      body: form,
    });
    if (!response.ok) return false;
    const outcome = await response.json();
    return outcome?.success === true;
  } catch {
    // Never the exception text: it can carry the request, and the request
    // carries the secret.
    console.warn("turnstile verification failed to complete");
    return false;
  }
}

/**
 * Apply both limiters. Returns the Retry-After to advertise, or null when the
 * request is within budget.
 *
 * Two bindings because they answer different questions: a burst limiter stops
 * a script hammering the endpoint, a sustained one stops a slow drip that
 * would still empty the account over an hour. Keyed on
 * `CF-Connecting-IP` plus the route — Cloudflare sets that header itself and a
 * client cannot forge it. It is a blunt instrument: everyone behind one NAT
 * shares a bucket, and a determined attacker has more than one address. It is
 * abuse control, not accounting, and it is not the only thing standing in
 * front of the origin.
 *
 * @param {Env} env
 * @param {Request} request
 * @param {URL} url
 * @returns {Promise<number | null>}
 */
async function checkRateLimits(env, request, url) {
  const key = `${clientIp(request) ?? "unknown"}:${url.pathname}`;

  const burst = await env.CHAT_BURST_LIMITER.limit({ key });
  if (!burst.success) return BURST_WINDOW_SECONDS;

  const sustained = await env.CHAT_SUSTAINED_LIMITER.limit({ key });
  if (!sustained.success) return SUSTAINED_WINDOW_SECONDS;

  return null;
}

/**
 * Rebuild the request for the origin and send it.
 *
 * The header set is built from scratch rather than copied: a client-supplied
 * `X-Edge-Auth` must not survive, and neither must anything else this gateway
 * has not decided to forward.
 *
 * @param {Request} request
 * @param {Env} env
 * @param {Uint8Array} body
 * @returns {Promise<Response>}
 */
async function forward(request, env, body) {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  headers.set(EDGE_AUTH_HEADER, env.EDGE_SHARED_SECRET);

  // Correlation only, and only when it is well-formed enough to be safe to
  // pass on. The origin mints its own when this is absent.
  const requestId = request.headers.get("X-Request-ID");
  if (requestId && /^[A-Za-z0-9-]{1,64}$/.test(requestId)) {
    headers.set("X-Request-ID", requestId);
  }

  const upstream = await fetch(`${env.ORIGIN_URL}${CHAT_PATH}`, {
    method: "POST",
    headers,
    body,
  });

  // The origin's body is returned as-is; its headers are not, so nothing the
  // origin sets can reach the browser unless this file decides it should.
  const out = new Headers({ "Content-Type": "application/json" });
  const echoed = upstream.headers.get("X-Request-ID");
  if (echoed) out.set("X-Request-ID", echoed);
  applyCors(out, request, env);
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

/**
 * @param {Request} request
 * @returns {string | null}
 */
function clientIp(request) {
  return request.headers.get("CF-Connecting-IP");
}

/**
 * @param {Headers} headers
 * @param {Request} request
 * @param {Env} env
 */
function applyCors(headers, request, env) {
  const origin = request.headers.get("Origin");
  const allowed = (env.ALLOWED_ORIGINS ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  if (origin && allowed.includes(origin)) {
    headers.set("Access-Control-Allow-Origin", origin);
    headers.set("Vary", "Origin");
  }
}

/**
 * @param {Request} request
 * @param {Env} env
 * @returns {Response}
 */
function preflight(request, env) {
  const headers = new Headers({
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": `Content-Type, ${TURNSTILE_HEADER}, X-Request-ID`,
    "Access-Control-Max-Age": "600",
  });
  applyCors(headers, request, env);
  return new Response(null, { status: 204, headers });
}

/**
 * A refusal. Short, stable, and saying nothing about why beyond the code —
 * no origin URL, no secret, no upstream detail, no stack.
 *
 * @param {number} status
 * @param {string} code
 * @param {string} message
 * @param {Record<string, string>} [extra]
 * @returns {Response}
 */
function refuse(status, code, message, extra = {}) {
  return new Response(JSON.stringify({ error: { code, message } }), {
    status,
    headers: { "Content-Type": "application/json", ...extra },
  });
}

export {
  CHAT_PATH,
  EDGE_AUTH_HEADER,
  MAX_BODY_BYTES,
  TURNSTILE_HEADER,
  readBounded,
};
