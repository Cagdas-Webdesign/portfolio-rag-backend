/**
 * The gateway's refusals, with nothing real behind them.
 *
 * `fetch` is stubbed, both rate limiters are stubs, and there is no account,
 * no Turnstile secret and no origin. What every test here is really asserting
 * is the same thing twice: the right status came back, and the origin was
 * never called.
 *
 * Run with:  node --test edge/test
 */

import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import worker, { EDGE_AUTH_HEADER, MAX_BODY_BYTES, TURNSTILE_HEADER } from "../src/worker.js";

const ORIGIN = "https://origin.invalid";
const CHAT_URL = "https://gateway.invalid/api/v1/chat";
const SECRET = "test-edge-secret-not-a-real-credential";
const TURNSTILE_SECRET = "test-turnstile-secret-not-a-real-credential";

/** Records every outbound fetch so a test can prove one did not happen. */
let calls = [];
const realFetch = globalThis.fetch;

function stubFetch({ turnstileOk = true, originStatus = 200 } = {}) {
  calls = [];
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    if (String(url).includes("turnstile")) {
      return new Response(JSON.stringify({ success: turnstileOk }), { status: 200 });
    }
    return new Response(JSON.stringify({ answer: "ok", citations: [] }), {
      status: originStatus,
      headers: { "Content-Type": "application/json" },
    });
  };
}

function limiter(success) {
  return { limit: async () => ({ success }) };
}

function env({ burst = true, sustained = true } = {}) {
  return {
    ORIGIN_URL: ORIGIN,
    ALLOWED_ORIGINS: "https://portfolio.invalid",
    TURNSTILE_SECRET_KEY: TURNSTILE_SECRET,
    EDGE_SHARED_SECRET: SECRET,
    CHAT_BURST_LIMITER: limiter(burst),
    CHAT_SUSTAINED_LIMITER: limiter(sustained),
  };
}

function chatRequest({ method = "POST", body = '{"message":"hi"}', headers = {} } = {}) {
  const base = { [TURNSTILE_HEADER]: "a-token", "Content-Type": "application/json" };
  return new Request(CHAT_URL, {
    method,
    headers: { ...base, ...headers },
    body: method === "GET" || method === "OPTIONS" ? undefined : body,
  });
}

/** Calls that went to the origin rather than to Turnstile. */
function originCalls() {
  return calls.filter((call) => call.url.startsWith(ORIGIN));
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("routing", () => {
  it("refuses an unknown path without calling the origin", async () => {
    stubFetch();
    const response = await worker.fetch(
      new Request("https://gateway.invalid/api/v1/admin", { method: "POST" }),
      env(),
    );

    assert.equal(response.status, 404);
    assert.equal(originCalls().length, 0);
  });

  it("refuses a method that is not POST", async () => {
    stubFetch();
    for (const method of ["GET", "PUT", "DELETE", "PATCH"]) {
      const response = await worker.fetch(chatRequest({ method }), env());
      assert.equal(response.status, 405, method);
      assert.equal(response.headers.get("Allow"), "POST, OPTIONS");
    }
    assert.equal(originCalls().length, 0);
  });

  it("answers a preflight itself, without a challenge and without the origin", async () => {
    stubFetch();
    const response = await worker.fetch(
      new Request(CHAT_URL, {
        method: "OPTIONS",
        headers: { Origin: "https://portfolio.invalid" },
      }),
      env(),
    );

    assert.equal(response.status, 204);
    assert.equal(
      response.headers.get("Access-Control-Allow-Origin"),
      "https://portfolio.invalid",
    );
    assert.equal(originCalls().length, 0);
  });

  it("does not grant CORS to an origin that is not configured", async () => {
    stubFetch();
    const response = await worker.fetch(
      new Request(CHAT_URL, { method: "OPTIONS", headers: { Origin: "https://evil.invalid" } }),
      env(),
    );

    assert.equal(response.headers.get("Access-Control-Allow-Origin"), null);
  });
});

describe("body size", () => {
  it("refuses a body over the limit", async () => {
    stubFetch();
    const response = await worker.fetch(
      chatRequest({ body: "y".repeat(MAX_BODY_BYTES + 1) }),
      env(),
    );

    assert.equal(response.status, 413);
    assert.equal(calls.length, 0, "not even Turnstile was asked");
  });

  it("accepts a body at the limit", async () => {
    stubFetch();
    const filler = "y".repeat(MAX_BODY_BYTES - 20);
    const response = await worker.fetch(
      chatRequest({ body: JSON.stringify({ message: filler }).slice(0, MAX_BODY_BYTES) }),
      env(),
    );

    assert.equal(response.status, 200);
  });
});

describe("turnstile", () => {
  it("refuses a request with no token", async () => {
    stubFetch();
    const request = new Request(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: '{"message":"hi"}',
    });

    const response = await worker.fetch(request, env());

    assert.equal(response.status, 401);
    assert.equal(originCalls().length, 0);
  });

  it("refuses a token Cloudflare does not accept", async () => {
    stubFetch({ turnstileOk: false });

    const response = await worker.fetch(chatRequest(), env());

    assert.equal(response.status, 403);
    assert.equal(originCalls().length, 0, "a failed challenge never reaches the origin");
  });

  it("treats a verification that errors as a failure", async () => {
    calls = [];
    globalThis.fetch = async (url) => {
      calls.push({ url: String(url) });
      if (String(url).includes("turnstile")) throw new Error("network down");
      return new Response("{}", { status: 200 });
    };

    const response = await worker.fetch(chatRequest(), env());

    assert.equal(response.status, 403, "fail closed, not open");
    assert.equal(originCalls().length, 0);
  });

  it("forwards a request whose token verified", async () => {
    stubFetch();

    const response = await worker.fetch(chatRequest(), env());

    assert.equal(response.status, 200);
    assert.equal(originCalls().length, 1);
  });

  it("never returns the turnstile secret to the browser", async () => {
    stubFetch({ turnstileOk: false });

    const response = await worker.fetch(chatRequest(), env());

    assert.ok(!(await response.text()).includes(TURNSTILE_SECRET));
  });
});

describe("rate limiting", () => {
  it("refuses a burst with 429 and a Retry-After", async () => {
    stubFetch();

    const response = await worker.fetch(chatRequest(), env({ burst: false }));

    assert.equal(response.status, 429);
    assert.equal(response.headers.get("Retry-After"), "10");
    assert.equal(originCalls().length, 0);
  });

  it("refuses a sustained overload with 429 and a longer Retry-After", async () => {
    stubFetch();

    const response = await worker.fetch(chatRequest(), env({ sustained: false }));

    assert.equal(response.status, 429);
    assert.equal(response.headers.get("Retry-After"), "60");
    assert.equal(originCalls().length, 0);
  });

  it("keys the limiters on the client address and the route", async () => {
    stubFetch();
    const seen = [];
    const environment = env();
    environment.CHAT_BURST_LIMITER = {
      limit: async ({ key }) => {
        seen.push(key);
        return { success: true };
      },
    };

    await worker.fetch(chatRequest({ headers: { "CF-Connecting-IP": "203.0.113.7" } }), environment);

    assert.deepEqual(seen, ["203.0.113.7:/api/v1/chat"]);
  });
});

describe("the origin secret", () => {
  it("is set by the worker on a forwarded request", async () => {
    stubFetch();

    await worker.fetch(chatRequest(), env());

    assert.equal(originCalls()[0].init.headers.get(EDGE_AUTH_HEADER), SECRET);
  });

  it("replaces one a client tried to supply", async () => {
    stubFetch();

    await worker.fetch(
      chatRequest({ headers: { [EDGE_AUTH_HEADER]: "smuggled-by-the-client" } }),
      env(),
    );

    assert.equal(originCalls()[0].init.headers.get(EDGE_AUTH_HEADER), SECRET);
  });

  it("is never returned to the browser", async () => {
    stubFetch();

    const response = await worker.fetch(chatRequest(), env());
    const body = await response.text();

    assert.ok(!body.includes(SECRET));
    assert.equal(response.headers.get(EDGE_AUTH_HEADER), null);
  });

  it("is not sent to Turnstile", async () => {
    stubFetch();

    await worker.fetch(chatRequest(), env());

    const turnstile = calls.find((call) => call.url.includes("turnstile"));
    assert.ok(turnstile);
    assert.equal(turnstile.init.headers?.get?.(EDGE_AUTH_HEADER) ?? null, null);
  });
});

describe("what a refusal reveals", () => {
  it("never names the origin", async () => {
    stubFetch({ turnstileOk: false });

    const body = await (await worker.fetch(chatRequest(), env())).text();

    assert.ok(!body.includes(ORIGIN));
    assert.ok(!body.includes("run.app"));
  });

  it("carries a stable code and a short message", async () => {
    stubFetch({ turnstileOk: false });

    const body = await (await worker.fetch(chatRequest(), env())).json();

    assert.equal(body.error.code, "challenge_failed");
    assert.ok(body.error.message.length < 120);
    assert.ok(!("stack" in body.error));
  });
});
