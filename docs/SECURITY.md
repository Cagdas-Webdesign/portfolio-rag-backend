# Security boundaries

What this system enforces, what it does not, and where the difference lies. Written for somebody
deciding whether to put it on the public internet.

**Scope.** A public, read-only knowledge assistant. No accounts, no sessions, no payments, no
personal data store, no user-generated content. The corpus is material its author chose to publish.
That scope is why there is no auth system here, and why adding one would be answering a question
nobody asked.

Nothing below claims the system is "secure". Each row says what is enforced, how, and what remains.

---

## 1. Enforced structurally

These hold regardless of what a language model does, what a knowledge document says, or what a
request contains. Each is a property of the code path, not of a prompt.

| Boundary | How it is enforced | Verified by |
| --- | --- | --- |
| A public request can only retrieve `visibility: public` documents | The filter is built into the query `PublicRetrievalService` sends. Not a parameter, not a policy field, not a keyword argument — the request that would ask for internal knowledge cannot be expressed | `tests/unit/test_retrieval_service.py`, `tests/evaluation/` at four thresholds |
| A document reclassified to `internal` after indexing stops being usable immediately | Resolved chunks are re-checked against the corpus, which is newer than the index | `test_retrieval_service.py::test_an_index_that_still_calls_a_document_public_does_not_override_the_corpus` |
| A model cannot name a source | Citations are built only from `RetrievedChunk`s this request retrieved; the model may select a backend-minted label and nothing else | `tests/unit/test_citations.py`, `tests/integration/test_adversarial.py` |
| A model cannot forge a label | Labels are minted per answer and mean nothing outside it; unknown labels are dropped and counted | `test_adversarial.py::test_a_forged_label_in_a_document_never_becomes_a_source` |
| An answer with no verifiable source is never published | Replaced by the fixed insufficient-knowledge sentence | `test_failure_matrix.py::test_an_answer_with_no_verifiable_source_is_replaced_not_published` |
| Knowledge content cannot become an instruction | Instructions are a system message; passages and the question are a user message. No configuration or document text can move a passage into the instruction role | `tests/unit/test_prompt_builder.py`, `test_adversarial.py` |
| A malformed provider reply produces no answer | Fails closed to `503`; the raw text never reaches the client | `test_failure_matrix.py` |
| Query spaces and index spaces cannot be mixed | Compared by `EmbeddingSpec` identity, not dimensionality | `test_retrieval_service.py::test_equal_dimensionality_is_not_compatibility` |
| Development stand-ins cannot run in production | The composition root refuses to build them when `PORTFOLIO_RAG_ENVIRONMENT=production`; the process fails to start | `tests/integration/test_composition.py`, and a CI step against the built image |
| No response carries internals | `ChatResponse` has three named fields; scores, labels, context, prompts, vector ids and timings are not among them | `test_chat.py::test_the_response_carries_no_internals` |
| No error carries a stack trace or a provider payload | One envelope, messages composed locally from status codes | `test_failure_matrix.py`, every cell |

## 2. Prompt injection — what is true

**It is not solved. Nothing solves it, and no prompt can.**

The realistic threat here is not a hostile user message — that content is obviously untrusted and
changes nothing the backend decides. It is a hostile *knowledge document*, which arrives inside the
context, in the position where a naive pipeline puts its own trusted material. The evaluation corpus
contains one on purpose (`evaluation/corpus/untrusted-content.md`), carrying instruction text and a
forged source label, and `tests/integration/test_adversarial.py` follows it through every stage.

What was demonstrated, including against a model scripted to have *fallen for it completely*:

* it cannot become a system message or add a chat role;
* it cannot change the retrieval policy, the temperature, the response format or the token budget;
* it cannot authorise a citation — the forged label maps to nothing;
* it cannot reach internal knowledge, which was never in the context to begin with;
* an answer it induced is not published, because none of its claimed sources verify.

What remains: a sufficiently persuasive passage can still influence what a model *writes*. The
mitigation is that whatever it writes is published only with sources the backend proved, and
otherwise replaced. Since the corpus is authored and reviewed in git by the same person who owns the
deployment, the realistic exposure is small — but it is not zero, and it is not closed by prompting.

## 3. What external providers receive

| Provider | Receives | Never receives |
| --- | --- | --- |
| Embedding provider | At indexing: document title, heading path, chunk content. At query time: the normalized question | Visibility, trust level, source paths, fingerprints, ids, internal documents |
| Generation provider | Grounding instructions, the selected **public** passages under their labels, the question | The corpus, internal documents, fingerprints, source paths, chunk ids, vectors, other settings |
| Vector store | Vectors, and a chosen metadata list (ids, title, heading path, source path, type, language, visibility, trust level, topics, technologies) | Chunk text — it is resolved locally from `knowledge/` |
| Public client | The answer, and verified citations (document id, title, source, section) | Everything else |

## 4. Logging

Every `extra=` field logged anywhere in `src/` was enumerated and reviewed. The complete set is
counts, durations, identifiers, statuses, the embedding space, the model name and the prompt
version. Not logged, at any level: the question, the context, the prompt, the answer, embedding
vectors, credentials, `Authorization` headers, provider response bodies, internal knowledge.

Provider error messages are composed locally from a status code and the operation, never echoed
from upstream — so a provider that returns a secret in an error body cannot put it in a log line.

## 5. Input bounds

Three, in the order a request meets them:

| Bound | Where | Refusal |
| --- | --- | --- |
| 16 KiB request body | `api/middleware/gate.py`, before routing | `413`, body never parsed |
| 2000 characters per message | the HTTP schema | `422` |
| 4000 characters per query | `rag/query.py` | the pipeline's own bound |

The last two are two numbers on purpose, and the only place in this project where that is the right
answer rather than a copy waiting to drift. They bound different things: the pipeline bound applies
to the CLI and the evaluation harness as well, while the public bound is what an anonymous web page
may send. A test asserts the invariant that keeps them honest — the public one can only ever be the
stricter of the pair.

Empty, whitespace-only and invisible-only messages are rejected before any provider call. Unknown
fields are rejected rather than ignored. A rejected value is never echoed back.

## 5a. The edge boundary

`POST /api/v1/chat` refuses any request that did not come through the gateway. The gateway sets
`X-Edge-Auth`; the origin compares it in constant time before routing, body parsing or any
dependency is resolved, and answers `403` otherwise — one string comparison, no embedding, no
Vectorize query, no generation.

It authenticates the *gateway*, not a user: there are no accounts here. What it buys is that the
public Cloud Run URL cannot be used to skip the bot check, the challenge and the rate limiter in
front of it. On by default in production, off elsewhere, and a production app that names no secret
refuses to start rather than starting unguarded. `/health` is outside the guard so a platform probe
still works.

The chain, and the manual steps that arm it, are in
[DEPLOYMENT.md](DEPLOYMENT.md#6-configure-the-edge); the gateway itself is in
[`edge/`](../edge/README.md).

Nothing is scrubbed. Angle brackets, braces, backticks and quotes survive, because
`How is <T> serialized?` is a legitimate question here and the message never reaches a shell, a
path, a template or a query — only an embedding input and a labelled section of a user message.

## 6. What the deployment must provide

| Requirement | Why it is not in the application |
| --- | --- |
| **Rate limiting on `POST /api/v1/chat`** | The application cannot identify a client correctly without knowing its proxy topology. See [ADR 0008](adr/0008-abuse-boundary-at-the-edge.md) |
| **TLS termination** | The application speaks HTTP; transport security belongs to the edge |
| **`PORTFOLIO_RAG_ALLOWED_ORIGINS` set to the real origin** | CORS restricts *browsers*, and only if configured. It is not access control |
| **Secrets from the environment or a secret store** | Nothing is baked into the image; the container carries no credential |

**CORS is not a security boundary.** It stops a browser on another origin from reading responses. It
stops nothing else — not curl, not a script, not a server. Wildcards are rejected by configuration
validation, and `allow_credentials` is `False` because there is no cookie or credential to send.

## 7. Residual risks, named

* **No rate limit in the application itself.** The current deployment has the gateway in front of it
  and the origin guard armed, but any deployment that skips either can have its free-tier quota
  drained. The consequence is `503`s, not exposure.
* **Prompt injection via a hostile knowledge document** can influence answer *text*. Sources remain
  backend-verified.
* **A retrieval measurement ages.** The real corpus has been evaluated on the production path
  (`mistral-embed` into Vectorize) and `PORTFOLIO_RAG_RETRIEVAL_MIN_SIMILARITY` is set from that
  rather than guessed — but a measurement is only valid for the corpus, model and chunking policy it
  was taken on. Change any of those without re-running
  [step 4 of DEPLOYMENT.md](DEPLOYMENT.md#4-measure-retrieval-against-the-real-corpus) and the
  threshold silently stops meaning what it meant. What that risks is a thin or over-eager retrieval
  set, not an unsupported answer — the grounding path is what refuses.
* **The similarity threshold cannot distinguish an out-of-scope question from an in-scope one**
  (measured, see `evaluation/README.md`). The grounding path is what refuses, and it is the part
  that must not be weakened.
* **A stale index** can surface a chunk id the corpus no longer has. It is dropped and counted, so
  the effect is a thinner answer, not a wrong one.
* **No audit trail.** Nothing records who asked what, deliberately — there is no user to attribute a
  question to, and storing questions would create a privacy obligation the system does not need.
