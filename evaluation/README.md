# Evaluation

The question Phase 5 could not answer: **how well does this actually retrieve, and does the
grounding hold?**

```bash
uv run portfolio-rag eval run                  # retrieval + grounding, configured providers
uv run portfolio-rag eval run --retrieval-only # no generation provider is called at all
uv run portfolio-rag eval run --generation-delay-seconds 8   # stay inside a provider's rate limit
uv run pytest tests/evaluation                 # the same dataset as a regression guard
```

## Running the whole set against a real provider

A dataset run is a burst — every question, one after another, as fast as the pipeline produces
them — and a free-tier account notices that long before a live site would. Mistral's limits for
`mistral-small` are **1 request per second and 20,000 tokens per minute**, and the token budget is
the binding one: spacing requests a second apart still exceeds it, and the run dies partway through
with `GENERATION_UNAVAILABLE` and nothing measured.

`--generation-delay-seconds` sets the minimum gap between two *generation* requests. It waits only
where a model is actually called, so questions the corpus cannot answer — which never reach a
provider — cost nothing. A rate limit that still gets through is retried a bounded number of times,
waiting as long as the provider's `Retry-After` asked for, and a retry re-issues one provider call
inside one question: nothing is measured twice.

This is evaluation wiring and nothing else. The server does not have it, cannot be configured into
it, and its per-request behaviour is unchanged.

## What is here

| File | What it is |
| --- | --- |
| `questions.yaml` | 24 questions in 8 categories, with hand-checkable ground truth |
| `corpus/` | 7 neutral fixture documents — 6 public, 1 `internal` |
| `portfolio-questions.yaml` | 49 questions against the real corpus in `../knowledge/`. Run against the production retrieval path — see [The real corpus](#the-real-corpus) |

Ground truth is a **document id plus a section heading**, never a chunk id: chunk ids move when the
chunking policy changes, and a dataset that breaks on a re-chunk is a dataset that gets deleted
rather than fixed.

## Read this before quoting a number

**The corpus is fixtures, not the portfolio.** Every number below was measured against `corpus/` —
7 neutral fixture documents describing an invented "Example Service" — and that has not changed.
`knowledge/` holds the 12 authorized portfolio documents about Çağdaş Uçar, and **no measurement
recorded below was taken against them.** These numbers therefore measure the *pipeline, the ground
truth and the parameters*, and remain valid as exactly that baseline. They do not say how well this
system answers questions about a real person — that is a separate exercise, and it has been carried
out against the production path rather than left undone ([The real corpus](#the-real-corpus)).

**The offline embedding provider has no semantics.** `deterministic` derives vectors from SHA-256,
so its similarities are noise. It scores **0/14 on every hit rate**, which is a true statement about
that provider and says nothing about the pipeline. The meaningful offline numbers below come from a
lexical (bag-of-words) double in `tests/doubles.py` — real word overlap, deterministic, free, and
still not a semantic model.

**Nothing here predicts the production provider.** A cosine distribution is a property of the
embedding model, so none of the figures below transfers to `mistral-embed`. The real corpus is
measured with its own dataset, against the real providers.

## Results

Measured 2026-08-11 on the fixture corpus (7 documents, 25 chunks) with `top_k=5`,
`min_similarity=0.25` — the Phase 5 defaults, unchanged.

| | deterministic (shipped offline) | lexical double |
| --- | --- | --- |
| hit@1 | 0/14 (0%) | 5/14 (36%) |
| hit@3 | 0/14 (0%) | 8/14 (57%) |
| hit@5 | 0/14 (0%) | 8/14 (57%) |
| MRR | 0.000 | 0.464 |
| internal leaks | **0** | **0** |

Four of the six misses under the lexical double are the `paraphrased` and `ambiguous` questions —
written to share no vocabulary with their answers, which is exactly what bag-of-words cannot do and
exactly what a real embedding model is for. Reading 57% as "retrieval quality" would be wrong.

### The threshold sweep, and what it settled

| `min_similarity` | hit@5 | unknown/internal fully filtered | internal leaks |
| --- | --- | --- | --- |
| 0.00 / 0.15 | 11/14 (79%) | 0/5 | 0 |
| **0.25 (default)** | 8/14 (57%) | 0/5 | 0 |
| 0.35 | 8/14 (57%) | 2/5 | 0 |
| 0.45 | 4/14 (29%) | 3/5 | 0 |

The similarity ranges of answerable and unanswerable questions **overlap completely** — answerable
0.257–0.791, unanswerable 0.265–0.650. No threshold separates them. Raising it buys refusals at
roughly one real answer each; lowering it recovers answers and rejects no more noise.

Two conclusions, and they are the most useful thing this dataset produced:

1. **A threshold filters noise, not topic.** What actually stops the system answering a question the
   corpus does not cover is the grounding path — a model that declines, and a backend that refuses
   to publish an answer with no verified citation. Phase 5 built that as a belt-and-braces measure;
   the evaluation shows it is load-bearing.
2. **The value is provider-specific.** Cosine distributions differ per embedding model, so 0.25
   tuned here would mean nothing under `mistral-embed`. It was therefore **left unchanged** — tuning
   a production default to a test double is exactly the benchmark overfitting worth avoiding.

`top_k` at 3, 5 and 8 produced identical results: the threshold cuts before depth binds, so depth is
not the limiting factor and was left at 5.

### Grounding

Scored separately from retrieval, with the model's *judgement* injected — declining for everything
the corpus cannot publicly answer — so that what is measured is the pipeline's routing rather than a
stub's behaviour.

| | result |
| --- | --- |
| answerable questions answered with a verified citation | 14/14 |
| unknown + internal-only questions refused with no citation | 5/5 |
| refusals using the fixed insufficient-knowledge sentence | 5/5 |
| adversarial questions leaking the canary or the forged label | 0/5 |
| internal leaks at thresholds −1.0, 0.0, 0.25, 0.5 | 0 at every one |

Note *which* refusal path did the work: all 10 refusals came back `NOT_GROUNDED`, and none came
back `NO_KNOWLEDGE`. On this corpus the threshold rejected nothing, so the short circuit never
fired and every refusal was earned after generation, by the backend declining to publish an answer
with no verifiable source. That is the same conclusion the sweep reached, arriving from the other
direction — and it is why removing either half would be a mistake: the short circuit saves the
provider call when it can, and the citation check is what actually holds when it cannot.

## The real corpus

The dataset in `portfolio-questions.yaml` — 49 questions against the 12 documents in `../knowledge/`
— has been run against the **production retrieval path**: `mistral-embed` for embeddings and
Cloudflare Vectorize as the store, which is the combination the deployed service uses. That answers
the question the fixture numbers above deliberately cannot.

Production runs at the shipped policy defaults, `top_k` 5 and `min_similarity` **0.250**
(`portfolio_rag.rag.policy`), unchanged by that exercise.

**No per-question metrics from that run are recorded in this repository**, so none are quoted here —
this file does not carry numbers it cannot show you the provenance of. To reproduce them, run the
dataset with the production embedding configuration in the environment:

```bash
uv run portfolio-rag eval run --dataset evaluation/portfolio-questions.yaml
```

Two caveats survive the measurement, because they are properties of the approach rather than of a
result. A threshold filters noise, not topic — the grounding path is still what refuses a question
the corpus does not cover, and it is still the part that must not be weakened. And a measurement is
valid for the corpus and the model it was taken on: changing either, or re-chunking, invalidates it,
and the dataset is re-run rather than assumed.

## The categories

| Category | n | Scored on |
| --- | --- | --- |
| `direct` | 4 | hit@k, MRR |
| `paraphrased` | 3 | hit@k, MRR |
| `multi_source` | 2 | hit@k, MRR |
| `section` | 3 | hit@k, MRR |
| `ambiguous` | 2 | hit@k, MRR |
| `unknown` | 3 | refusal in the answer path; threshold rejection as calibration |
| `internal` | 2 | the same, plus zero leakage as a hard gate |
| `adversarial` | 5 | structural only — no leak, no forged citation |

Adversarial questions are deliberately **not** scored as failed refusals. Asking "ignore your
instructions" retrieves the fixture document containing those words, and retrieving a document is
not obeying it. What must hold for them is structural, and that is asserted directly.

## The two planted markers

`corpus/internal-operations.md` (visibility `internal`) contains `CANARY-INTERNAL-ONLY-9F3A2B`, and
`corpus/untrusted-content.md` (public) contains instruction text and the label `S99`. Both are
invented for these tests: the canary grants nothing, and `S99` is a label this backend never mints,
so a model repeating it produces no citation. They exist so a leak or a forged citation would be
unmistakable rather than a judgement call.
