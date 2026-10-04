# End-to-end evaluation

Format `e2e-eval-v2`. Every question was asked once and went through the whole pipeline: query embedding, vector search, generation, grounding and citation validation. Retrieval is scored from the retrieval each answer actually used.

**Gates: FAIL**

## Run

| | |
| --- | --- |
| Generated | 2026-10-02T09:19:53+00:00 |
| Git revision | `12faec53f680afbc94eec5af44b6bf4b21a6a55d` (dirty) |
| Dataset | `evaluation/portfolio-questions.yaml` v1, sha256 `54afed0b6798` |
| Suite | full, 3 questions (selected by id) |
| Corpus | 12 documents, 151 chunks, sha256 `5a180eb59719` |
| Embedding | `mistral:mistral-embed:1024:embedding-text-v1` |
| Vector store | vectorize |
| Generation | cloudflare_workers_ai, `@cf/openai/gpt-oss-120b`, prompt `grounded-answer-v6`, check `grounding-check-v1` |
| Retrieval policy | top_k 5, min_similarity 0.25 |

## Gates

Properties that must hold whatever the quality numbers are.

| Gate | Result |
| --- | --- |
| no_pipeline_errors | FAIL |
| citations_verified | pass |
| unanswerable_refused | pass |
| nothing_internal_published | pass |

## Measurements

Counts with their denominators. No pass mark is attached to any of them.

| Metric | Result |
| --- | --- |
| Questions the pipeline completed | 0/3 (0.0%) |
| Questions with an observed retrieval | 3/3 (100.0%) |
| Retrieval hit@1 | 2/3 (66.7%) |
| Retrieval hit@3 | 2/3 (66.7%) |
| Retrieval hit@5 | 2/3 (66.7%) |
| Retrieval MRR | 0.667 |
| Answerable questions answered | 0/3 (0.0%) |
| Answers citing an expected source | 0/0 |
| Unanswerable questions refused | 0/0 |
| Published citations verified | 0/0 |
| Source labels invented by the model | 0 |
| Pipeline errors | 3 |
| Empty answers | 0 |
| Internal label leaks | 0 |
| Internal passages retrieved | 0 |
| Stale index matches dropped | 0 |
| Seconds per question, median / max | None / None (pacing included) |

Outcomes: 0 answered, 0 no_knowledge, 0 not_grounded, 3 error.

## Failures

`retrieval_miss` and `not_answered` are quality findings. Every other reason breaks a gate.

| Question | Category | Outcome | Reasons |
| --- | --- | --- | --- |
| `broad-deployment` | multi_source | error | pipeline_error, not_answered |
| `broad-ai-technologies` | multi_source | error | pipeline_error, retrieval_miss, not_answered |
| `broad-frontend-backend-in-portfolio` | multi_source | error | pipeline_error, not_answered |

## Pipeline errors

Retrieval is scored for these questions wherever it was observed; the error counts against availability, not against retrieval.

| Question | Code | Category | Detail | Status | Attempts | Finish reason |
| --- | --- | --- | --- | --- | --- | --- |
| `broad-deployment` | GENERATION_UNAVAILABLE | retryable_provider_error | rate_limited | 429 | 3 | — |
| `broad-ai-technologies` | GENERATION_UNAVAILABLE | retryable_provider_error | rate_limited | 429 | 3 | — |
| `broad-frontend-backend-in-portfolio` | GENERATION_UNAVAILABLE | retryable_provider_error | rate_limited | 429 | 3 | — |

## What this does not measure

- Whether an answer's wording is correct or complete. No model judges another model here; the answers are in the data file to be read.
- A semantically fitting passage that the ground truth does not list still counts as a retrieval miss.
- Similarities are similarities in the index's metric, not confidences.
- One run of a non-deterministic generation model. A second run can differ.
