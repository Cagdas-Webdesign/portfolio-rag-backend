# End-to-end evaluation

Format `e2e-eval-v2`. Every question was asked once and went through the whole pipeline: query embedding, vector search, generation, grounding and citation validation. Retrieval is scored from the retrieval each answer actually used.

**Gates: PASS**

## Run

| | |
| --- | --- |
| Generated | 2026-10-02T08:32:24+00:00 |
| Git revision | `12faec53f680afbc94eec5af44b6bf4b21a6a55d` (dirty) |
| Dataset | `evaluation/portfolio-questions.yaml` v1, sha256 `54afed0b6798` |
| Suite | full, 1 questions (selected by id) |
| Corpus | 12 documents, 151 chunks, sha256 `5a180eb59719` |
| Embedding | `mistral:mistral-embed:1024:embedding-text-v1` |
| Vector store | vectorize |
| Generation | cloudflare_workers_ai, `@cf/openai/gpt-oss-120b`, prompt `grounded-answer-v6`, check `grounding-check-v1` |
| Retrieval policy | top_k 5, min_similarity 0.25 |

## Gates

Properties that must hold whatever the quality numbers are.

| Gate | Result |
| --- | --- |
| no_pipeline_errors | pass |
| citations_verified | pass |
| unanswerable_refused | pass |
| nothing_internal_published | pass |

## Measurements

Counts with their denominators. No pass mark is attached to any of them.

| Metric | Result |
| --- | --- |
| Questions the pipeline completed | 1/1 (100.0%) |
| Questions with an observed retrieval | 1/1 (100.0%) |
| Retrieval hit@1 | 1/1 (100.0%) |
| Retrieval hit@3 | 1/1 (100.0%) |
| Retrieval hit@5 | 1/1 (100.0%) |
| Retrieval MRR | 1.000 |
| Answerable questions answered | 1/1 (100.0%) |
| Answers citing an expected source | 1/1 (100.0%) |
| Unanswerable questions refused | 0/0 |
| Published citations verified | 4/4 (100.0%) |
| Source labels invented by the model | 0 |
| Pipeline errors | 0 |
| Empty answers | 0 |
| Internal label leaks | 0 |
| Internal passages retrieved | 0 |
| Stale index matches dropped | 0 |
| Seconds per question, median / max | 20.09 / 20.09 (pacing included) |

Outcomes: 1 answered, 0 no_knowledge, 0 not_grounded, 0 error.

## Failures

`retrieval_miss` and `not_answered` are quality findings. Every other reason breaks a gate.

None.

## What this does not measure

- Whether an answer's wording is correct or complete. No model judges another model here; the answers are in the data file to be read.
- A semantically fitting passage that the ground truth does not list still counts as a retrieval miss.
- Similarities are similarities in the index's metric, not confidences.
- One run of a non-deterministic generation model. A second run can differ.
