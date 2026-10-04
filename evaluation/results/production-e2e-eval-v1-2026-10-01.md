# End-to-end evaluation

Format `e2e-eval-v1`. Every question was asked once and went through the whole pipeline: query embedding, vector search, generation, grounding and citation validation. Retrieval is scored from the retrieval each answer actually used.

**Gates: FAIL**

## Run

| | |
| --- | --- |
| Generated | 2026-10-01T18:32:42+00:00 |
| Git revision | `12faec53f680afbc94eec5af44b6bf4b21a6a55d` (dirty) |
| Dataset | `evaluation/portfolio-questions.yaml` v1, sha256 `54afed0b6798` |
| Suite | full, 49 questions |
| Corpus | 12 documents, 151 chunks, sha256 `5a180eb59719` |
| Embedding | `mistral:mistral-embed:1024:embedding-text-v1` |
| Vector store | vectorize |
| Generation | deterministic, `context-echo-v1`, prompt `grounded-answer-v3` |
| Retrieval policy | top_k 5, min_similarity 0.25 |

## Gates

Properties that must hold whatever the quality numbers are.

| Gate | Result |
| --- | --- |
| no_pipeline_errors | pass |
| citations_verified | pass |
| unanswerable_refused | FAIL |
| nothing_internal_published | pass |

## Measurements

Counts with their denominators. No pass mark is attached to any of them.

| Metric | Result |
| --- | --- |
| Retrieval hit@1 | 33/44 (75.0%) |
| Retrieval hit@3 | 40/44 (90.9%) |
| Retrieval hit@5 | 41/44 (93.2%) |
| Retrieval MRR | 0.830 |
| Answerable questions answered | 44/44 (100.0%) |
| Answers citing an expected source | 41/44 (93.2%) |
| Unanswerable questions refused | 0/5 (0.0%) |
| Published citations verified | 243/243 (100.0%) |
| Source labels invented by the model | 0 |
| Pipeline errors | 0 |
| Empty answers | 0 |
| Internal label leaks | 0 |
| Internal passages retrieved | 0 |
| Stale index matches dropped | 0 |
| Seconds per question, median / max | 8.001 / 8.002 (pacing included) |

Outcomes: 49 answered, 0 no_knowledge, 0 not_grounded, 0 error.

## Failures

`retrieval_miss` and `not_answered` are quality findings. Every other reason breaks a gate.

| Question | Category | Outcome | Reasons |
| --- | --- | --- | --- |
| `paraphrased-scroll-animation` | paraphrased | answered | retrieval_miss |
| `unknown-phone-number` | unknown | answered | answered_unknown |
| `unknown-major-clients` | unknown | answered | answered_unknown |
| `unknown-salary` | unknown | answered | answered_unknown |
| `unknown-kubernetes-production` | unknown | answered | answered_unknown |
| `unknown-medical-training` | unknown | answered | answered_unknown |
| `broad-ai-technologies` | multi_source | answered | retrieval_miss |
| `broad-project-scope` | multi_source | answered | retrieval_miss |

## What this does not measure

- Whether an answer's wording is correct or complete. No model judges another model here; the answers are in the data file to be read.
- A semantically fitting passage that the ground truth does not list still counts as a retrieval miss.
- Similarities are similarities in the index's metric, not confidences.
- One run of a non-deterministic generation model. A second run can differ.

## Notes for this run

- After the last index sync, knowledge index --dry-run still plans 1 re-embed and 7 metadata-only updates for professional-profile, while a direct read of professional-profile--0004 matches the locally computed fingerprints. Not resolved before this run.
