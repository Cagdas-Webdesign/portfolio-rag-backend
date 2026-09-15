"""Measuring whether retrieval and grounding actually work.

The harness is small on purpose — a dataset file, four metrics and a runner — because the
alternative on a corpus this size is an evaluation framework producing a score
that nobody can trace back to a question.

    dataset (YAML, hand-checkable ground truth)
      → retrieval run    → hit@1 / hit@3 / hit@5 / MRR / refusal correctness
      → grounding run    → answered / no-knowledge / not-grounded, per question
      → failures         → named individually, never averaged away

**What this measures and what it does not.** It measures the pipeline, the
ground truth and the parameters: chunking, the similarity threshold, `top_k`,
the refusal paths and citation validation. It cannot measure how well a
particular embedding model understands language — that is a property of the
configured provider, and running the same dataset against a different one is
the only way to find out.

No external evaluation framework, and no model judging another model's answers:
on a set this small, a human reading twenty rows is both cheaper and more
trustworthy than a judge whose own errors nobody is checking.
"""

from portfolio_rag.evaluation.dataset import (
    EvaluationDataset,
    EvaluationDatasetError,
    EvaluationQuestion,
    ExpectedSource,
    QuestionCategory,
    load_dataset,
    resolve_corpus,
)
from portfolio_rag.evaluation.metrics import (
    HIT_RATE_CUTOFFS,
    CategoryScore,
    RetrievalOutcomeRecord,
    RetrievalReport,
    build_report,
    score_retrieval,
)
from portfolio_rag.evaluation.pacing import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_WAIT_SECONDS,
    GenerationPacing,
    PacedLLMProvider,
)
from portfolio_rag.evaluation.runner import (
    GroundingRecord,
    GroundingReport,
    run_grounding_evaluation,
    run_retrieval_evaluation,
)

__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_WAIT_SECONDS",
    "HIT_RATE_CUTOFFS",
    "CategoryScore",
    "EvaluationDataset",
    "EvaluationDatasetError",
    "EvaluationQuestion",
    "ExpectedSource",
    "GenerationPacing",
    "GroundingRecord",
    "GroundingReport",
    "PacedLLMProvider",
    "QuestionCategory",
    "RetrievalOutcomeRecord",
    "RetrievalReport",
    "build_report",
    "load_dataset",
    "resolve_corpus",
    "run_grounding_evaluation",
    "run_retrieval_evaluation",
    "score_retrieval",
]
