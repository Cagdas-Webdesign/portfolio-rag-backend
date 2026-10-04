"""Measuring whether retrieval and grounding actually work.

The harness is small on purpose — a dataset file, four metrics and a runner — because the
alternative on a corpus this size is an evaluation framework producing a score
that nobody can trace back to a question.

    dataset (YAML, hand-checkable ground truth)
      → retrieval run    → hit@1 / hit@3 / hit@5 / MRR / refusal correctness
      → grounding run    → answered / no-knowledge / not-grounded, per question
      → failures         → named individually, never averaged away
      → export           → the retrieval run as JSON, scores included, for offline reading
      → end-to-end run   → each question once through the whole pipeline, as JSON and Markdown
      → acceptance       → whether one end-to-end export is a release acceptance, by fixed rule

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
    EvaluationSuite,
    ExpectedSource,
    QuestionCategory,
    SuiteIdentity,
    SuiteSelection,
    load_dataset,
    resolve_corpus,
    select_suite,
    suite_identity,
    suite_path,
)
from portfolio_rag.evaluation.e2e import (
    E2E_FORMAT_VERSION,
    GATES,
    CorpusIdentity,
    E2EFailure,
    E2ERecord,
    E2EReport,
    E2ERunMetadata,
    corpus_identity,
    export_e2e,
    render_summary,
    score_answer,
)
from portfolio_rag.evaluation.export import (
    EXPORT_FORMAT_VERSION,
    RunMetadata,
    dataset_fingerprint,
    export_retrieval,
    write_export,
)
from portfolio_rag.evaluation.metrics import (
    HIT_RATE_CUTOFFS,
    CategoryScore,
    RetrievalOutcomeRecord,
    RetrievalReport,
    build_report,
    is_relevant,
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
    run_e2e_evaluation,
    run_grounding_evaluation,
    run_retrieval_evaluation,
    validate_question_delay,
)

__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_WAIT_SECONDS",
    "E2E_FORMAT_VERSION",
    "EXPORT_FORMAT_VERSION",
    "GATES",
    "HIT_RATE_CUTOFFS",
    "CategoryScore",
    "CorpusIdentity",
    "E2EFailure",
    "E2ERecord",
    "E2EReport",
    "E2ERunMetadata",
    "EvaluationDataset",
    "EvaluationDatasetError",
    "EvaluationQuestion",
    "EvaluationSuite",
    "ExpectedSource",
    "GenerationPacing",
    "GroundingRecord",
    "GroundingReport",
    "PacedLLMProvider",
    "QuestionCategory",
    "RetrievalOutcomeRecord",
    "RetrievalReport",
    "RunMetadata",
    "SuiteIdentity",
    "SuiteSelection",
    "build_report",
    "corpus_identity",
    "dataset_fingerprint",
    "export_e2e",
    "export_retrieval",
    "is_relevant",
    "load_dataset",
    "render_summary",
    "resolve_corpus",
    "run_e2e_evaluation",
    "run_grounding_evaluation",
    "run_retrieval_evaluation",
    "score_answer",
    "score_retrieval",
    "select_suite",
    "suite_identity",
    "suite_path",
    "validate_question_delay",
    "write_export",
]
