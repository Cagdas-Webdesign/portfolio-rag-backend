"""Developer CLI — the second entry point into the application.

``main.py`` serves HTTP; this module serves a terminal. Both are delivery
mechanisms: they parse input, call into the application and format the result.
No pipeline logic lives here.

    portfolio-rag knowledge validate
    portfolio-rag knowledge inspect <id-or-path>
    portfolio-rag knowledge chunks <id-or-path> [--show-content]
    portfolio-rag knowledge chunks --all
    portfolio-rag knowledge embedding <chunk-or-document> [--show-text]
    portfolio-rag knowledge index [--dry-run] [--rebuild]
    portfolio-rag query retrieve "<question>" [--top-k N] [--min-similarity F]
    portfolio-rag query answer "<question>" [--show-retrieval] [--show-context]
    portfolio-rag eval run [--dataset PATH] [--retrieval-only]
    portfolio-rag eval run [--generation-delay-seconds F]
    portfolio-rag eval run [--suite full|smoke] [--output PATH]
    portfolio-rag eval run [--retrieval-delay-seconds F]
    portfolio-rag eval run --e2e --output PATH [--summary PATH] [--note TEXT]
    portfolio-rag eval run [--question-id ID]

Built on ``argparse`` from the standard library. A CLI framework would be a
runtime dependency bought for a handful of subcommands.

This module is where the pipeline is joined end to end: the loader reads files,
the chunker cuts them, and the composition root supplies the adapters. None of
those call each other; they meet here, at the edge. The orchestration lives in
:mod:`portfolio_rag.application.indexing` and :mod:`portfolio_rag.rag`, not in
this file.

**The CLI shows more than the API does, on purpose.** Similarity scores, source
labels, the assembled context — a developer debugging retrieval needs all of
it, and a browser client needs none of it. What the CLI still never prints is a
credential, an ``Authorization`` header or a provider payload, and none of its
diagnostic output is logged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NamedTuple, TextIO

from pydantic import ValidationError

from portfolio_rag import SERVICE_NAME, __version__
from portfolio_rag.application.indexing import IndexingError, IndexPlan
from portfolio_rag.composition import (
    ConfigurationError,
    QueryComponents,
    build_embedding_spec,
    build_indexing_components,
    build_query_components,
)
from portfolio_rag.core.config import (
    EmbeddingProviderName,
    LLMProviderName,
    Settings,
    get_settings,
)
from portfolio_rag.core.errors import AppError
from portfolio_rag.core.request_context import new_request_id, reset_request_id, set_request_id
from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.domain.knowledge import KnowledgeChunk, KnowledgeDocument
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.evaluation import (
    E2EReport,
    E2ERunMetadata,
    EvaluationDataset,
    EvaluationDatasetError,
    EvaluationQuestion,
    EvaluationSuite,
    GenerationPacing,
    GroundingReport,
    PacedLLMProvider,
    RetrievalReport,
    RunMetadata,
    corpus_identity,
    dataset_fingerprint,
    export_e2e,
    export_retrieval,
    load_dataset,
    render_summary,
    resolve_corpus,
    run_e2e_evaluation,
    run_grounding_evaluation,
    run_retrieval_evaluation,
    select_suite,
    suite_identity,
    validate_question_delay,
    write_export,
)
from portfolio_rag.evaluation.acceptance import (
    SOURCE_IDENTITY_EXCLUDES,
    SOURCE_IDENTITY_INCLUDES,
    source_identity,
    validate_artifact,
    with_release_acceptance,
)
from portfolio_rag.evaluation.budget import (
    DAILY_BUDGET_NEURONS,
    MINIMUM_RESERVE_NEURONS,
    BudgetPolicy,
    BudgetZone,
    Ledger,
    LedgerEntry,
    cost_profile,
    spent,
)
from portfolio_rag.evaluation.e2e import AbortReason
from portfolio_rag.evaluation.experiment import (
    DEFAULT_VARIANTS,
    ExperimentError,
    ExperimentLimits,
    ExperimentMetadata,
    ExperimentRun,
    PreparedQuestion,
    StopReason,
    export_experiment,
    plan_calls,
    prepare_questions,
)
from portfolio_rag.evaluation.export import GIT_DIRTY_EXCLUDES
from portfolio_rag.evaluation.operations import (
    TIER_SUITES,
    Preflight,
    ProfileRequirements,
    RunGuard,
    SmokeVerdict,
    Tier,
    describe_preflight,
    load_history,
    preflight,
    smoke_verdict,
)
from portfolio_rag.ingestion import KnowledgeBaseReport, collect_knowledge_base
from portfolio_rag.ingestion.chunking import (
    ChunkingError,
    ChunkingPolicy,
    ChunkStatistics,
    chunk_document,
    chunk_knowledge_base,
)
from portfolio_rag.ingestion.embedding import (
    EMBEDDING_REPRESENTATION_VERSION,
    build_embedding_text,
    compute_embedding_fingerprint,
)
from portfolio_rag.ports.errors import PortError
from portfolio_rag.ports.llm import ResponseFormat
from portfolio_rag.rag.context import GroundedContext
from portfolio_rag.rag.errors import QueryValidationError
from portfolio_rag.rag.policy import DEFAULT_CONTEXT_POLICY, RetrievalPolicy
from portfolio_rag.rag.prompt import GROUNDED_PROMPT_VERSION, GROUNDED_RESPONSE_FORMAT
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import RetrievalOutcome
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswer, GroundedAnswerService
from portfolio_rag.rag.verification import GROUNDING_CHECK_VERSION

#: Exit codes. 2 is left to argparse for usage errors.
EXIT_OK: Final = 0
EXIT_INVALID: Final = 1
EXIT_INTERNAL_ERROR: Final = 3
#: An acceptance-tier run that finished — or stopped — without a passing
#: release-acceptance verdict, for any reason the verdict names: a gate, a
#: dirty tree, an aborted or incomplete run, missing provenance. Apart from
#: EXIT_INVALID so release automation can tell "not a release acceptance" from
#: "the command was used wrongly".
EXIT_ACCEPTANCE_FAILED: Final = 4

DEFAULT_KNOWLEDGE_ROOT: Final = Path("knowledge")
DEFAULT_EVALUATION_DATASET: Final = Path("evaluation/questions.yaml")

#: Transport attempts the generation adapter gets during a *paced* evaluation.
#: One, because the pacer above it is doing the retrying — with the provider's
#: own ``Retry-After`` rather than a fixed second — and two bounded budgets in
#: series multiply into a burst aimed at an account that is already refusing.
EVALUATION_GENERATION_ATTEMPTS: Final = 1

_LABEL_WIDTH: Final = 15


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code rather than calling `sys.exit`."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return _dispatch(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    except Exception as exc:  # the CLI's outermost boundary
        # A crash must never be mistaken for a clean validation run. The type
        # is shown so the failure is identifiable; the traceback is not, because
        # a wall of stack frames is not a usable error message.
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_INTERNAL_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="portfolio-rag",
        description=f"{SERVICE_NAME} developer tools.",
    )
    parser.add_argument("--version", action="version", version=f"{SERVICE_NAME} {__version__}")

    groups = parser.add_subparsers(dest="group", required=True)
    knowledge = groups.add_parser("knowledge", help="Inspect and validate the knowledge base.")
    commands = knowledge.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate",
        help="Ingest every knowledge document and report problems.",
    )
    _add_root_option(validate)

    inspect = commands.add_parser(
        "inspect",
        help="Show what ingestion made of a single document.",
    )
    _add_root_option(inspect)
    inspect.add_argument(
        "target",
        help="Document id (e.g. `api-integrations`) or path relative to the knowledge root.",
    )

    chunks = commands.add_parser(
        "chunks",
        help="Show where a document is cut into retrieval units.",
    )
    _add_root_option(chunks)
    chunks.add_argument(
        "target",
        nargs="?",
        help="Document id or path relative to the knowledge root. Omit with --all.",
    )
    chunks.add_argument(
        "--all",
        action="store_true",
        help="Summarize the whole corpus instead of showing one document.",
    )
    chunks.add_argument(
        "--show-content",
        action="store_true",
        help="Print each chunk's text, so boundaries can be checked by eye.",
    )
    _add_policy_options(chunks)

    embedding = commands.add_parser(
        "embedding",
        help="Show the text that would be embedded for a chunk or document.",
    )
    _add_root_option(embedding)
    embedding.add_argument(
        "target",
        help="Chunk id (e.g. `profile--0000`), or a document id or path.",
    )
    embedding.add_argument(
        "--show-text",
        action="store_true",
        help="Print the full representation instead of a preview.",
    )
    _add_policy_options(embedding)

    index = commands.add_parser(
        "index",
        help="Synchronize the vector index with the knowledge base.",
    )
    _add_root_option(index)
    index.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the plan without embedding, writing or deleting anything.",
    )
    index.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-embed every chunk instead of reusing what the index already holds.",
    )
    _add_policy_options(index)

    _add_query_group(groups)
    _add_eval_group(groups)
    return parser


def _add_eval_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Measuring retrieval and grounding against a dataset with known answers."""
    evaluate = groups.add_parser("eval", help="Measure retrieval and grounding against a dataset.")
    commands = evaluate.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="Run the evaluation dataset against the configured providers.",
    )
    run.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
        help=f"Evaluation dataset file (default: {DEFAULT_EVALUATION_DATASET}).",
    )
    run.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip the answering pass, so no generation provider is called at all.",
    )
    run.add_argument(
        "--generation-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Minimum seconds between two generation requests, so a full run stays "
            "inside a provider's rate limits. Waits only where a model is actually "
            "called. Default: 0, which paces nothing."
        ),
    )
    run.add_argument(
        "--retrieval-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait between two questions, so a run does not send its "
            "query embeddings back to back. Never before the first question or "
            "after the last. Default: 0, which waits nothing."
        ),
    )
    run.add_argument(
        "--suite",
        choices=[suite.value for suite in EvaluationSuite],
        default=EvaluationSuite.FULL.value,
        help=(
            "Which questions to run: every one (full, the default) or the fixed "
            "subset named in <dataset>.smoke.yaml beside the dataset (smoke)."
        ),
    )
    run.add_argument(
        "--output",
        type=Path,
        help=(
            "Also write the retrieval results as JSON: run metadata and, per "
            "question, every retrieved passage with its similarity. Combine with "
            "--min-similarity -1 to keep every match the store returned."
        ),
    )
    run.add_argument(
        "--question-id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "Run only the question with this id, out of the suite. May be given more "
            "than once. For re-asking the questions a run failed on without paying "
            "for the rest; a result from a selection is a check, not a measurement."
        ),
    )
    run.add_argument(
        "--e2e",
        action="store_true",
        help=(
            "Ask every question once through the whole pipeline and score retrieval "
            "from the retrieval each answer used, instead of a retrieval pass followed "
            "by an answering pass. --output then receives the end-to-end results."
        ),
    )
    run.add_argument(
        "--summary",
        type=Path,
        help="With --e2e: also write the results as a Markdown summary.",
    )
    run.add_argument(
        "--tier",
        choices=[tier.value for tier in Tier],
        help=(
            "Required with --e2e against a real provider: smoke (a handful of questions, a "
            "provider health check) or acceptance (the release-acceptance suite). There is no "
            "default: "
            "a run that spends the provider allocation is always asked for by name."
        ),
    )
    run.add_argument(
        "--preflight-only",
        action="store_true",
        help="With --e2e: print the forecast and budget decision, and call nothing.",
    )
    run.add_argument(
        "--allow-yellow",
        action="store_true",
        help=(
            "With --e2e: start a run whose forecast is above the target share of the daily "
            "budget but keeps the minimum reserve. A run that would not keep it never starts."
        ),
    )
    run.add_argument(
        "--daily-budget",
        type=float,
        default=DAILY_BUDGET_NEURONS,
        help=f"Nominal daily neuron allocation (default: {DAILY_BUDGET_NEURONS:,.0f}).",
    )
    run.add_argument(
        "--minimum-reserve",
        type=float,
        default=MINIMUM_RESERVE_NEURONS,
        help=(
            "Neurons that must remain for production after the run "
            f"(default: {MINIMUM_RESERVE_NEURONS:,.0f})."
        ),
    )
    run.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help=f"Local record of provider runs and their estimated cost (default: {DEFAULT_LEDGER}).",
    )
    run.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY,
        help=(
            "Directory of earlier end-to-end exports the forecast learns from "
            f"(default: {DEFAULT_HISTORY})."
        ),
    )
    run.add_argument(
        "--note",
        action="append",
        default=[],
        help=(
            "With --e2e: a known condition a reader needs in order to interpret the "
            "run, recorded in the export and the summary. May be given more than once."
        ),
    )
    run.add_argument(
        "--rerun-of",
        metavar="RUN_ID",
        help=(
            "With --e2e --tier acceptance: this run repeats that one, under the "
            "provider-outlier rule — once per commit, only after a run that failed on "
            "provider availability alone, and with a --note naming the outlier. See "
            "docs/RELEASE_ACCEPTANCE.md."
        ),
    )
    _add_retrieval_options(run)

    validate_acceptance = commands.add_parser(
        "validate-acceptance",
        help=(
            "Check an end-to-end export as a release-acceptance artifact: recompute its "
            "gates and verdict from its own records, and look for anything that must not "
            "be published. Reads one file; calls nothing."
        ),
    )
    validate_acceptance.add_argument("artifact", type=Path, help="The end-to-end JSON export.")
    validate_acceptance.add_argument(
        "--development",
        action="store_true",
        help=(
            "Accept an artifact that is consistent but not a passing release acceptance "
            "(a dirty tree, a smoke run, a failed gate) — for reading development runs. "
            "Without it, only a PASS that may be published is accepted."
        ),
    )

    experiment = commands.add_parser(
        "experiment",
        help=(
            "Compare provider request configurations on fixed contexts (Paket 3). "
            "Calls the configured real generation provider."
        ),
    )
    experiment.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
        help=f"Evaluation dataset the questions come from (default: {DEFAULT_EVALUATION_DATASET}).",
    )
    experiment.add_argument(
        "--question-id",
        action="append",
        required=True,
        metavar="ID",
        help="A question to ask. May be given more than once; each is retrieved once.",
    )
    experiment.add_argument(
        "--variant",
        action="append",
        choices=[ResponseFormat.JSON_OBJECT.value, ResponseFormat.TEXT.value],
        default=[],
        help=(
            "A response_format to compare: json_object (as shipped) or text (none "
            "requested). Default: both, in that order."
        ),
    )
    experiment.add_argument(
        "--repetitions", type=int, default=3, help="Calls per question and variant (default: 3)."
    )
    experiment.add_argument(
        "--max-calls",
        type=int,
        required=True,
        help="Hard limit on provider calls. The run stops before exceeding it.",
    )
    experiment.add_argument(
        "--neuron-budget",
        type=float,
        help=(
            "Stop before a call that could take the estimated spend past this many "
            "neurons. An estimate from reported tokens; needs both rate options."
        ),
    )
    experiment.add_argument(
        "--input-neurons-per-million",
        type=float,
        help="Neurons per million input tokens of the model, from the provider's price list.",
    )
    experiment.add_argument(
        "--output-neurons-per-million",
        type=float,
        help="Neurons per million output tokens of the model, from the provider's price list.",
    )
    experiment.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Seconds to wait between two calls (default: 0).",
    )
    experiment.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the results as JSON. Written even when the run stops early.",
    )
    experiment.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help=(
            "Local record of provider runs; the experiment is entered as tier `experiment`, "
            f"apart from smoke and acceptance (default: {DEFAULT_LEDGER})."
        ),
    )


def _add_query_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The query side: what a question finds, and what it is answered with."""
    query = groups.add_parser("query", help="Ask the knowledge base a question.")
    commands = query.add_subparsers(dest="command", required=True)

    retrieve = commands.add_parser(
        "retrieve",
        help="Show which passages a question retrieves. No language model is called.",
    )
    _add_root_option(retrieve)
    retrieve.add_argument("question", help="The question to search with.")
    _add_retrieval_options(retrieve)
    retrieve.add_argument(
        "--show-content",
        action="store_true",
        help="Print each passage in full instead of a preview.",
    )

    answer = commands.add_parser(
        "answer",
        help="Run the whole query pipeline: retrieval, context, generation, citations.",
    )
    _add_root_option(answer)
    answer.add_argument("question", help="The question to answer.")
    _add_retrieval_options(answer)
    answer.add_argument(
        "--show-retrieval",
        action="store_true",
        help="Also print what retrieval found, with similarities.",
    )
    answer.add_argument(
        "--show-context",
        action="store_true",
        help="Also print the exact context the model was given. Developer output.",
    )


def _add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    """Per-run retrieval overrides.

    Defaults live in :class:`RetrievalPolicy`; these override them for one
    invocation. Visibility is deliberately absent — it is not a policy field,
    and public retrieval is not something a flag can turn off.
    """
    parser.add_argument("--top-k", type=int, help="Override how many candidates are considered.")
    parser.add_argument(
        "--min-similarity",
        type=float,
        help="Override the minimum similarity a match needs to count as evidence.",
    )


def _add_root_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_KNOWLEDGE_ROOT,
        help=f"Knowledge base directory (default: {DEFAULT_KNOWLEDGE_ROOT}).",
    )


def _add_policy_options(parser: argparse.ArgumentParser) -> None:
    """Per-run size budget overrides.

    Defaults live in :class:`ChunkingPolicy`; these only override them for one
    invocation. Nothing is written to disk or to the environment — the point is
    to compare boundaries quickly, not to configure the system.
    """
    parser.add_argument("--target-chars", type=int, help="Override the soft size goal.")
    parser.add_argument("--max-chars", type=int, help="Override the hard size ceiling.")
    parser.add_argument("--overlap-chars", type=int, help="Override the overlap budget.")


def _policy_from(args: argparse.Namespace) -> ChunkingPolicy:
    """Build the policy, letting the model itself do the validating."""
    overrides = {
        name: value
        for name, value in (
            ("target_chars", args.target_chars),
            ("max_chars", args.max_chars),
            ("overlap_chars", args.overlap_chars),
        )
        if value is not None
    }
    return ChunkingPolicy(**overrides)


def _dispatch(args: argparse.Namespace) -> int:
    if args.group == "query":
        return _run_query(args, sys.stdout)
    if args.group == "eval" and args.command == "experiment":
        return _run_experiment(args, sys.stdout)
    if args.group == "eval" and args.command == "validate-acceptance":
        return _run_validate_acceptance(args, sys.stdout)
    if args.group == "eval":
        return _run_eval(args, sys.stdout)
    if args.command == "validate":
        return _run_validate(args.root, sys.stdout)
    if args.command == "inspect":
        return _run_inspect(args.root, args.target, sys.stdout)
    if args.command == "chunks":
        return _run_chunks(args, sys.stdout)
    if args.command == "embedding":
        return _run_embedding(args, sys.stdout)
    if args.command == "index":
        return _run_index(args, sys.stdout)
    raise AssertionError(f"unhandled command: {args.command}")  # pragma: no cover


def _run_validate(root: Path, out: TextIO) -> int:
    report = collect_knowledge_base(root)

    print(f"Knowledge root: {root}", file=out)
    print(file=out)

    for line in _status_lines(report):
        print(line, file=out)
    if report.document_count:
        print(file=out)

    print(_count(report.document_count, "document"), file=out)
    print(f"{len(report.documents)} valid", file=out)
    print(_count(len(report.issues), "error"), file=out)

    return EXIT_OK if report.is_valid else EXIT_INVALID


def _status_lines(report: KnowledgeBaseReport) -> list[str]:
    """One line per document, failures expanded, ordered by path."""
    entries: list[tuple[str, list[str]]] = [
        (document.provenance.source_path, [f"✓ {document.provenance.source_path}"])
        for document in report.documents
    ]
    entries += [
        (issue.source_path, [f"✗ {issue.source_path}", f"    {issue.describe()}"])
        for issue in report.issues
    ]
    entries.sort(key=lambda entry: entry[0])
    return [line for _, lines in entries for line in lines]


def _run_inspect(root: Path, target: str, out: TextIO) -> int:
    report = collect_knowledge_base(root)
    document = _find(report, target)

    if document is None:
        print(f"No valid document matches `{target}` under {root}.", file=sys.stderr)
        matching_issue = next(
            (issue for issue in report.issues if issue.source_path == target), None
        )
        if matching_issue is not None:
            print(f"It failed to ingest: {matching_issue.describe()}", file=sys.stderr)
        return EXIT_INVALID

    metadata = document.metadata
    rows: list[tuple[str, str]] = [
        ("ID", metadata.id),
        ("Title", metadata.title),
        ("Schema", str(metadata.schema_version)),
        ("Type", metadata.document_type.value),
        ("Language", metadata.language),
        ("Version", str(metadata.version)),
        ("Updated", metadata.updated_at.isoformat()),
        ("Visibility", metadata.visibility.value),
        ("Trust", metadata.trust_level.value),
        ("Source", metadata.source),
        ("Path", document.provenance.source_path),
        ("Fingerprint", document.provenance.document_fingerprint),
        ("Characters", str(len(document.content))),
        ("Topics", ", ".join(metadata.topics) or "—"),
        ("Technologies", ", ".join(metadata.technologies) or "—"),
    ]
    if metadata.license is not None:
        rows.append(("License", metadata.license))

    for label, value in rows:
        print(f"{label:<{_LABEL_WIDTH}}{value}", file=out)

    return EXIT_OK


def _run_chunks(args: argparse.Namespace, out: TextIO) -> int:
    """Show chunk boundaries — for one document, or across the corpus."""
    try:
        policy = _policy_from(args)
    except ValidationError as exc:
        print(f"Invalid chunking policy: {_first_policy_problem(exc)}", file=sys.stderr)
        return EXIT_INVALID

    if not args.all and args.target is None:
        print("Give a document id or path, or use --all for a corpus summary.", file=sys.stderr)
        return EXIT_INVALID

    report = collect_knowledge_base(args.root)

    try:
        if args.all:
            return _run_corpus_chunks(report, policy, out)
        return _run_document_chunks(report, args.target, policy, args.show_content, out)
    except ChunkingError as exc:
        print(f"Chunking failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID


def _run_document_chunks(
    report: KnowledgeBaseReport,
    target: str,
    policy: ChunkingPolicy,
    show_content: bool,
    out: TextIO,
) -> int:
    document = _find(report, target)
    if document is None:
        print(f"No valid document matches `{target}` under {report.root}.", file=sys.stderr)
        matching_issue = next(
            (issue for issue in report.issues if issue.source_path == target), None
        )
        if matching_issue is not None:
            print(f"It failed to ingest: {matching_issue.describe()}", file=sys.stderr)
        return EXIT_INVALID

    chunks = chunk_document(document, policy)

    print(f"Document: {document.metadata.title}", file=out)
    print(f"ID: {document.metadata.id}", file=out)
    print(file=out)
    _print_labelled("Policy", policy.describe(), out)

    for chunk in chunks:
        print(file=out)
        print(f"Chunk {chunk.ordinal:04d}", file=out)
        _print_rows(_chunk_rows(chunk), out)
        if show_content:
            print(file=out)
            for line in chunk.content.split("\n"):
                print(f"    │ {line}" if line else "    │", file=out)

    print(file=out)
    print(_count(len(chunks), "chunk"), file=out)
    return EXIT_OK


def _run_corpus_chunks(report: KnowledgeBaseReport, policy: ChunkingPolicy, out: TextIO) -> int:
    """Summarize the whole corpus — but only if the corpus is actually whole."""
    if not report.is_valid:
        print(
            f"Knowledge base has {_count(len(report.issues), 'unusable document')}; "
            "chunking a partial corpus would be misleading. "
            "Run `knowledge validate` for details.",
            file=sys.stderr,
        )
        return EXIT_INVALID

    chunks = chunk_knowledge_base(report.documents, policy)

    print(f"Knowledge root: {report.root}", file=out)
    print(file=out)
    _print_labelled("Policy", policy.describe(), out)
    print(file=out)
    _print_rows(ChunkStatistics.of(chunks, policy).describe(), out)
    return EXIT_OK


def _run_embedding(args: argparse.Namespace, out: TextIO) -> int:
    """Show what would be sent to an embedding provider — without sending it.

    Building the representation and its fingerprint is a local computation, so
    this command needs no credentials and makes no request. It reports the
    configured embedding space because the fingerprint depends on it.
    """
    try:
        policy = _policy_from(args)
    except ValidationError as exc:
        print(f"Invalid chunking policy: {_first_policy_problem(exc)}", file=sys.stderr)
        return EXIT_INVALID

    settings = get_settings()
    try:
        spec = build_embedding_spec(settings)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    report = collect_knowledge_base(args.root)
    try:
        chunks = _resolve_chunks(report, args.target, policy)
    except ChunkingError as exc:
        print(f"Chunking failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID

    if not chunks:
        print(f"No chunk or document matches `{args.target}` under {args.root}.", file=sys.stderr)
        return EXIT_INVALID

    _print_labelled("Embedding space", spec.describe(), out)

    for chunk in chunks:
        text = build_embedding_text(chunk)
        print(file=out)
        print(f"Chunk {chunk.ordinal:04d}", file=out)
        _print_rows(_embedding_rows(chunk, text, spec), out)
        print(file=out)
        for line in (text if args.show_text else _preview(text)).split("\n"):
            print(f"    │ {line}" if line else "    │", file=out)

    print(file=out)
    print(_count(len(chunks), "chunk"), file=out)
    return EXIT_OK


def _embedding_rows(
    chunk: KnowledgeChunk, text: str, spec: EmbeddingSpec
) -> tuple[tuple[str, str], ...]:
    return (
        ("Chunk ID", chunk.id),
        ("Document", chunk.document_id),
        ("Heading", " > ".join(chunk.heading_path) or "—"),
        ("Representation", EMBEDDING_REPRESENTATION_VERSION),
        ("Dimensions", str(spec.dimensions)),
        ("Fingerprint", compute_embedding_fingerprint(text, spec)),
        ("Characters", str(len(text))),
    )


def _preview(text: str, limit: int = 240) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"


def _resolve_chunks(
    report: KnowledgeBaseReport, target: str, policy: ChunkingPolicy
) -> list[KnowledgeChunk]:
    """Resolve a chunk id, or a document id/path, to the chunks it names."""
    document = _find(report, target)
    if document is not None:
        return list(chunk_document(document, policy))

    for candidate in report.documents:
        for chunk in chunk_document(candidate, policy):
            if chunk.id == target:
                return [chunk]
    return []


def _run_index(args: argparse.Namespace, out: TextIO) -> int:
    """Plan — and unless this is a dry run, apply — index changes."""
    try:
        policy = _policy_from(args)
    except ValidationError as exc:
        print(f"Invalid chunking policy: {_first_policy_problem(exc)}", file=sys.stderr)
        return EXIT_INVALID

    report = collect_knowledge_base(args.root)
    if not report.is_valid:
        print(
            f"Knowledge base has {_count(len(report.issues), 'unusable document')}; "
            "indexing a partial corpus would publish an incomplete index. "
            "Run `knowledge validate` for details.",
            file=sys.stderr,
        )
        return EXIT_INVALID

    try:
        chunks = chunk_knowledge_base(report.documents, policy)
    except ChunkingError as exc:
        print(f"Chunking failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID

    try:
        return asyncio.run(_index(args, report, chunks, out))
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except IndexingError as exc:
        print(f"Indexing failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID
    except PortError as exc:
        print(f"Indexing failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID


async def _index(
    args: argparse.Namespace,
    report: KnowledgeBaseReport,
    chunks: Sequence[KnowledgeChunk],
    out: TextIO,
) -> int:
    settings: Settings = get_settings()
    components = build_indexing_components(settings)
    try:
        print(f"Knowledge root: {report.root}", file=out)
        print(file=out)
        _print_labelled("Embedding space", components.provider.spec.describe(), out)
        print(file=out)
        _print_rows(
            (
                ("Documents", str(len(report.documents))),
                ("Chunks", str(len(chunks))),
            ),
            out,
        )

        if args.dry_run:
            plan = await components.service.plan(chunks, rebuild=args.rebuild)
            print(file=out)
            _print_labelled(
                "Index plan",
                (*plan.describe(), ("embeddings required", str(len(plan.embeddings_required)))),
                out,
            )
            _warn_if_stale_undetectable(plan, out)
            print(file=out)
            print("Dry run: nothing was embedded, written or deleted.", file=out)
            return EXIT_OK

        result = await components.service.synchronize(chunks, rebuild=args.rebuild)
        print(file=out)
        _print_labelled("Result", result.describe(), out)
        if not result.stale_detection:
            print(file=out)
            print(
                "  Note: this store cannot enumerate its contents, so stale records "
                "could not be detected.",
                file=out,
            )
        print(file=out)
        print(
            "Index synchronized." if not result.is_noop else "Index already up to date.", file=out
        )
        return EXIT_OK
    finally:
        await components.aclose()


def _warn_if_stale_undetectable(plan: IndexPlan, out: TextIO) -> None:
    if not plan.stale_detection:
        print(
            "  Note: this store cannot enumerate its contents, so stale records "
            "could not be detected.",
            file=out,
        )


# --- evaluation ---------------------------------------------------------------


def _run_eval(args: argparse.Namespace, out: TextIO) -> int:
    """Measure retrieval, and unless asked not to, grounding as well."""
    try:
        policy = _retrieval_policy_from(args)
    except ValidationError as exc:
        print(f"Invalid retrieval policy: {_first_policy_problem(exc)}", file=sys.stderr)
        return EXIT_INVALID

    try:
        pacing = _pacing_from(args)
    except ValueError as exc:
        print(f"Invalid generation pacing: {exc}", file=sys.stderr)
        return EXIT_INVALID

    try:
        validate_question_delay(float(args.retrieval_delay_seconds))
    except ValueError as exc:
        print(f"Invalid retrieval pacing: --retrieval-delay-seconds: {exc}", file=sys.stderr)
        return EXIT_INVALID

    problem = _e2e_argument_problem(args) or _e2e_provider_problem(args, get_settings())
    if problem is not None:
        print(f"Invalid end-to-end run: {problem}", file=sys.stderr)
        return EXIT_INVALID

    suite = EvaluationSuite(args.suite)
    if args.tier is not None and not args.question_id:
        suite = TIER_SUITES[Tier(args.tier)]
    # Taken once, before anything runs: the commit a run is attributed to is the
    # one it started from, whatever happens to the tree while it runs.
    git = _git_state()
    try:
        full = load_dataset(args.dataset)
        dataset = select_suite(args.dataset, full, suite)
    except EvaluationDatasetError as exc:
        print(f"Evaluation dataset error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    if args.question_id:
        unknown = sorted(set(args.question_id) - {question.id for question in dataset.questions})
        if unknown:
            print(
                f"Evaluation dataset error: no such question in this suite: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return EXIT_INVALID
        # The dataset's own order is kept, whatever order the ids were given in.
        dataset = dataset.model_copy(
            update={
                "questions": tuple(
                    question
                    for question in dataset.questions
                    if question.id in set(args.question_id)
                )
            }
        )

    plan: Preflight | None = None
    if args.e2e and args.tier is not None:
        plan = _plan_provider_run(args, len(dataset.questions), get_settings(), policy, git)
        _print_labelled(
            f"{args.tier.upper()} PREFLIGHT",
            describe_preflight(plan, yellow_confirmed=args.allow_yellow),
            out,
        )
        for warning in plan.warnings:
            print(f"  ! {warning}", file=out)
        if not plan.can_start(yellow_confirmed=args.allow_yellow):
            for blocker in plan.blockers:
                print(f"Not started: {blocker}", file=sys.stderr)
            if plan.zone is BudgetZone.RED:
                print(
                    "Not started: budget zone RED — the forecast would not leave the minimum "
                    "reserve. Nothing was requested.",
                    file=sys.stderr,
                )
            elif plan.zone is BudgetZone.YELLOW and not plan.blockers:
                print(
                    "Not started: budget zone YELLOW — above the target share of the daily "
                    "budget. Start it deliberately with --allow-yellow.",
                    file=sys.stderr,
                )
            return EXIT_INVALID
        if args.preflight_only:
            print(file=out)
            print("Preflight only: no provider was called.", file=out)
            return EXIT_OK

    corpus = resolve_corpus(args.dataset, dataset)
    try:
        return asyncio.run(
            _evaluate(
                args,
                dataset,
                corpus,
                policy,
                pacing,
                out,
                full_count=len(full.questions),
                plan=plan,
                suite=suite,
                git=git,
            )
        )
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except IndexingError as exc:
        print(f"Indexing failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID
    except AppError as exc:
        print(f"Evaluation failed: {exc.code.value}: {exc.message}", file=sys.stderr)
        return EXIT_INVALID
    except PortError as exc:
        print(f"Evaluation failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID


def _e2e_argument_problem(args: argparse.Namespace) -> str | None:
    """What is wrong with the end-to-end flags, if anything.

    Checked before a provider is built, so a run that could not have written
    its result never spends a request finding that out.
    """
    if args.rerun_of is not None:
        if not args.e2e or args.tier != Tier.ACCEPTANCE.value:
            return "--rerun-of belongs to an --e2e --tier acceptance run"
        if not args.note:
            return "--rerun-of needs a --note naming the provider outlier it repeats"
    if not args.e2e:
        if args.summary is not None or args.note:
            return "--summary and --note belong to an --e2e run"
        if args.tier is not None or args.preflight_only or args.allow_yellow:
            return "--tier, --preflight-only and --allow-yellow belong to an --e2e run"
        return None
    if args.retrieval_only:
        return "--e2e answers every question, which --retrieval-only rules out"
    if args.output is None:
        return "--e2e needs --output, or the run would be paid for and not kept"
    if args.tier is not None:
        if args.suite != EvaluationSuite.FULL.value:
            return "with --tier the tier chooses the questions; leave out --suite"
        if args.tier == Tier.ACCEPTANCE.value and args.question_id:
            return "an acceptance run asks its whole suite; --question-id is for a smoke run"
    return None


def _e2e_provider_problem(args: argparse.Namespace, settings: Settings) -> str | None:
    """Refuse an end-to-end run that would measure a development stand-in.

    The deterministic adapters exist so the pipeline runs with no key and no
    network, and every other evaluation mode is welcome to use them. An
    end-to-end run is the one result that claims "this is what the configured
    system does": with the echoing stub behind it, every question comes back
    answered and every label cited, which reads as a finding about grounding
    and is one about configuration. Checked before anything is built, so it
    costs no request to find out.
    """
    real_generation = settings.llm_provider is not LLMProviderName.DETERMINISTIC
    if not args.e2e:
        if real_generation and not args.retrieval_only:
            return (
                "an answering pass against a real generation provider runs only as "
                "--e2e --tier smoke|acceptance, with a budget preflight; use "
                "--retrieval-only to measure retrieval alone"
            )
        return None
    stand_ins = _stand_in_roles(settings)
    if not stand_ins:
        if args.tier is None:
            return (
                "an end-to-end run against a real generation provider needs --tier smoke or "
                "--tier acceptance — there is no default for a run that spends the allocation"
            )
        return None
    return (
        f"--e2e measures real providers, and the configured {' and '.join(stand_ins)} "
        f"provider is a development stand-in (embedding provider: "
        f"`{settings.embedding_provider.value}`, generation provider: "
        f"`{settings.llm_provider.value}`). Set PORTFOLIO_RAG_EMBEDDING_PROVIDER and "
        "PORTFOLIO_RAG_LLM_PROVIDER to real providers; nothing was requested."
    )


def _run_experiment(args: argparse.Namespace, out: TextIO) -> int:
    """Compare provider request configurations on fixed contexts. Paket 3, E1."""
    settings = get_settings()
    stand_ins = _stand_in_roles(settings)
    if stand_ins:
        print(
            f"Invalid experiment: the configured {' and '.join(stand_ins)} provider is a "
            "development stand-in, and an experiment on it measures nothing. Nothing was "
            "requested.",
            file=sys.stderr,
        )
        return EXIT_INVALID
    try:
        limits = ExperimentLimits(
            max_calls=args.max_calls,
            neuron_budget=args.neuron_budget,
            input_neurons_per_million=args.input_neurons_per_million,
            output_neurons_per_million=args.output_neurons_per_million,
        )
        variants = tuple(ResponseFormat(value) for value in args.variant) or DEFAULT_VARIANTS
        if args.repetitions <= 0 or args.delay_seconds < 0:
            raise ValueError("--repetitions must be positive and --delay-seconds not negative")
        if len(set(variants)) != len(variants):
            raise ValueError("each --variant may be given once")
    except ValueError as exc:
        print(f"Invalid experiment: {exc}", file=sys.stderr)
        return EXIT_INVALID

    try:
        dataset = load_dataset(args.dataset)
    except EvaluationDatasetError as exc:
        print(f"Evaluation dataset error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    by_id = {question.id: question for question in dataset.questions}
    unknown = sorted(set(args.question_id) - set(by_id))
    if unknown:
        print(f"Evaluation dataset error: no such question: {', '.join(unknown)}", file=sys.stderr)
        return EXIT_INVALID
    questions = [by_id[question_id] for question_id in dict.fromkeys(args.question_id)]

    try:
        return asyncio.run(
            _experiment(
                args,
                questions,
                resolve_corpus(args.dataset, dataset),
                limits,
                variants,
                out,
            )
        )
    except ExperimentError as exc:
        print(f"Invalid experiment: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except AppError as exc:
        print(f"Experiment failed: {exc.code.value}: {exc.message}", file=sys.stderr)
        return EXIT_INVALID
    except PortError as exc:
        print(f"Experiment failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID


async def _experiment(
    args: argparse.Namespace,
    questions: Sequence[EvaluationQuestion],
    corpus: Path,
    limits: ExperimentLimits,
    variants: tuple[ResponseFormat, ...],
    out: TextIO,
) -> int:
    """Prepare once, run the plan, and write what was measured however the run ends."""
    settings = get_settings().model_copy(update={"knowledge_root": corpus})
    # One transport attempt: a call is one request, and its cost and outcome
    # are that request's alone. A rate limit is a reason to stop, not to retry.
    components = build_query_components(
        settings, generation_attempts=EVALUATION_GENERATION_ATTEMPTS
    )
    experiment_id = new_request_id()
    binding = set_request_id(experiment_id)
    run: ExperimentRun | None = None
    prepared: tuple[PreparedQuestion, ...] = ()
    try:
        await components.prepare()
        prepared = await prepare_questions(
            questions, components.retrieval, components.answers.context_policy
        )
        run = ExperimentRun(
            plan=plan_calls(prepared, variants, args.repetitions),
            llm=components.llm,
            limits=limits,
            delay_seconds=args.delay_seconds,
        )
        print(f"Experiment: {experiment_id}", file=out)
        print(
            f"Planned:    {len(run.plan)} calls "
            f"({len(prepared)} questions x {len(variants)} variants x {args.repetitions}), "
            f"limit {limits.max_calls}",
            file=out,
        )
        reason = await run.run()
        print(f"Stopped:    {reason.value} after {len(run.calls)} calls", file=out)
        return EXIT_OK if reason in _EXPERIMENT_CLEAN_STOPS else EXIT_INVALID
    finally:
        if run is not None:
            revision, dirty, _, _ = _git_state()
            metadata = ExperimentMetadata(
                experiment_id=experiment_id,
                generated_at=datetime.now(UTC),
                git_revision=revision,
                git_dirty=dirty,
                dataset_path=args.dataset.as_posix(),
                dataset_sha256=dataset_fingerprint(args.dataset),
                provider=settings.llm_provider.value,
                model=components.llm.model,
                repetitions=args.repetitions,
                variants=variants,
                delay_seconds=args.delay_seconds,
            )
            try:
                write_export(args.output, export_experiment(run, prepared, metadata))
                print(f"Results:    {args.output}", file=out)
            finally:
                Ledger(args.ledger).append(
                    _experiment_ledger_entry(experiment_id, run, settings, components, args.output)
                )
        await components.aclose()
        reset_request_id(binding)


def _experiment_ledger_entry(
    experiment_id: str,
    run: ExperimentRun,
    settings: Settings,
    components: QueryComponents,
    artifact: Path,
) -> LedgerEntry:
    """An experiment's line in the ledger: its own tier, its own spend."""
    records = [call.record for call in run.calls]
    reported = [record for record in records if record.usage_reported]
    profile = cost_profile(settings.llm_provider.value, components.llm.model)
    neurons = (
        spent(records, profile, components.answers.context_policy)[0]
        if profile is not None
        else 0.0
    )
    return LedgerEntry(
        date_utc=datetime.now(UTC).date().isoformat(),
        run_id=experiment_id,
        tier="experiment",
        status="complete" if run.stop_reason is StopReason.COMPLETED else "aborted",
        calls=len(records),
        input_tokens=sum(record.input_tokens or 0 for record in reported),
        output_tokens=sum(record.output_tokens or 0 for record in reported),
        estimated_neurons=round(neurons, 1),
        usage_coverage=round(len(reported) / len(records), 3) if records else None,
        artifact=artifact.as_posix(),
        abort_reason=(
            run.stop_reason.value if run.stop_reason not in (None, StopReason.COMPLETED) else None
        ),
    )


#: Ends a run is planned to have. Every other one means the provider stopped it.
_EXPERIMENT_CLEAN_STOPS: Final = frozenset(
    {StopReason.COMPLETED, StopReason.MAX_CALLS, StopReason.NEURON_BUDGET}
)


#: Where provider runs are recorded, and where earlier exports are read from.
DEFAULT_LEDGER: Final = Path("evaluation/results/provider-ledger.jsonl")
DEFAULT_HISTORY: Final = Path("evaluation/results")


def _generation_model(settings: Settings) -> str:
    """The model the configured generation provider will use, from settings alone."""
    if settings.llm_provider is LLMProviderName.CLOUDFLARE_WORKERS_AI:
        return settings.cloudflare_workers_ai_chat_model
    if settings.llm_provider is LLMProviderName.MISTRAL:
        return settings.mistral_chat_model
    return settings.llm_provider.value


def _plan_provider_run(
    args: argparse.Namespace,
    questions: int,
    settings: Settings,
    retrieval_policy: RetrievalPolicy,
    git: GitState,
) -> Preflight:
    """The preflight of a tiered run: computed from files, before anything is built."""
    provider = settings.llm_provider.value
    model = _generation_model(settings)
    requirements = ProfileRequirements(
        provider=provider,
        model=model,
        prompt_version=GROUNDED_PROMPT_VERSION,
        grounding_check_version=GROUNDING_CHECK_VERSION,
        response_format=GROUNDED_RESPONSE_FORMAT.value,
        max_prompt_tokens=DEFAULT_CONTEXT_POLICY.max_prompt_tokens,
        output_reserve_tokens=DEFAULT_CONTEXT_POLICY.output_reserve_tokens,
        top_k=retrieval_policy.top_k,
        min_similarity=retrieval_policy.min_similarity,
    )
    plan = preflight(
        tier=Tier(args.tier),
        questions=questions,
        provider=provider,
        model=model,
        profile=cost_profile(provider, model),
        history=load_history(args.history, requirements),
        ledger=Ledger(args.ledger).read(),
        today=datetime.now(UTC).date(),
        policy=BudgetPolicy(daily_budget=args.daily_budget, minimum_reserve=args.minimum_reserve),
        commit=git.revision,
        dirty=git.dirty,
        rerun_of=args.rerun_of,
        source_identity=git.source_identity,
    )
    target = args.output.parent if args.output.parent != Path() else Path.cwd()
    if not target.is_dir():
        return replace(plan, blockers=(*plan.blockers, f"export directory {target} does not exist"))
    return plan


def _operations(
    args: argparse.Namespace,
    plan: Preflight,
    guard: RunGuard,
    report: E2EReport,
    verdict: SmokeVerdict | None,
) -> dict[str, Any]:
    calls = [call for record in report.records for call in record.provider_calls]
    reported = [call for call in calls if call.usage_reported]
    known_after = plan.known_consumption + guard.estimated_neurons
    return {
        **plan.fields(),
        "status": "complete" if report.aborted is None else "aborted",
        "abort_reason": report.aborted.reason.value if report.aborted else None,
        "observed": {
            "calls": len(calls),
            "input_tokens": sum(call.input_tokens or 0 for call in reported) if reported else None,
            "output_tokens": (
                sum(call.output_tokens or 0 for call in reported) if reported else None
            ),
            "usage_coverage": round(len(reported) / len(calls), 3) if calls else None,
            "estimated_neurons": round(guard.estimated_neurons, 1),
            "unreported_calls": guard.unreported_calls,
        },
        "known_daily_after": round(known_after, 1),
        "nominal_reserve_after": round(plan.policy.daily_budget - known_after, 1),
        "guard": {
            "status": guard.status,
            "projected_final_neurons": (
                round(guard.projected_final_neurons, 1)
                if guard.projected_final_neurons is not None
                else None
            ),
            "projected_final_share": (
                round(guard.projected_final_neurons / plan.policy.daily_budget, 3)
                if guard.projected_final_neurons is not None
                else None
            ),
            "projected_band": guard.projected_band.value if guard.projected_band else None,
        },
        "smoke": verdict.fields() if verdict is not None else None,
    }


def _ledger_entry(
    run_id: str,
    plan: Preflight,
    guard: RunGuard,
    report: E2EReport,
    verdict: SmokeVerdict | None,
    artifact: Path,
    git: GitState,
    rerun_of: str | None,
) -> LedgerEntry:
    calls = [call for record in report.records for call in record.provider_calls]
    reported = [call for call in calls if call.usage_reported]
    return LedgerEntry(
        date_utc=datetime.now(UTC).date().isoformat(),
        run_id=run_id,
        tier=plan.tier.value,
        status="complete" if report.aborted is None else "aborted",
        calls=len(calls),
        input_tokens=sum(call.input_tokens or 0 for call in reported),
        output_tokens=sum(call.output_tokens or 0 for call in reported),
        estimated_neurons=round(guard.estimated_neurons, 1),
        usage_coverage=round(len(reported) / len(calls), 3) if calls else None,
        artifact=artifact.as_posix(),
        abort_reason=report.aborted.reason.value if report.aborted else None,
        smoke_passed=verdict.passed if verdict is not None else None,
        commit_sha=git.revision,
        git_dirty=git.dirty,
        rerun_of=rerun_of,
        failed_gates=[name for name, ok in report.gates.items() if not ok],
        release_source_identity=git.source_identity,
    )


def _stand_in_roles(settings: Settings) -> list[str]:
    """Which configured providers are development stand-ins."""
    return [
        role
        for role, is_stand_in in (
            ("embedding", settings.embedding_provider is EmbeddingProviderName.DETERMINISTIC),
            ("generation", settings.llm_provider is LLMProviderName.DETERMINISTIC),
        )
        if is_stand_in
    ]


def _pacing_from(args: argparse.Namespace) -> GenerationPacing | None:
    """Build the pacing the flag asked for, or nothing at all.

    ``None`` is not "pace with a delay of zero": it means no decorator is put
    in front of the provider, so an unpaced run is byte for byte the run that
    existed before the flag did.
    """
    delay = float(args.generation_delay_seconds)
    if delay < 0:
        raise ValueError("--generation-delay-seconds cannot be negative")
    if delay == 0:
        return None
    return GenerationPacing(min_interval_seconds=delay)


def _paced_answers(components: QueryComponents, pacing: GenerationPacing) -> GroundedAnswerService:
    """Re-wire the answering service onto a paced view of the same provider.

    Deliberately here and not in the composition root: pacing is a property of
    one evaluation run, not of a configured deployment, and the server must
    never be able to acquire it by setting an environment variable. The inner
    provider is the one ``components.aclose()`` closes — this wraps it, it does
    not replace it.

    That provider was built with a single transport attempt (see
    :data:`EVALUATION_GENERATION_ATTEMPTS`), so the budget the pacer enforces
    is the whole budget: its attempts are requests, one for one.

    No request deadline: pacing waits happen beneath the port, so a paced
    question legitimately takes longer than any production request may.
    """
    return GroundedAnswerService(
        retrieval=components.retrieval,
        llm=PacedLLMProvider(components.llm, pacing),
        context_policy=components.answers.context_policy,
    )


async def _evaluate(
    args: argparse.Namespace,
    dataset: EvaluationDataset,
    corpus: Path,
    policy: RetrievalPolicy,
    pacing: GenerationPacing | None,
    out: TextIO,
    *,
    full_count: int,
    plan: Preflight | None = None,
    suite: EvaluationSuite = EvaluationSuite.FULL,
    git: GitState | None = None,
) -> int:
    git = git if git is not None else _git_state()
    settings: Settings = get_settings().model_copy(
        update={
            "knowledge_root": corpus,
            "retrieval_top_k": policy.top_k,
            "retrieval_min_similarity": policy.min_similarity,
        }
    )
    # A paced run owns its own retrying, so the adapter under it does none.
    # Left on its default the two budgets would multiply rather than add.
    components = build_query_components(
        settings,
        generation_attempts=None if pacing is None else EVALUATION_GENERATION_ATTEMPTS,
    )
    # One id for the whole run, bound as the correlation id every log line
    # already carries — so the log lines of a run and its export name the same
    # run, without a second kind of id.
    run_id = new_request_id()
    run_binding = set_request_id(run_id)
    try:
        await components.prepare()

        print(f"Run:     {run_id}", file=out)
        print(f"Dataset: {args.dataset}  (version {dataset.version})", file=out)
        if suite is not EvaluationSuite.FULL:
            # Only a subset says so: a full run's header is the one it always had.
            print(
                f"Suite:   {suite.value}  ({len(dataset.questions)} of {full_count} questions)",
                file=out,
            )
        if args.question_id:
            print(
                f"Selection: {len(dataset.questions)} of {full_count} questions, by id",
                file=out,
            )
        print(f"Corpus:  {corpus}", file=out)
        print(file=out)
        _print_labelled("Embedding space", components.embeddings.spec.describe(), out)
        print(file=out)
        _print_labelled("Retrieval policy", components.retrieval.policy.describe(), out)

        delay = float(args.retrieval_delay_seconds)
        if delay > 0:
            # Said before the first request, so a live run shows what it is
            # about to do. Absent at the default, whose header is unchanged.
            questions = len(dataset.questions)
            print(file=out)
            _print_labelled(
                "Retrieval pacing",
                (
                    ("questions", str(questions)),
                    ("delay", f"{delay}s between questions"),
                    ("pauses", f"{max(questions - 1, 0)} per pass"),
                    ("generation", "off (--retrieval-only)" if args.retrieval_only else "on"),
                ),
                out,
            )

        if args.e2e:
            return await _evaluate_e2e(
                args,
                dataset,
                settings,
                components,
                pacing,
                out,
                full_count=full_count,
                run_id=run_id,
                plan=plan,
                suite=suite,
                git=git,
            )

        report = await run_retrieval_evaluation(dataset, components.retrieval, delay_seconds=delay)
        _print_retrieval_report(dataset, report, out)

        if args.output is not None:
            # Written before any generation, so a generation failure later in
            # the run cannot cost the retrieval data already paid for.
            metadata = _run_metadata(
                args, dataset, settings, components, full_count, run_id, suite, git
            )
            try:
                write_export(args.output, export_retrieval(report, metadata))
            except OSError as exc:
                print(f"Retrieval export could not be written: {exc}", file=sys.stderr)
                return EXIT_INVALID
            print(file=out)
            print(f"Retrieval export: {args.output}", file=out)

        if args.retrieval_only:
            print(file=out)
            print("Retrieval only: no generation provider was called.", file=out)
            return EXIT_OK if not report.internal_leaks else EXIT_INVALID

        answers = components.answers if pacing is None else _paced_answers(components, pacing)
        if pacing is not None:
            print(file=out)
            _print_labelled("Generation pacing", pacing.describe(), out)

        grounding = await run_grounding_evaluation(dataset, answers, delay_seconds=delay)
        _print_grounding_report(grounding, out)

        # A leak is the one result that makes the run itself a failure. Weak
        # retrieval is a number to act on; published internal knowledge is a bug.
        return EXIT_INVALID if report.internal_leaks else EXIT_OK
    finally:
        await components.aclose()
        reset_request_id(run_binding)


async def _evaluate_e2e(
    args: argparse.Namespace,
    dataset: EvaluationDataset,
    settings: Settings,
    components: QueryComponents,
    pacing: GenerationPacing | None,
    out: TextIO,
    *,
    full_count: int,
    run_id: str,
    plan: Preflight | None = None,
    suite: EvaluationSuite = EvaluationSuite.FULL,
    git: GitState | None = None,
) -> int:
    """One pass per question, then the same result three ways: stdout, JSON, Markdown."""
    git = git if git is not None else _git_state()
    answers = components.answers if pacing is None else _paced_answers(components, pacing)
    if pacing is not None:
        print(file=out)
        _print_labelled("Generation pacing", pacing.describe(), out)

    guard = (
        RunGuard(
            preflight=plan,
            total_questions=len(dataset.questions),
            context_policy=components.answers.context_policy,
        )
        if plan is not None
        else None
    )
    report = await run_e2e_evaluation(
        dataset, answers, delay_seconds=float(args.retrieval_delay_seconds), guard=guard
    )
    verdict = smoke_verdict(report) if plan is not None and plan.tier is Tier.SMOKE else None
    operations = (
        _operations(args, plan, guard, report, verdict)
        if plan is not None and guard is not None
        else None
    )
    _print_retrieval_report(dataset, report.retrieval, out)
    _print_e2e_report(report, out)

    metadata = E2ERunMetadata(
        retrieval=_run_metadata(
            args, dataset, settings, components, full_count, run_id, suite, git
        ),
        generation_provider=settings.llm_provider.value,
        generation_model=components.llm.model,
        prompt_version=GROUNDED_PROMPT_VERSION,
        context_policy=components.answers.context_policy,
        generation_delay_seconds=float(args.generation_delay_seconds),
        corpus=corpus_identity(components.chunks),
        notes=tuple(args.note),
        selected_question_ids=tuple(question.id for question in dataset.questions)
        if args.question_id
        else (),
        tier=args.tier,
        transport_attempts=None if pacing is None else EVALUATION_GENERATION_ATTEMPTS,
        rerun_of=args.rerun_of,
    )
    payload = with_release_acceptance(export_e2e(report, metadata, operations))
    try:
        write_export(args.output, payload)
        if args.summary is not None:
            args.summary.write_text(render_summary(payload), encoding="utf-8")
    except OSError as exc:
        print(f"End-to-end export could not be written: {exc}", file=sys.stderr)
        return EXIT_INVALID
    finally:
        # Whatever the export did, the calls were made and paid for.
        if plan is not None and guard is not None:
            Ledger(args.ledger).append(
                _ledger_entry(run_id, plan, guard, report, verdict, args.output, git, args.rerun_of)
            )
    print(file=out)
    print(f"End-to-end export: {args.output}", file=out)
    if args.summary is not None:
        print(f"Summary:           {args.summary}", file=out)
    if operations is not None:
        print(
            f"Estimated spend:   {operations['observed']['estimated_neurons']:,.0f} neurons "
            f"(estimate); known local today {operations['known_daily_after']:,.0f}",
            file=out,
        )
    release = payload["release_acceptance"]
    print(f"Release acceptance: {release['verdict']}", file=out)
    for reason in release["reasons"]:
        print(f"  - {reason}", file=out)
    if verdict is not None:
        state = "PASS" if verdict.passed else "FAIL"
        print(f"Smoke:             {state}", file=out)
        for reason in verdict.reasons:
            print(f"  - {reason}", file=out)

    if report.aborted is not None:
        if report.aborted.reason is AbortReason.INTERNAL_DEFECT:
            # A defect in this code, not a provider failure: the partial results
            # above are written, and the run is reported as the crash it was.
            print(
                f"Run aborted at {report.aborted.question_id} by an internal defect "
                f"({report.aborted.error_type}); the export holds the questions before it.",
                file=sys.stderr,
            )
            return EXIT_INTERNAL_ERROR
        print(
            f"Run stopped after {report.aborted.question_id}: {report.aborted.reason.value}. "
            "The export holds what was measured. It is not restarted automatically.",
            file=sys.stderr,
        )
        if args.tier != Tier.ACCEPTANCE.value:
            return EXIT_INVALID
    if args.tier == Tier.ACCEPTANCE.value:
        # An acceptance run is judged by its release verdict, not by its gates
        # alone: a dirty tree with green gates is still not a release acceptance.
        return EXIT_OK if release["verdict"] == "PASS" else EXIT_ACCEPTANCE_FAILED
    if verdict is not None and not verdict.passed:
        return EXIT_INVALID
    # The gates are what makes a run a failure; the measurements are numbers.
    return EXIT_OK if report.passed else EXIT_INVALID


def _print_e2e_report(report: E2EReport, out: TextIO) -> None:
    answerable = report.answerable
    answered = sum(1 for record in answerable if record.outcome is AnswerOutcome.ANSWERED)
    must_refuse = report.must_refuse
    refused = sum(1 for record in must_refuse if record.is_controlled_refusal)
    published = sum(len(record.citations) for record in report.records)
    verified = sum(record.verified_citations for record in report.records)

    print(file=out)
    _print_labelled(
        "End to end",
        (
            ("answered", str(report.count(AnswerOutcome.ANSWERED))),
            ("no knowledge", str(report.count(AnswerOutcome.NO_KNOWLEDGE))),
            ("not grounded", str(report.count(AnswerOutcome.NOT_GROUNDED))),
            ("errors", str(report.count(None))),
            ("answerable answered", f"{answered}/{len(answerable)}"),
            ("refusals correct", f"{refused}/{len(must_refuse)}"),
            ("citations verified", f"{verified}/{published}"),
        ),
        out,
    )
    print(file=out)
    _print_labelled(
        "Gates",
        tuple((name, "pass" if ok else "FAIL") for name, ok in report.gates.items()),
        out,
    )

    failing = [record for record in report.records if record.failures]
    if not failing:
        print(file=out)
        print("No end-to-end failures.", file=out)
        return

    print(file=out)
    print(f"End-to-end failures ({len(failing)}):", file=out)
    for record in failing:
        reasons = ", ".join(failure.value for failure in record.failures)
        print(f"  ✗ {record.question.id}  {reasons}", file=out)
        if record.error is not None:
            # Technical facts only — see `GenerationFailure`.
            known = ", ".join(
                f"{name}={value}"
                for name, value in record.error.fields().items()
                if value is not None
            )
            print(f"      {record.error_code}: {known}", file=out)


def _run_metadata(
    args: argparse.Namespace,
    dataset: EvaluationDataset,
    settings: Settings,
    components: QueryComponents,
    full_count: int,
    run_id: str,
    suite: EvaluationSuite,
    git: GitState,
) -> RunMetadata:
    """What the export says about its run — named fields only, never settings wholesale."""
    return RunMetadata(
        run_id=run_id,
        generated_at=datetime.now(UTC),
        git_revision=git.revision,
        git_dirty=git.dirty,
        git_tag=git.tag,
        source_identity=git.source_identity,
        dataset_path=args.dataset.as_posix(),
        dataset_sha256=dataset_fingerprint(args.dataset),
        dataset=dataset,
        dataset_question_count=full_count,
        suite=suite,
        suite_identity=suite_identity(args.dataset, dataset, suite),
        embedding=components.embeddings.spec,
        vector_store=settings.vector_store.value,
        policy=components.retrieval.policy,
        retrieval_delay_seconds=float(args.retrieval_delay_seconds),
    )


def _run_validate_acceptance(args: argparse.Namespace, out: TextIO) -> int:
    """Judge one export as a release-acceptance artifact. Local; calls nothing."""
    try:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"Artifact could not be read as JSON: {exc}", file=sys.stderr)
        return EXIT_INVALID

    check = validate_artifact(payload, secrets=_configured_secrets())
    publication = not args.development
    _print_labelled(
        "Release acceptance",
        (
            ("artifact", args.artifact.as_posix()),
            ("mode", "publication" if publication else "development"),
            ("verdict", check.verdict or "unreadable"),
            ("consistent", "yes" if check.consistent else "NO"),
        ),
        out,
    )
    for reason in check.reasons:
        print(f"  - {reason}", file=out)
    for problem in check.problems:
        print(f"  ✗ {problem}", file=out)
    print(file=out)
    if check.accepted(publication=publication):
        print(
            "Accepted: a passing release acceptance that may be published."
            if publication
            else "Accepted as a development artifact: consistent, not a release acceptance."
            if check.verdict != "PASS"
            else "Accepted as a development artifact.",
            file=out,
        )
        return EXIT_OK
    print(
        "Rejected: the artifact is inconsistent or carries something that must not be published."
        if not check.consistent
        else "Rejected for publication: not a passing release acceptance.",
        file=out,
    )
    return EXIT_INVALID


def _configured_secrets() -> list[str]:
    """Credential values from the environment, to make sure none was written
    down. Compared, never printed."""
    settings = get_settings()
    values: list[str] = [
        secret.get_secret_value()
        for secret in (
            settings.mistral_api_key,
            settings.cloudflare_api_token,
            settings.cloudflare_workers_ai_token,
            settings.edge_shared_secret,
        )
        if secret is not None
    ]
    values += [
        value
        for value in (settings.cloudflare_account_id, settings.cloudflare_vectorize_index)
        if value
    ]
    return values


class GitState(NamedTuple):
    """The commit a run starts from, whether the tree differs from it, the
    release tag on it, and the release-relevant content it holds. ``None``
    wherever git could not say."""

    revision: str | None
    dirty: bool | None
    tag: str | None = None
    source_identity: str | None = None


def _git_state() -> GitState:
    """The commit this run was made from, and whether the tree had changes.

    Unknown when git is missing or this is not a checkout: an unknown revision
    is recorded as unknown rather than guessed. Untracked files count as
    changes; the run-output directories (:data:`GIT_DIRTY_EXCLUDES`) do not.
    """
    git = shutil.which("git")
    if git is None:
        return GitState(None, None)
    # Pathspecs from the repository root, wherever the command runs from.
    excludes = [f":(top,exclude){path}" for path in GIT_DIRTY_EXCLUDES]
    try:
        revision = subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [git, "status", "--porcelain", "--", ":(top)", *excludes],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout
        # No tag on HEAD is a normal state, not an error: check=False.
        tag = subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [git, "describe", "--tags", "--exact-match", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        ).stdout.strip()
        identity = _source_identity(git)
    except (OSError, subprocess.SubprocessError):
        return GitState(None, None)
    return GitState(revision or None, bool(status.strip()), tag or None, identity)


def _source_identity(git: str) -> str:
    """The release candidate as content: every file under
    :data:`SOURCE_IDENTITY_INCLUDES` that git tracks or would track, read from
    the working tree, minus :data:`SOURCE_IDENTITY_EXCLUDES`.

    Working-tree content rather than ``HEAD``: on a clean tree the two are the
    same, and on a dirty one the identity describes what actually runs.
    """
    root = Path(
        subprocess.run(  # noqa: S603 - fixed arguments, no user input
            [git, "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    )
    listed = subprocess.run(  # noqa: S603 - fixed arguments, no user input
        [
            git,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *(f":(top){path}" for path in SOURCE_IDENTITY_INCLUDES),
            *(f":(top,exclude){path}" for path in SOURCE_IDENTITY_EXCLUDES),
        ],
        capture_output=True,
        check=True,
        timeout=10,
        cwd=root,
    ).stdout.decode("utf-8")
    paths = sorted({path for path in listed.split("\0") if path})
    # A tracked file deleted from the working tree is simply absent: the
    # listing it leaves behind already differs.
    return source_identity(
        (path, (root / path).read_bytes()) for path in paths if (root / path).is_file()
    )


def _print_retrieval_report(
    dataset: EvaluationDataset, report: RetrievalReport, out: TextIO
) -> None:
    answerable = len(report.answerable)
    rows: list[tuple[str, str]] = [
        ("questions", str(len(dataset.questions))),
        ("answerable", str(answerable)),
    ]
    rows += [(score.label, score.describe()) for score in report.hit_rates.values()]
    rows.append(("MRR", f"{report.mean_reciprocal_rank:.3f}"))
    rows += [
        (f"{label} filtered out", score.describe())
        for label, score in sorted(report.threshold_rejections.items())
    ]
    rows.append(("internal leaks", str(len(report.internal_leaks))))

    print(file=out)
    _print_labelled("Retrieval", tuple(rows), out)

    if not report.failures:
        print(file=out)
        print("No retrieval failures.", file=out)
        return

    print(file=out)
    print(f"Retrieval failures ({len(report.failures)}):", file=out)
    for record in report.failures:
        print(f"  ✗ {record.question.id}", file=out)
        print(f"      {record.describe_failure()}", file=out)


def _print_grounding_report(report: GroundingReport, out: TextIO) -> None:
    answered_correct, answered_total = report.correct_for(expects_an_answer=True)
    refused_correct, refused_total = report.correct_for(expects_an_answer=False)

    print(file=out)
    _print_labelled(
        "Grounding",
        (
            ("answered", str(report.answered)),
            ("no knowledge", str(report.no_knowledge)),
            ("not grounded", str(report.not_grounded)),
            ("answerable correct", f"{answered_correct}/{answered_total}"),
            ("refusals correct", f"{refused_correct}/{refused_total}"),
        ),
        out,
    )

    if not report.failures:
        print(file=out)
        print("No grounding failures.", file=out)
        return

    print(file=out)
    print(f"Grounding failures ({len(report.failures)}):", file=out)
    for record in report.failures:
        print(
            f"  ✗ {record.question.id}  outcome={record.outcome.value} "
            f"citations={record.citation_count}",
            file=out,
        )


# --- query side --------------------------------------------------------------


def _run_query(args: argparse.Namespace, out: TextIO) -> int:
    """Retrieve, or retrieve and answer. Both need the same wired stack."""
    try:
        policy = _retrieval_policy_from(args)
    except ValidationError as exc:
        print(f"Invalid retrieval policy: {_first_policy_problem(exc)}", file=sys.stderr)
        return EXIT_INVALID

    try:
        normalize_query(args.question)
    except QueryValidationError as exc:
        print(f"Invalid question: {exc.message}", file=sys.stderr)
        return EXIT_INVALID

    try:
        return asyncio.run(_query(args, policy, out))
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except IndexingError as exc:
        print(f"Indexing failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID
    except AppError as exc:
        # The query side's own failures: an unreachable store or provider, or
        # a query space that does not match the index.
        print(f"Query failed: {exc.code.value}: {exc.message}", file=sys.stderr)
        return EXIT_INVALID
    except PortError as exc:
        print(f"Query failed: {exc.describe()}", file=sys.stderr)
        return EXIT_INVALID


async def _query(args: argparse.Namespace, policy: RetrievalPolicy, out: TextIO) -> int:
    """Wire the query side for one invocation, with the run's overrides applied.

    The overrides go through the settings the composition root reads, so the
    policy the header prints is the same object retrieval and generation use —
    rather than one displayed here and another one quietly in effect.
    """
    settings: Settings = get_settings().model_copy(
        update={
            "knowledge_root": args.root,
            "retrieval_top_k": policy.top_k,
            "retrieval_min_similarity": policy.min_similarity,
        }
    )
    components = build_query_components(settings)
    try:
        await components.prepare()
        _print_query_header(args, components, out)
        if args.command == "retrieve":
            return await _run_retrieve(args, components, out)
        return await _run_answer(args, components, out)
    finally:
        await components.aclose()


def _print_query_header(args: argparse.Namespace, components: QueryComponents, out: TextIO) -> None:
    print(f"Question: {args.question}", file=out)
    print(file=out)
    _print_labelled("Embedding space", components.embeddings.spec.describe(), out)
    print(file=out)
    _print_labelled("Retrieval policy", components.retrieval.policy.describe(), out)
    print(file=out)
    _print_labelled(
        "Stack",
        (
            ("generation model", components.llm.model),
            ("corpus chunks", str(len(components.chunks))),
        ),
        out,
    )


async def _run_retrieve(args: argparse.Namespace, components: QueryComponents, out: TextIO) -> int:
    query = normalize_query(args.question)
    outcome = await components.retrieval.retrieve(query)

    print(file=out)
    _print_labelled("Retrieval", outcome.describe(), out)
    _print_retrieved(outcome, show_content=args.show_content, out=out)

    print(file=out)
    if outcome.is_sufficient:
        print(_count(len(outcome.chunks), "passage"), file=out)
    else:
        print("No passage met the retrieval criteria.", file=out)
    return EXIT_OK


async def _run_answer(args: argparse.Namespace, components: QueryComponents, out: TextIO) -> int:
    """Run the whole pipeline once, then show as much of it as was asked for.

    Everything printed comes off the single result, so ``--show-context``
    prints the context the model actually received rather than a second one
    rebuilt to resemble it.
    """
    answer = await components.answers.answer(args.question)

    if args.show_retrieval:
        print(file=out)
        _print_labelled("Retrieval", answer.retrieval.describe(), out)
        _print_retrieved(answer.retrieval, show_content=False, out=out)
    if args.show_context and answer.context is not None:
        _print_context(answer.context, out)

    _print_answer(answer, out)
    return EXIT_OK


def _print_retrieved(outcome: RetrievalOutcome, *, show_content: bool, out: TextIO) -> None:
    for position, retrieved in enumerate(outcome.chunks, start=1):
        print(file=out)
        print(f"Match {position}", file=out)
        _print_rows(_retrieved_rows(retrieved), out)
        text = retrieved.chunk.content if show_content else _preview(retrieved.chunk.content)
        print(file=out)
        for line in text.split("\n"):
            print(f"    │ {line}" if line else "    │", file=out)


def _retrieved_rows(retrieved: RetrievedChunk) -> tuple[tuple[str, str], ...]:
    chunk = retrieved.chunk
    return (
        # A similarity, not a confidence and not a percentage: it orders
        # results, it does not say how likely an answer is to be right.
        ("Similarity", f"{retrieved.similarity:.4f}"),
        ("Chunk ID", chunk.id),
        ("Document", chunk.document_metadata.title),
        ("Heading", " > ".join(chunk.heading_path) or "—"),
        ("Source", chunk.document_metadata.source),
        ("Visibility", chunk.document_metadata.visibility.value),
    )


def _print_context(context: GroundedContext, out: TextIO) -> None:
    print(file=out)
    _print_labelled("Context", context.describe(), out)
    print(file=out)
    for line in context.text.split("\n"):
        print(f"    │ {line}" if line else "    │", file=out)


def _print_answer(answer: GroundedAnswer, out: TextIO) -> None:
    print(file=out)
    _print_labelled("Answer", answer.describe(), out)
    print(file=out)
    for line in answer.answer.split("\n"):
        print(f"    │ {line}" if line else "    │", file=out)

    print(file=out)
    if not answer.citations:
        print("No sources: this answer is not grounded in the knowledge base.", file=out)
        return
    print("Sources", file=out)
    for position, citation in enumerate(answer.citations, start=1):
        section = f" > {citation.section}" if citation.section else ""
        print(f"  [{position}] {citation.title}{section}  ({citation.source})", file=out)


def _retrieval_policy_from(args: argparse.Namespace) -> RetrievalPolicy:
    """Build the policy, letting the model itself do the validating."""
    overrides = {
        name: value
        for name, value in (("top_k", args.top_k), ("min_similarity", args.min_similarity))
        if value is not None
    }
    return RetrievalPolicy(**overrides)


def _chunk_rows(chunk: KnowledgeChunk) -> tuple[tuple[str, str], ...]:
    return (
        ("ID", chunk.id),
        ("Heading", " > ".join(chunk.heading_path) or "—"),
        ("Characters", str(len(chunk.content))),
        ("Fingerprint", chunk.provenance.fingerprint),
        ("Overlap", str(chunk.provenance.overlap_prefix_chars)),
    )


def _print_labelled(title: str, rows: tuple[tuple[str, str], ...], out: TextIO) -> None:
    print(title, file=out)
    _print_rows(rows, out)


def _print_rows(rows: tuple[tuple[str, str], ...], out: TextIO) -> None:
    """Print label/value pairs in one aligned column.

    The column grows for labels that do not fit the default width, so a long
    one never ends up jammed against its value.
    """
    width = max(_LABEL_WIDTH, *(len(label) + 1 for label, _ in rows)) if rows else _LABEL_WIDTH
    for label, value in rows:
        print(f"  {label:<{width}}{value}", file=out)


def _first_policy_problem(exc: ValidationError) -> str:
    problem = exc.errors()[0]
    field = ".".join(str(part) for part in problem["loc"])
    message = str(problem["msg"]).removeprefix("Value error, ")
    return f"{field}: {message}" if field else message


def _find(report: KnowledgeBaseReport, target: str) -> KnowledgeDocument | None:
    """Match by document id first, then by source path."""
    for document in report.documents:
        if document.metadata.id == target:
            return document
    for document in report.documents:
        if document.provenance.source_path == target:
            return document
    return None


def _count(value: int, noun: str) -> str:
    return f"{value} {noun}" if value == 1 else f"{value} {noun}s"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
