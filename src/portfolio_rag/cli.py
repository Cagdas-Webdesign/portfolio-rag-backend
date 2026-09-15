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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TextIO

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
from portfolio_rag.core.config import Settings, get_settings
from portfolio_rag.core.errors import AppError
from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.domain.knowledge import KnowledgeChunk, KnowledgeDocument
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.evaluation import (
    EvaluationDataset,
    EvaluationDatasetError,
    GenerationPacing,
    GroundingReport,
    PacedLLMProvider,
    RetrievalReport,
    load_dataset,
    resolve_corpus,
    run_grounding_evaluation,
    run_retrieval_evaluation,
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
from portfolio_rag.rag.context import GroundedContext
from portfolio_rag.rag.errors import QueryValidationError
from portfolio_rag.rag.policy import RetrievalPolicy
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import RetrievalOutcome
from portfolio_rag.rag.service import GroundedAnswer, GroundedAnswerService

#: Exit codes. 2 is left to argparse for usage errors.
EXIT_OK: Final = 0
EXIT_INVALID: Final = 1
EXIT_INTERNAL_ERROR: Final = 3

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
    _add_retrieval_options(run)


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
        dataset = load_dataset(args.dataset)
    except EvaluationDatasetError as exc:
        print(f"Evaluation dataset error: {exc}", file=sys.stderr)
        return EXIT_INVALID

    corpus = resolve_corpus(args.dataset, dataset)
    try:
        return asyncio.run(_evaluate(args, dataset, corpus, policy, pacing, out))
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
) -> int:
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
    try:
        await components.prepare()

        print(f"Dataset: {args.dataset}  (version {dataset.version})", file=out)
        print(f"Corpus:  {corpus}", file=out)
        print(file=out)
        _print_labelled("Embedding space", components.embeddings.spec.describe(), out)
        print(file=out)
        _print_labelled("Retrieval policy", components.retrieval.policy.describe(), out)

        report = await run_retrieval_evaluation(dataset, components.retrieval)
        _print_retrieval_report(dataset, report, out)

        if args.retrieval_only:
            print(file=out)
            print("Retrieval only: no generation provider was called.", file=out)
            return EXIT_OK if not report.internal_leaks else EXIT_INVALID

        answers = components.answers if pacing is None else _paced_answers(components, pacing)
        if pacing is not None:
            print(file=out)
            _print_labelled("Generation pacing", pacing.describe(), out)

        grounding = await run_grounding_evaluation(dataset, answers)
        _print_grounding_report(grounding, out)

        # A leak is the one result that makes the run itself a failure. Weak
        # retrieval is a number to act on; published internal knowledge is a bug.
        return EXIT_INVALID if report.internal_leaks else EXIT_OK
    finally:
        await components.aclose()


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
