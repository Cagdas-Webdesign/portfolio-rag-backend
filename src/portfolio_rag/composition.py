"""The composition root: the one place that turns settings into adapters.

Everywhere else in the codebase works against ports and is handed what it
needs. Here — and only here — a configuration value becomes a concrete class.
That is the whole reason a provider swap is a setting rather than a refactor,
and why nothing above this module has to know that Mistral or Cloudflare exist.

Explicit Python, not a framework: a dependency-injection container for a
handful of adapters would be more machinery than mapping. And explicitly not
dynamic imports from a configuration string — a mistyped setting should be a
validation error, not an arbitrary module being loaded.

This is also the one place where the corpus side and the query side meet. The
query side needs the text behind a retrieved chunk id, and the only source of
that text is ``knowledge/`` — so the composition root reads the corpus, chunks
it, and hands the result to a resolver. Neither side calls the other; they are
composed at the edge, exactly as the CLI has always composed loader and chunker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.core.config import (
    EmbeddingProviderName,
    Environment,
    LLMProviderName,
    Settings,
    VectorStoreName,
)
from portfolio_rag.core.logging import get_logger
from portfolio_rag.domain.embedding import EmbeddingSpec, VectorIndexSpec
from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.infrastructure.embedding import (
    DeterministicEmbeddingProvider,
    MistralEmbeddingProvider,
    mistral,
)
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.llm import (
    DeterministicLLMProvider,
    MistralChatProvider,
    WorkersAIChatProvider,
)
from portfolio_rag.infrastructure.vector_store import CloudflareVectorizeStore, InMemoryVectorStore
from portfolio_rag.ingestion import KnowledgeBaseError, load_knowledge_base
from portfolio_rag.ingestion.chunking import ChunkingError, chunk_knowledge_base
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION
from portfolio_rag.ports.embeddings import EmbeddingProvider
from portfolio_rag.ports.knowledge import ChunkResolver
from portfolio_rag.ports.llm import LLMProvider
from portfolio_rag.ports.vector_store import VectorStore
from portfolio_rag.rag.policy import ContextPolicy, RetrievalPolicy
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import GroundedAnswerService

_logger = get_logger(__name__)


class ConfigurationError(Exception):
    """A selected adapter is missing configuration it cannot work without."""


@dataclass(slots=True)
class IndexingComponents:
    """A wired indexing stack, and the means to shut it down."""

    provider: EmbeddingProvider
    store: VectorStore
    service: IndexingService

    async def aclose(self) -> None:
        """Release adapter resources — HTTP clients, mostly."""
        for component in (self.provider, self.store):
            closer = getattr(component, "aclose", None)
            if closer is not None:
                await closer()


@dataclass(slots=True)
class QueryComponents:
    """A wired query stack: everything one chat request needs.

    Built once per process, not per request: reading the corpus and opening
    HTTP clients on every question would be a latency and rate-limit problem
    invented for no reason.
    """

    embeddings: EmbeddingProvider
    store: VectorStore
    resolver: ChunkResolver
    llm: LLMProvider
    retrieval: PublicRetrievalService
    answers: GroundedAnswerService
    chunks: tuple[KnowledgeChunk, ...]
    indexes_on_start: bool

    async def prepare(self) -> None:
        """Fill the index, but only when this process is the one that owns it.

        An in-memory index lives and dies with the process, so it is empty
        until this process fills it — and a server that never fills it answers
        "not covered" to everything, correctly and uselessly. A remote index is
        the opposite case: it is filled by ``knowledge index`` from a terminal,
        it survives restarts, and rebuilding it on every boot would burn
        provider calls and have concurrent workers writing over each other.
        """
        if not self.indexes_on_start:
            return
        result = await IndexingService(self.embeddings, self.store).synchronize(self.chunks)
        _logger.info(
            "in-process index prepared",
            extra={"records_written": result.records_written, "chunk_count": len(self.chunks)},
        )

    async def aclose(self) -> None:
        """Release adapter resources — HTTP clients, mostly."""
        for component in (self.embeddings, self.store, self.llm):
            closer = getattr(component, "aclose", None)
            if closer is not None:
                await closer()


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider.

    The deterministic provider is refused in production for the same reason as
    the deterministic LLM: its vectors carry no meaning, so retrieval over them
    returns whichever passages happen to hash nearby. That is not a degraded
    assistant, it is a confident one with random evidence.
    """
    match settings.embedding_provider:
        case EmbeddingProviderName.DETERMINISTIC:
            _require_development_environment(settings, adapter="deterministic embedding provider")
            return DeterministicEmbeddingProvider()
        case EmbeddingProviderName.MISTRAL:
            if settings.mistral_api_key is None:
                raise ConfigurationError(
                    "The Mistral embedding provider needs PORTFOLIO_RAG_MISTRAL_API_KEY."
                )
            return MistralEmbeddingProvider(
                api_key=settings.mistral_api_key.get_secret_value(),
                model=settings.mistral_embedding_model,
                timeout_seconds=settings.provider_timeout_seconds,
            )


def build_embedding_spec(settings: Settings) -> EmbeddingSpec:
    """Describe the configured embedding space without building anything.

    Inspecting what *would* be embedded must not require a credential, open a
    connection or satisfy an environment rule — showing a representation is a
    local question about the corpus, not a request to a provider.
    """
    match settings.embedding_provider:
        case EmbeddingProviderName.DETERMINISTIC:
            return DeterministicEmbeddingProvider().spec
        case EmbeddingProviderName.MISTRAL:
            return EmbeddingSpec(
                provider=mistral.PROVIDER_NAME,
                model=settings.mistral_embedding_model,
                dimensions=mistral.DEFAULT_DIMENSIONS,
                representation_version=EMBEDDING_REPRESENTATION_VERSION,
            )


def build_vector_store(settings: Settings, index_spec: VectorIndexSpec) -> VectorStore:
    """Build the configured vector store for *index_spec*.

    The spec comes from the provider that will fill the index, so a store can
    refuse incompatible vectors from its first write rather than its hundredth.
    """
    match settings.vector_store:
        case VectorStoreName.MEMORY:
            return InMemoryVectorStore(index_spec)
        case VectorStoreName.VECTORIZE:
            missing = [
                name
                for name, value in (
                    ("PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID", settings.cloudflare_account_id),
                    ("PORTFOLIO_RAG_CLOUDFLARE_API_TOKEN", settings.cloudflare_api_token),
                    (
                        "PORTFOLIO_RAG_CLOUDFLARE_VECTORIZE_INDEX",
                        settings.cloudflare_vectorize_index,
                    ),
                )
                if not value
            ]
            if missing:
                raise ConfigurationError(
                    f"The Cloudflare Vectorize store needs {', '.join(missing)}."
                )
            assert settings.cloudflare_account_id is not None  # noqa: S101 - checked above
            assert settings.cloudflare_api_token is not None  # noqa: S101
            assert settings.cloudflare_vectorize_index is not None  # noqa: S101
            return CloudflareVectorizeStore(
                account_id=settings.cloudflare_account_id,
                api_token=settings.cloudflare_api_token.get_secret_value(),
                index_name=settings.cloudflare_vectorize_index,
                index_spec=index_spec,
                timeout_seconds=settings.provider_timeout_seconds,
            )


def build_llm_provider(
    settings: Settings, *, generation_attempts: int | None = None
) -> LLMProvider:
    """Build the configured generation provider.

    Generation and embedding are chosen separately, and nothing here couples
    them: Mistral embeddings with Workers AI generation is as ordinary a
    combination as either vendor on both sides. The two Cloudflare adapters
    share an account id and nothing else — Vectorize and Workers AI each carry
    their own token, so a leaked one cannot spend the other.

    ``generation_attempts`` overrides how many transport attempts the adapter
    may make per call. ``None`` leaves each adapter on its own default, which
    is what every production path uses. The one caller that passes anything is
    the evaluation, which passes ``1``: a batch runner does its own retrying
    with the provider's own ``Retry-After``, and two retry budgets stacked on
    top of each other multiply into a burst that a rate-limited account sees as
    an attack. It is a number, not a policy object, because that is all either
    adapter needs and a policy type here would have exactly one field. Unset it
    is passed through as ``None``, which each adapter reads as "use your own
    default" — so the numbers stay in one place.

    Refuses the development stub in a production environment. A public site
    that answers with a placeholder looks exactly like a working assistant
    until somebody reads the text, and by then it has been answering strangers
    for a while. Failing to start is the loud version of the same problem.
    """
    match settings.llm_provider:
        case LLMProviderName.DETERMINISTIC:
            _require_development_environment(settings, adapter="deterministic LLM provider")
            return DeterministicLLMProvider()
        case LLMProviderName.MISTRAL:
            if settings.mistral_api_key is None:
                raise ConfigurationError(
                    "The Mistral chat provider needs PORTFOLIO_RAG_MISTRAL_API_KEY."
                )
            return MistralChatProvider(
                api_key=settings.mistral_api_key.get_secret_value(),
                model=settings.mistral_chat_model,
                timeout_seconds=settings.provider_timeout_seconds,
                max_attempts=generation_attempts,
            )
        case LLMProviderName.CLOUDFLARE_WORKERS_AI:
            missing = [
                name
                for name, value in (
                    ("PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID", settings.cloudflare_account_id),
                    (
                        "PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN",
                        settings.cloudflare_workers_ai_token,
                    ),
                )
                if not value
            ]
            if missing:
                raise ConfigurationError(
                    f"The Cloudflare Workers AI chat provider needs {', '.join(missing)}."
                )
            assert settings.cloudflare_account_id is not None  # noqa: S101 - checked above
            assert settings.cloudflare_workers_ai_token is not None  # noqa: S101
            return WorkersAIChatProvider(
                account_id=settings.cloudflare_account_id,
                api_token=settings.cloudflare_workers_ai_token.get_secret_value(),
                model=settings.cloudflare_workers_ai_chat_model,
                timeout_seconds=settings.provider_timeout_seconds,
                max_attempts=generation_attempts,
            )


def build_retrieval_policy(settings: Settings) -> RetrievalPolicy:
    """Apply the environment's overrides on top of the defaults in `rag.policy`."""
    overrides = {
        name: value
        for name, value in (
            ("top_k", settings.retrieval_top_k),
            ("min_similarity", settings.retrieval_min_similarity),
        )
        if value is not None
    }
    return RetrievalPolicy(**overrides)


def load_corpus_chunks(root: Path) -> tuple[KnowledgeChunk, ...]:
    """Read and chunk the knowledge base — strictly.

    The strict loader, not the collecting one: a server that starts with three
    documents quietly missing answers confidently and incompletely, which is
    the failure mode this project has been avoiding since Phase 2. A broken
    corpus should stop a deploy, not degrade one.
    """
    try:
        documents = load_knowledge_base(root)
        return chunk_knowledge_base(documents)
    except (KnowledgeBaseError, ChunkingError) as exc:
        raise ConfigurationError(
            f"The knowledge base at {root} could not be loaded: {exc}"
        ) from exc


def build_query_components(
    settings: Settings, *, generation_attempts: int | None = None
) -> QueryComponents:
    """Wire the whole query side: corpus, retrieval, generation, orchestration.

    Nothing is called yet — no embedding, no index read, no provider request.
    :meth:`QueryComponents.prepare` is the step that touches the outside world,
    so wiring can be checked without one.

    ``generation_attempts`` is forwarded to the generation adapter unchanged;
    see :func:`build_llm_provider` for the one caller that sets it.
    """
    embeddings = build_embedding_provider(settings)
    store = build_vector_store(settings, VectorIndexSpec(embedding=embeddings.spec))
    chunks = load_corpus_chunks(settings.knowledge_root)
    resolver = InMemoryChunkResolver(chunks)
    llm = build_llm_provider(settings, generation_attempts=generation_attempts)

    retrieval = PublicRetrievalService(
        embeddings=embeddings,
        store=store,
        resolver=resolver,
        policy=build_retrieval_policy(settings),
    )
    return QueryComponents(
        embeddings=embeddings,
        store=store,
        resolver=resolver,
        llm=llm,
        retrieval=retrieval,
        answers=GroundedAnswerService(
            retrieval=retrieval,
            llm=llm,
            context_policy=ContextPolicy(),
            deadline_seconds=settings.request_deadline_seconds,
        ),
        chunks=chunks,
        indexes_on_start=settings.vector_store is VectorStoreName.MEMORY,
    )


def build_indexing_components(settings: Settings) -> IndexingComponents:
    """Wire provider, store and service together for one indexing run."""
    provider = build_embedding_provider(settings)
    store = build_vector_store(settings, VectorIndexSpec(embedding=provider.spec))
    return IndexingComponents(
        provider=provider,
        store=store,
        service=IndexingService(provider, store),
    )


def _require_development_environment(settings: Settings, *, adapter: str) -> None:
    if settings.environment is Environment.PRODUCTION:
        raise ConfigurationError(
            f"The {adapter} is a development stand-in and cannot be used in a "
            "production environment. Configure a real provider, or run with "
            "PORTFOLIO_RAG_ENVIRONMENT=local."
        )
