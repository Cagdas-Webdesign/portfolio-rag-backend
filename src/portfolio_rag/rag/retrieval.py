"""Finding the passages that might answer a question.

    question
      → query representation      what the provider is allowed to see
      → query embedding           the existing EmbeddingProvider, no second port
      → vector search             public-only, top-k, in one compatible space
      → threshold                 weak matches are not evidence
      → chunk resolution          ids and scores back into real corpus text
      → RetrievedChunk[]          ranked, deterministic

One responsibility. This module does not build prompts, does not call a
language model and does not know one exists.

**Public by construction.** :class:`PublicRetrievalService` puts
``visibility=public`` into the query it sends to the store. There is no
parameter for it, no policy field, no keyword argument and no override — a
caller cannot ask this service for internal knowledge, correctly or
incorrectly, because the request it would have to make cannot be expressed.
That is the difference between a filter and a boundary: filtering after the
fact is a step somebody can forget, and the failure is silent and public.

Two more checks belong to the same idea. Resolved chunks are re-checked for
public visibility, which catches an index that is one deploy behind a document
that has since been reclassified — the store filtered on metadata written when
the record was indexed, and this filters on the corpus as it is now. And the
query space is compared with the index space by *identity*, not by
dimensionality: two models that both emit 1024 numbers produce vectors that are
not comparable, and an index that mixes them answers confidently and wrongly
with no error anywhere.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from portfolio_rag.core.logging import get_logger
from portfolio_rag.domain.embedding import EmbeddingSpec
from portfolio_rag.domain.knowledge import KnowledgeChunk, Visibility
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingProvider
from portfolio_rag.ports.errors import EmbeddingProviderError, VectorStoreError
from portfolio_rag.ports.knowledge import ChunkResolver
from portfolio_rag.ports.vector_store import VectorMatch, VectorQuery, VectorStore
from portfolio_rag.rag.errors import (
    EmbeddingSpaceMismatchError,
    QueryEmbeddingError,
    RetrievalFailure,
    RetrievalStage,
    RetrievalUnavailableError,
    embedding_failure,
    vector_store_failure,
)
from portfolio_rag.rag.policy import DEFAULT_RETRIEVAL_POLICY, RetrievalPolicy
from portfolio_rag.rag.query import UserQuery, build_query_embedding_text

_logger = get_logger(__name__)

#: The filter every public search carries. Not a default, not an argument.
PUBLIC_VISIBILITY_FILTER: Final[Mapping[str, str]] = {"visibility": Visibility.PUBLIC.value}

#: Handle for the single embedding input. The provider port matches results by
#: id; with one input the value is arbitrary, and it is still checked.
_QUERY_INPUT_ID: Final = "query"


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """What a search found, and what it discarded on the way.

    The counts are diagnostics, not a public response: they are what a
    developer needs to tell "the corpus does not cover this" apart from "the
    threshold is too high" apart from "the index is stale".
    """

    chunks: tuple[RetrievedChunk, ...]
    matches_returned: int
    """Candidates the store returned, before any of this module's filtering."""

    below_threshold: int
    """Candidates dropped for scoring under the policy's minimum similarity."""

    unresolved: int
    """Candidates whose chunk id no longer exists in the corpus — a stale index."""

    withheld: int
    """Candidates dropped because the corpus no longer classifies them public."""

    duration_seconds: float

    @property
    def is_sufficient(self) -> bool:
        """Whether anything survived. Nothing surviving is an answer, not a failure."""
        return bool(self.chunks)

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("matches returned", str(self.matches_returned)),
            ("below threshold", str(self.below_threshold)),
            ("unresolved", str(self.unresolved)),
            ("withheld", str(self.withheld)),
            ("retrieved", str(len(self.chunks))),
        )


class PublicRetrievalService:
    """Retrieves public knowledge for one question. Nothing else, ever."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        resolver: ChunkResolver,
        policy: RetrievalPolicy = DEFAULT_RETRIEVAL_POLICY,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._resolver = resolver
        self._policy = policy

    @property
    def policy(self) -> RetrievalPolicy:
        return self._policy

    @property
    def spec(self) -> EmbeddingSpec:
        """The space questions are embedded into."""
        return self._embeddings.spec

    async def retrieve(
        self, query: UserQuery, *, policy: RetrievalPolicy | None = None
    ) -> RetrievalOutcome:
        """Search for passages that may answer *query*.

        *policy* overrides ``top_k`` and the similarity threshold for one call —
        the retrieval inspection CLI needs that. It cannot override visibility,
        which is not part of the policy.
        """
        effective_policy = policy or self._policy
        started = time.perf_counter()

        self._require_compatible_space()
        embedding = await self._embed(query)
        matches = await self._search(embedding, effective_policy)

        qualified_matches = [
            match for match in matches if match.score >= effective_policy.min_similarity
        ]
        retrieved, unresolved, withheld = await self._resolve(qualified_matches)

        # The store already orders by score, but the guarantee is restated here
        # rather than inherited: resolution can drop entries, and a second
        # adapter's "deterministic" may not mean the same tie-break as this one.
        retrieved.sort(key=lambda item: (-item.similarity, item.chunk.id))

        return RetrievalOutcome(
            chunks=tuple(retrieved),
            matches_returned=len(matches),
            below_threshold=len(matches) - len(qualified_matches),
            unresolved=unresolved,
            withheld=withheld,
            duration_seconds=round(time.perf_counter() - started, 4),
        )

    # --- steps --------------------------------------------------------------

    async def _embed(self, query: UserQuery) -> tuple[float, ...]:
        """Turn the question into a vector, using the corpus's own provider.

        Deliberately not a separate ``QueryEmbeddingProvider``: a query and a
        chunk must land in the same space to be comparable at all, and two
        ports that are required to agree are one port with extra steps.
        """
        text = build_query_embedding_text(query)
        try:
            results = await self._embeddings.embed([EmbeddingInput(id=_QUERY_INPUT_ID, text=text)])
        except EmbeddingProviderError as exc:
            _logger.warning("query embedding failed", extra={"reason": exc.describe()})
            raise _embedding_error(embedding_failure(exc)) from exc

        if len(results) != 1 or results[0].id != _QUERY_INPUT_ID:
            _logger.warning("query embedding returned an unusable batch")
            raise _embedding_error(_lasting(RetrievalStage.EMBEDDING, "unusable_batch"))
        vector = results[0].vector
        if len(vector) != self._embeddings.spec.dimensions:
            _logger.warning("query embedding has the wrong dimensionality")
            raise _embedding_error(_lasting(RetrievalStage.EMBEDDING, "wrong_dimensionality"))
        return vector

    async def _search(
        self, embedding: tuple[float, ...], policy: RetrievalPolicy
    ) -> list[VectorMatch]:
        query = VectorQuery(
            embedding=embedding,
            top_k=policy.top_k,
            filters=PUBLIC_VISIBILITY_FILTER,
        )
        try:
            return await self._store.query(query)
        except VectorStoreError as exc:
            _logger.warning("vector search failed", extra={"reason": exc.describe()})
            raise _unavailable(vector_store_failure(exc)) from exc

    async def _resolve(
        self, matches: Sequence[VectorMatch]
    ) -> tuple[list[RetrievedChunk], int, int]:
        """Turn matches into chunks with text, dropping what must not be used."""
        if not matches:
            return [], 0, 0

        try:
            chunks = await self._resolver.resolve([match.record.id for match in matches])
        except Exception as exc:  # a resolver is I/O-backed in every real setup
            _logger.warning("chunk resolution failed", extra={"reason": type(exc).__name__})
            raise _unavailable(
                _lasting(RetrievalStage.CHUNK_RESOLUTION, type(exc).__name__)
            ) from exc

        by_id: dict[str, KnowledgeChunk] = {chunk.id: chunk for chunk in chunks}
        retrieved: list[RetrievedChunk] = []
        unresolved = 0
        withheld = 0

        for match in matches:
            chunk = by_id.get(match.record.id)
            if chunk is None:
                unresolved += 1
                continue
            if chunk.document_metadata.visibility is not Visibility.PUBLIC:
                # The index said public, the corpus says otherwise. The corpus
                # wins: it is the source of truth, and it is the newer of the two.
                withheld += 1
                continue
            retrieved.append(RetrievedChunk(chunk=chunk, similarity=match.score))

        if unresolved:
            _logger.warning("retrieved records are not in the corpus", extra={"count": unresolved})
        if withheld:
            _logger.warning("retrieved records are no longer public", extra={"count": withheld})
        return retrieved, unresolved, withheld

    # --- compatibility ------------------------------------------------------

    def _require_compatible_space(self) -> None:
        index_spec = self._store.index_spec.embedding
        query_spec = self._embeddings.spec
        if not index_spec.is_compatible_with(query_spec):
            _logger.error(
                "query and index embedding spaces differ",
                extra={"index_space": index_spec.identity, "query_space": query_spec.identity},
            )
            raise EmbeddingSpaceMismatchError


def _lasting(stage: RetrievalStage, detail: str) -> RetrievalFailure:
    """A retrieval failure that asking again would not change."""
    return RetrievalFailure(stage=stage, transient=False, detail=detail)


def _embedding_error(failure: RetrievalFailure) -> QueryEmbeddingError:
    error = QueryEmbeddingError()
    error.failure = failure
    return error


def _unavailable(failure: RetrievalFailure) -> RetrievalUnavailableError:
    error = RetrievalUnavailableError()
    error.failure = failure
    return error
