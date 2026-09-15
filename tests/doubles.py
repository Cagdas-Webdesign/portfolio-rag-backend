"""Test doubles for the query side.

Three of these deserve an explanation, because "why not use the real adapter?"
has a different answer for each.

:class:`ScriptedLLMProvider` exists because the orchestration has branches a
real model cannot be asked to take on demand — an unknown label, a duplicate
label, an empty answer, malformed JSON, a provider outage. It is test
infrastructure, not a simulated assistant: it returns exactly what a test told
it to return, in order.

:class:`LexicalEmbeddingProvider` exists because
:class:`~portfolio_rag.infrastructure.embedding.DeterministicEmbeddingProvider`
is honest about having no semantics: it derives vectors from SHA-256, so a
question and the passage that answers it are as unrelated as a question and any
other passage. That is exactly right for indexing tests and useless for
retrieval tests, where "the relevant passage ranks above the irrelevant one"
*is* the assertion. This is a real technique — bag-of-words feature hashing —
so similarity here means genuine word overlap, offline and for free. It is
still not a semantic model: it knows nothing about meaning, only about words
appearing in both texts.

The failing adapters exist because provider outages have to be tested, and
unplugging the network is not a test.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

from portfolio_rag.domain.embedding import EmbeddingSpec, VectorIndexSpec
from portfolio_rag.domain.knowledge import (
    ChunkProvenance,
    DocumentMetadata,
    DocumentProvenance,
    KnowledgeChunk,
)
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingResult
from portfolio_rag.ports.errors import (
    EmbeddingProviderError,
    LLMProviderError,
    VectorStoreError,
)
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse, MessageRole
from portfolio_rag.ports.vector_store import VectorMatch, VectorQuery

# --- generation --------------------------------------------------------------

#: Source labels as the context builder writes them.
_SOURCE_LABEL: Final = re.compile(r"\[SOURCE (S\d+)\]")


@dataclass(frozen=True, slots=True)
class ScriptedReply:
    """One thing the scripted provider will do when called."""

    answer: str | None = None
    sources: tuple[str, ...] = ()
    raw_text: str | None = None
    """Bypasses the JSON contract entirely — for malformed-output tests."""

    error: Exception | None = None

    def render(self) -> str:
        if self.raw_text is not None:
            return self.raw_text
        return json.dumps({"answer": self.answer or "", "sources": list(self.sources)})


def grounded(answer: str, *sources: str) -> ScriptedReply:
    return ScriptedReply(answer=answer, sources=sources)


class ScriptedLLMProvider:
    """Returns what a test scripted, and records what it was asked."""

    MODEL: Final = "scripted-test-model"

    def __init__(self, *replies: ScriptedReply) -> None:
        self._replies = list(replies) or [grounded("A scripted answer.", "S1")]
        self.requests: list[GenerationRequest] = []

    @property
    def model(self) -> str:
        return self.MODEL

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        reply = self._replies[min(len(self.requests) - 1, len(self._replies) - 1)]
        if reply.error is not None:
            raise reply.error
        return GenerationResponse(text=reply.render(), model=self.MODEL, finish_reason="stop")


class DecidingLLMProvider:
    """A provider whose *judgement* is scripted, so the pipeline can be measured.

    The evaluation needs to know what the system does when a model says "these
    passages answer the question" and when it says "they do not". A real model
    cannot be asked to take a specific branch on demand, and the shipped
    development stub cites whatever it is handed — which measures the stub.

    So the decision is injected: questions listed in *declines* get an empty
    source list, everything else cites the first label in the prompt. Nothing
    here simulates understanding; the model's answer is the *input* to the
    measurement, and what is measured is the routing, the citation validation
    and the refusal path around it.
    """

    MODEL: Final = "deciding-test-model"

    def __init__(self, declines: Iterable[str] = ()) -> None:
        self._declines = tuple(declines)
        self.requests: list[GenerationRequest] = []

    @property
    def model(self) -> str:
        return self.MODEL

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        user = "\n".join(
            message.content for message in request.messages if message.role is MessageRole.USER
        )
        question = user.rpartition("QUESTION\n")[2].strip()

        if any(declined in question for declined in self._declines):
            payload = {"answer": "The provided passages do not cover this.", "sources": []}
        else:
            labels = _SOURCE_LABEL.findall(user)
            payload = {
                "answer": "A scripted grounded answer.",
                "sources": labels[:1],
            }
        return GenerationResponse(text=json.dumps(payload), model=self.MODEL, finish_reason="stop")


class FailingLLMProvider:
    """A generation provider that is simply not there."""

    def __init__(self, error: Exception | None = None) -> None:
        self._error = error or LLMProviderError("provider unreachable", retryable=True)

    @property
    def model(self) -> str:
        return "failing-test-model"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        raise self._error


# --- embeddings --------------------------------------------------------------

_WORD: Final = re.compile(r"\w+", re.UNICODE)

#: Function words carry no topic and would otherwise dominate the similarity
#: between two short texts — "the" appearing in both a question about pizza and
#: a passage about storage is not evidence of anything. A real embedding model
#: learns to discount them; a bag of words has to be told.
_STOPWORD_TEXT: Final = (
    "a an and are as at be by do does for from has have how in is it its of on or "
    "that the this to use used uses was what when where which who why with your "
    "das dass der die ein eine einem einen einer eines es fuer für ist mit sich "
    "sie und von was welche welcher welches wie wird zu zum zur über"
)
_STOPWORDS: Final[frozenset[str]] = frozenset(_STOPWORD_TEXT.split())


class LexicalEmbeddingProvider:
    """Hashed bag of words — deterministic, offline, and actually comparable."""

    PROVIDER: Final = "lexical-test"
    MODEL: Final = "hashed-bow-v1"
    DIMENSIONS: Final = 64

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self._spec = EmbeddingSpec(
            provider=self.PROVIDER,
            model=self.MODEL,
            dimensions=dimensions,
            representation_version=EMBEDDING_REPRESENTATION_VERSION,
        )
        self.embedded: list[str] = []

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    @property
    def index_spec(self) -> VectorIndexSpec:
        return VectorIndexSpec(embedding=self._spec)

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        self.embedded.extend(item.text for item in inputs)
        return [EmbeddingResult(id=item.id, vector=self.vector_for(item.text)) for item in inputs]

    def vector_for(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self._spec.dimensions
        for word in _WORD.findall(text.lower()):
            if len(word) < 3 or word in _STOPWORDS:
                continue
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            values[int.from_bytes(digest[:4], "big") % self._spec.dimensions] += 1.0

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            # An input with no words at all. A zero vector scores 0 against
            # everything, which is the honest answer for "no shared words".
            values[0] = 1.0
            norm = 1.0
        return tuple(value / norm for value in values)


class FailingEmbeddingProvider:
    """An embedding provider that cannot be reached."""

    def __init__(self, spec: EmbeddingSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        raise EmbeddingProviderError("provider unreachable", retryable=True)


class MiscountingEmbeddingProvider:
    """Returns a batch that does not match what was asked for."""

    def __init__(
        self, spec: EmbeddingSpec, *, results: list[EmbeddingResult] | None = None
    ) -> None:
        self._spec = spec
        self._results = results if results is not None else []

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        return list(self._results)


# --- storage -----------------------------------------------------------------


class FailingVectorStore:
    """A vector store whose every search fails."""

    def __init__(self, index_spec: VectorIndexSpec) -> None:
        self._index_spec = index_spec

    @property
    def index_spec(self) -> VectorIndexSpec:
        return self._index_spec

    @property
    def supports_enumeration(self) -> bool:
        return True

    async def upsert(self, records: Sequence[Any]) -> None:
        raise VectorStoreError("store unreachable", retryable=True)

    async def fetch(self, ids: Sequence[str]) -> list[Any]:
        raise VectorStoreError("store unreachable", retryable=True)

    async def fetch_states(self, ids: Sequence[str]) -> list[Any]:
        raise VectorStoreError("store unreachable", retryable=True)

    async def list_state(self) -> list[Any]:
        raise VectorStoreError("store unreachable", retryable=True)

    async def delete(self, ids: Sequence[str]) -> None:
        raise VectorStoreError("store unreachable", retryable=True)

    async def query(self, query: VectorQuery) -> list[VectorMatch]:
        raise VectorStoreError("store unreachable", retryable=True)


class FailingChunkResolver:
    """A resolver whose backing store is unavailable."""

    async def resolve(self, chunk_ids: Sequence[str]) -> list[KnowledgeChunk]:
        raise RuntimeError("corpus unavailable")


# --- corpus ------------------------------------------------------------------


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "doc",
    title: str = "Test Document",
    heading_path: tuple[str, ...] = ("Section",),
    visibility: str = "public",
    source: str = "docs/test.md",
    ordinal: int = 0,
    **metadata: Any,
) -> KnowledgeChunk:
    """A valid chunk with everything a retrieval test does not care about filled in."""
    fields: dict[str, Any] = {
        "schema_version": 1,
        "id": document_id,
        "title": title,
        "document_type": "reference",
        "language": "en",
        "source": source,
        "source_type": "authored",
        "version": 1,
        "updated_at": date(2026, 8, 7),
        "visibility": visibility,
        "trust_level": "verified",
    }
    fields.update(metadata)
    digest = hashlib.sha256(chunk_id.encode()).hexdigest()
    return KnowledgeChunk(
        id=chunk_id,
        document_id=document_id,
        ordinal=ordinal,
        content=content,
        heading_path=heading_path,
        document_metadata=DocumentMetadata.model_validate(fields),
        provenance=ChunkProvenance(
            document=DocumentProvenance(
                source_path=f"{document_id}.md",
                document_fingerprint=hashlib.sha256(document_id.encode()).hexdigest(),
            ),
            strategy_version="markdown-structure-v1",
            fingerprint=digest,
            section_ordinal=0,
        ),
    )
