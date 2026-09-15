"""Port contracts.

The point of these tests is twofold: mypy checks structurally that a plain
class satisfying the signatures *is* a valid adapter (no base class, no
registration), and the assertions check that the value objects around the ports
validate what they promise to validate.

If a future adapter cannot be written against these protocols, that is a signal
to change the port deliberately — not to reach around it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    VectorIndexSpec,
    VectorRecord,
    VectorRecordState,
)
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingProvider, EmbeddingResult
from portfolio_rag.ports.llm import (
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    MessageRole,
    PromptMessage,
    TokenUsage,
)
from portfolio_rag.ports.vector_store import VectorMatch, VectorQuery, VectorStore
from tests.factories import make_vector_record

SPEC = EmbeddingSpec(
    provider="fake", model="fake-embed", dimensions=3, representation_version="embedding-text-v1"
)


class FakeLLM:
    """An adapter needs nothing but matching signatures."""

    @property
    def model(self) -> str:
        return "fake-model"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        return GenerationResponse(
            text=f"answered {len(request.messages)} message(s)",
            model="fake-model",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=10, output_tokens=4),
        )


class FakeEmbeddings:
    @property
    def spec(self) -> EmbeddingSpec:
        return SPEC

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        return [
            EmbeddingResult(id=item.id, vector=(float(len(item.text)), 0.0, 1.0)) for item in inputs
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}

    @property
    def index_spec(self) -> VectorIndexSpec:
        return VectorIndexSpec(embedding=SPEC)

    @property
    def supports_enumeration(self) -> bool:
        return True

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        self.records.update({record.id: record for record in records})

    async def fetch(self, ids: Sequence[str]) -> list[VectorRecord]:
        return [self.records[key] for key in ids if key in self.records]

    async def fetch_states(self, ids: Sequence[str]) -> list[VectorRecordState]:
        return [record.state() for record in await self.fetch(ids)]

    async def list_state(self) -> list[VectorRecordState]:
        return [record.state() for record in self.records.values()]

    async def delete(self, ids: Sequence[str]) -> None:
        for key in ids:
            self.records.pop(key, None)

    async def query(self, query: VectorQuery) -> list[VectorMatch]:
        return [
            VectorMatch(record=record.state(), score=1.0)
            for record in list(self.records.values())[: query.top_k]
        ]


# --- LLM --------------------------------------------------------------------


def test_an_llm_adapter_satisfies_the_port_structurally():
    provider: LLMProvider = FakeLLM()
    request = GenerationRequest(
        messages=(PromptMessage(role=MessageRole.USER, content="hi"),),
        max_output_tokens=64,
    )

    response = asyncio.run(provider.generate(request))

    assert response.text == "answered 1 message(s)"
    assert response.usage is not None


def test_a_generation_request_needs_at_least_one_message():
    with pytest.raises(ValidationError):
        GenerationRequest(messages=())


def test_generation_parameters_stay_inside_their_documented_ranges():
    with pytest.raises(ValidationError):
        GenerationRequest(
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),),
            temperature=3.0,
        )
    with pytest.raises(ValidationError):
        GenerationRequest(
            messages=(PromptMessage(role=MessageRole.USER, content="hi"),),
            max_output_tokens=0,
        )


# --- embeddings -------------------------------------------------------------


def test_an_embedding_adapter_satisfies_the_port_structurally():
    provider: EmbeddingProvider = FakeEmbeddings()
    inputs = [EmbeddingInput(id="a", text="abc"), EmbeddingInput(id="b", text="de")]

    results = asyncio.run(provider.embed(inputs))

    assert [result.id for result in results] == ["a", "b"]
    assert all(len(result.vector) == provider.spec.dimensions for result in results)


def test_embedding_inputs_and_results_reject_empty_values():
    with pytest.raises(ValidationError):
        EmbeddingInput(id="a", text="")
    with pytest.raises(ValidationError):
        EmbeddingInput(id="", text="text")
    with pytest.raises(ValidationError):
        EmbeddingResult(id="a", vector=())


# --- vector store -----------------------------------------------------------


def test_a_vector_store_adapter_satisfies_the_port_structurally():
    store: VectorStore = FakeVectorStore()
    record = make_vector_record("doc--0000", spec=SPEC, embedding=(0.1, 0.2, 0.3))

    asyncio.run(store.upsert([record]))
    matches = asyncio.run(store.query(VectorQuery(embedding=(0.1, 0.2, 0.3), top_k=5)))

    assert [match.record.id for match in matches] == ["doc--0000"]
    assert store.supports_enumeration is True

    asyncio.run(store.delete(["doc--0000"]))
    assert asyncio.run(store.list_state()) == []


def test_deleting_an_unknown_record_is_not_an_error():
    store: VectorStore = FakeVectorStore()

    asyncio.run(store.delete(["never-indexed"]))


def test_a_query_carries_exact_match_filters_and_a_bounded_top_k():
    query = VectorQuery(embedding=(1.0,), filters={"visibility": "public"})

    assert query.top_k == 5
    assert query.filters["visibility"] == "public"

    with pytest.raises(ValidationError):
        VectorQuery(embedding=(1.0,), top_k=0)
    with pytest.raises(ValidationError):
        VectorQuery(embedding=())


def test_a_record_projects_down_to_its_state_without_the_vector():
    record = make_vector_record("doc--0000", spec=SPEC, embedding=(0.1, 0.2, 0.3))

    state = record.state()

    assert state.id == record.id
    assert state.embedding_fingerprint == record.embedding_fingerprint
    assert "embedding" not in state.model_dump()
