"""Opt-in smoke tests against the real Mistral and Cloudflare services.

**Skipped unless credentials are present.** Never required, never part of CI,
never a reason for a red build. Standard development and CI run entirely on the
deterministic provider and the in-memory store, at zero cost — that is a design
constraint of this project, not an accident.

Enabling them costs money (Mistral, Workers AI) and touches a real index
(Vectorize), so both are additionally gated behind an explicit opt-in variable. Having a key in
your environment is not the same as asking for your account to be used.

    PORTFOLIO_RAG_LIVE_TESTS=1 \\
    PORTFOLIO_RAG_MISTRAL_API_KEY=... \\
    uv run pytest tests/live -v

The Vectorize test writes and then removes records under a unique prefix, and
refuses to run against anything but a dedicated test index. The generation test
sends one short request against a fixture passage invented for it.
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import date

import pytest

from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.infrastructure.embedding import MistralEmbeddingProvider
from portfolio_rag.infrastructure.vector_store import CloudflareVectorizeStore
from portfolio_rag.ports.embeddings import EmbeddingInput
from portfolio_rag.ports.llm import GenerationRequest
from portfolio_rag.ports.vector_store import VectorQuery
from tests.factories import make_metadata
from tests.support import run

LIVE_ENABLED = os.getenv("PORTFOLIO_RAG_LIVE_TESTS") == "1"

MISTRAL_KEY = os.getenv("PORTFOLIO_RAG_MISTRAL_API_KEY")
CLOUDFLARE_ACCOUNT = os.getenv("PORTFOLIO_RAG_CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_TOKEN = os.getenv("PORTFOLIO_RAG_CLOUDFLARE_API_TOKEN")
#: A dedicated index, named separately from the production one on purpose: a
#: test that can reach production data is a test that will eventually delete it.
VECTORIZE_TEST_INDEX = os.getenv("PORTFOLIO_RAG_VECTORIZE_TEST_INDEX")

#: Workers AI is a separate product with a separate token, even though it is
#: reached with the same account id as Vectorize.
WORKERS_AI_TOKEN = os.getenv("PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN")
WORKERS_AI_MODEL = os.getenv(
    "PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_CHAT_MODEL", "@cf/openai/gpt-oss-120b"
)

requires_live = pytest.mark.skipif(
    not LIVE_ENABLED, reason="live provider tests are opt-in (PORTFOLIO_RAG_LIVE_TESTS=1)"
)


@requires_live
@pytest.mark.skipif(not MISTRAL_KEY, reason="no Mistral API key configured")
def test_mistral_returns_usable_vectors_for_two_short_strings():
    """Deliberately tiny: two neutral strings, one request, negligible cost."""
    provider = MistralEmbeddingProvider(api_key=MISTRAL_KEY or "")
    inputs = [
        EmbeddingInput(id="a", text="A short neutral sentence about software."),
        EmbeddingInput(id="b", text="Another short neutral sentence about testing."),
    ]

    try:
        results = run(provider.embed(inputs))
    finally:
        run(provider.aclose())

    assert [result.id for result in results] == ["a", "b"]
    for result in results:
        assert len(result.vector) == provider.spec.dimensions
        assert all(math.isfinite(value) for value in result.vector)
    assert results[0].vector != results[1].vector


@requires_live
@pytest.mark.skipif(
    not (CLOUDFLARE_ACCOUNT and CLOUDFLARE_TOKEN and VECTORIZE_TEST_INDEX),
    reason="no dedicated Cloudflare Vectorize test index configured",
)
def test_vectorize_round_trips_a_record_and_cleans_up_after_itself():
    """Upsert, fetch, query, delete — against a real, dedicated test index.

    Every id is prefixed with a fresh UUID, so a run cannot collide with
    anything else in the index, and the cleanup only ever removes what this run
    created.
    """
    from portfolio_rag.domain.embedding import EmbeddingSpec

    dimensions = 4
    spec = EmbeddingSpec(
        provider="deterministic",
        model="live-contract-test",
        dimensions=dimensions,
        representation_version="embedding-text-v1",
    )
    store = CloudflareVectorizeStore(
        account_id=CLOUDFLARE_ACCOUNT or "",
        api_token=CLOUDFLARE_TOKEN or "",
        index_name=VECTORIZE_TEST_INDEX or "",
        index_spec=VectorIndexSpec(embedding=spec),
    )

    prefix = f"livetest-{uuid.uuid4().hex[:12]}"
    record_id = f"{prefix}--0000"
    from tests.contracts.vector_store import record as build_record

    original = build_record(record_id, (1.0, 0.0, 0.0, 0.0)).model_copy(
        update={"spec": spec, "metadata": make_metadata(record_id, visibility="public")}
    )

    try:
        run(store.upsert([original]))
        (fetched,) = run(store.fetch([record_id]))
        assert fetched.id == record_id
        assert fetched.metadata.visibility.value == "public"

        matches = run(
            store.query(
                VectorQuery(
                    embedding=(1.0, 0.0, 0.0, 0.0),
                    top_k=5,
                    filters={"visibility": "public"},
                )
            )
        )
        assert any(match.record.id == record_id for match in matches)
    finally:
        # Cleanup, and the last thing this test actually proves: `delete`
        # raises unless Vectorize answered with a mutation id, so reaching the
        # line below means the removal of this run's record was *accepted*.
        #
        # Accepted is as far as the claim goes. Vectorize applies mutations
        # asynchronously (see the adapter's module docstring), so the record
        # may still be readable for a while afterwards and its absence is not
        # observable here. There used to be an `assert fetch(...) == [] or True`
        # at this point: `or True` made it vacuous, and it sat after the client
        # had been closed, so it could only ever raise. Asserting absence needs
        # a convergence window nothing in this repository establishes, so it is
        # not asserted rather than asserted falsely.
        run(store.delete([record_id]))
        run(store.aclose())


@requires_live
@pytest.mark.skipif(not MISTRAL_KEY, reason="no Mistral API key configured")
def test_the_real_provider_reports_the_dimensionality_it_documents():
    provider = MistralEmbeddingProvider(api_key=MISTRAL_KEY or "")
    try:
        (result,) = run(provider.embed([EmbeddingInput(id="a", text="dimension check")]))
    finally:
        run(provider.aclose())

    assert len(result.vector) == provider.spec.dimensions


def _grounded_fixture_request() -> GenerationRequest:
    """One short grounded prompt over a passage invented for this file.

    Shared by every live generation test so that each provider is asked the
    same question in the same shape — which is the only way the answers say
    anything about the providers rather than about the prompts.
    """
    from portfolio_rag.domain.knowledge import (
        ChunkProvenance,
        DocumentMetadata,
        DocumentProvenance,
        KnowledgeChunk,
    )
    from portfolio_rag.domain.retrieval import RetrievedChunk
    from portfolio_rag.rag.context import build_context
    from portfolio_rag.rag.prompt import build_generation_request

    chunk = KnowledgeChunk(
        id="live-fixture--0000",
        document_id="live-fixture",
        ordinal=0,
        content="The example service uses FastAPI for its HTTP API.",
        heading_path=("Web framework",),
        document_metadata=DocumentMetadata(
            schema_version=1,
            id="live-fixture",
            title="Live Test Fixture",
            document_type="reference",
            language="en",
            source="tests/live/fixture",
            source_type="authored",
            version=1,
            updated_at=date(2026, 8, 7),
            visibility="public",
            trust_level="verified",
        ),
        provenance=ChunkProvenance(
            document=DocumentProvenance(
                source_path="live-fixture.md", document_fingerprint="0" * 64
            ),
            strategy_version="markdown-structure-v1",
            fingerprint="1" * 64,
            section_ordinal=0,
        ),
    )
    context = build_context([RetrievedChunk(chunk=chunk, similarity=0.9)], available_tokens=2000)
    return build_generation_request(
        question="Which HTTP framework does the example service use?",
        context=context,
        max_output_tokens=200,
    )


@requires_live
@pytest.mark.skipif(not MISTRAL_KEY, reason="no Mistral API key configured")
def test_mistral_answers_the_grounded_prompt_in_the_shape_it_was_asked_for():
    """One short request against a neutral fixture passage.

    The point is not answer quality — it is that the real model honours the
    contract this system depends on: structured output, and a source label that
    the backend can map. Everything in the prompt is invented for this test.
    """
    from portfolio_rag.infrastructure.llm import MistralChatProvider
    from portfolio_rag.rag.generation import parse_generation

    provider = MistralChatProvider(api_key=MISTRAL_KEY or "")
    try:
        response = run(provider.generate(_grounded_fixture_request()))
    finally:
        run(provider.aclose())

    draft = parse_generation(response.text)
    assert draft.answer
    assert draft.source_labels == ("S1",), "the model must cite the label it was given"
    assert (MISTRAL_KEY or "") not in response.text


@requires_live
@pytest.mark.skipif(
    not (CLOUDFLARE_ACCOUNT and WORKERS_AI_TOKEN),
    reason="no Cloudflare Workers AI credentials configured",
)
def test_workers_ai_answers_the_grounded_prompt_in_the_shape_it_was_asked_for():
    """The same question, against the provider this project generates with.

    `@cf/openai/gpt-oss-120b` is a reasoning model, so this also checks the
    thing that would be easiest to get wrong: that what comes back is the final
    answer and parses as the contract, rather than the model's thinking.
    """
    from portfolio_rag.infrastructure.llm import WorkersAIChatProvider
    from portfolio_rag.rag.generation import parse_generation

    provider = WorkersAIChatProvider(
        account_id=CLOUDFLARE_ACCOUNT or "",
        api_token=WORKERS_AI_TOKEN or "",
        model=WORKERS_AI_MODEL,
    )
    try:
        response = run(provider.generate(_grounded_fixture_request()))
    finally:
        run(provider.aclose())

    draft = parse_generation(response.text)
    assert draft.answer, "a reasoning model must still return a final answer"
    assert draft.source_labels == ("S1",), "the model must cite the label it was given"
    assert (WORKERS_AI_TOKEN or "") not in response.text
