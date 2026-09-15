"""The composition root: what it builds, and what it refuses to build.

The refusals matter most. A development stand-in reaching a production
deployment would be a service that looks like it works — a fake answer on a
public page, or retrieval over vectors that mean nothing — and the only place
that can be prevented once, for every entry point, is here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from portfolio_rag.composition import (
    ConfigurationError,
    build_embedding_provider,
    build_llm_provider,
    build_query_components,
    build_retrieval_policy,
    load_corpus_chunks,
)
from portfolio_rag.core.config import (
    EmbeddingProviderName,
    Environment,
    LLMProviderName,
    LogLevel,
    Settings,
    VectorStoreName,
)
from portfolio_rag.infrastructure.embedding import (
    DeterministicEmbeddingProvider,
    MistralEmbeddingProvider,
)
from portfolio_rag.infrastructure.llm import (
    DeterministicLLMProvider,
    MistralChatProvider,
    WorkersAIChatProvider,
)
from portfolio_rag.rag.policy import DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K
from portfolio_rag.rag.service import AnswerOutcome
from tests.support import run

RAG_ROOT = Path("tests/fixtures/knowledge/rag")

FAKE_KEY = "test-key-not-a-real-credential"
FAKE_ACCOUNT = "test-account-id"
FAKE_WORKERS_AI_TOKEN = "test-token-not-a-real-credential"  # noqa: S105 - a fixture value


def workers_ai(**overrides: object) -> Settings:
    """Settings that select Workers AI for generation and nothing else."""
    fields: dict[str, object] = {
        "llm_provider": LLMProviderName.CLOUDFLARE_WORKERS_AI,
        "cloudflare_account_id": FAKE_ACCOUNT,
        "cloudflare_workers_ai_token": FAKE_WORKERS_AI_TOKEN,
    }
    fields.update(overrides)
    return local(**fields)


def local(**overrides: object) -> Settings:
    fields: dict[str, object] = {
        "environment": Environment.LOCAL,
        "log_level": LogLevel.WARNING,
        "knowledge_root": RAG_ROOT,
    }
    fields.update(overrides)
    return Settings(**fields)  # type: ignore[arg-type]


# --- development defaults ----------------------------------------------------


def test_the_defaults_build_a_stack_that_needs_nothing():
    components = build_query_components(local())

    assert isinstance(components.embeddings, DeterministicEmbeddingProvider)
    assert isinstance(components.llm, DeterministicLLMProvider)
    assert components.chunks


def test_an_in_memory_index_is_filled_by_the_process_that_owns_it():
    components = build_query_components(local())

    run(components.prepare())

    assert components.indexes_on_start
    assert run(components.store.list_state())


def test_the_wired_stack_answers_a_question_offline():
    components = build_query_components(local(retrieval_min_similarity=-1.0))
    run(components.prepare())

    answer = run(components.answers.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    run(components.aclose())


def test_a_remote_index_is_not_rebuilt_on_every_start():
    """It is filled by `knowledge index`, survives restarts, and has other writers."""
    settings = local(
        vector_store=VectorStoreName.VECTORIZE,
        cloudflare_account_id="account",
        cloudflare_api_token=FAKE_KEY,
        cloudflare_vectorize_index="test-index",
    )

    components = build_query_components(settings)

    assert not components.indexes_on_start
    run(components.aclose())


# --- production safety -------------------------------------------------------


def test_the_deterministic_generation_stub_cannot_be_built_in_production():
    with pytest.raises(ConfigurationError, match="development stand-in"):
        build_llm_provider(local(environment=Environment.PRODUCTION))


def test_the_deterministic_embedding_provider_cannot_be_built_in_production():
    with pytest.raises(ConfigurationError, match="development stand-in"):
        build_embedding_provider(local(environment=Environment.PRODUCTION))


def test_a_production_application_refuses_to_start_on_development_defaults():
    """Loudly failing to start beats quietly answering strangers with a stub."""
    with pytest.raises(ConfigurationError):
        build_query_components(local(environment=Environment.PRODUCTION))


def test_a_production_application_starts_with_real_providers():
    settings = local(
        environment=Environment.PRODUCTION,
        embedding_provider=EmbeddingProviderName.MISTRAL,
        llm_provider=LLMProviderName.MISTRAL,
        mistral_api_key=FAKE_KEY,
    )

    components = build_query_components(settings)

    assert isinstance(components.embeddings, MistralEmbeddingProvider)
    assert isinstance(components.llm, MistralChatProvider)
    run(components.aclose())


def test_a_real_provider_without_its_credential_is_refused():
    with pytest.raises(ConfigurationError, match="MISTRAL_API_KEY"):
        build_llm_provider(local(llm_provider=LLMProviderName.MISTRAL))


# --- Cloudflare Workers AI ----------------------------------------------------


def test_workers_ai_is_selectable_as_the_generation_provider():
    provider = build_llm_provider(workers_ai())

    assert isinstance(provider, WorkersAIChatProvider)
    run(provider.aclose())


def test_workers_ai_generates_with_the_configured_model():
    provider = build_llm_provider(
        workers_ai(cloudflare_workers_ai_chat_model="@cf/meta/llama-3.1-8b-instruct")
    )

    assert provider.model == "@cf/meta/llama-3.1-8b-instruct"
    run(provider.aclose())  # type: ignore[attr-defined]


def test_the_workers_ai_default_model_is_the_one_this_project_runs_on():
    provider = build_llm_provider(workers_ai())

    assert provider.model == "@cf/openai/gpt-oss-120b"
    run(provider.aclose())  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("cloudflare_account_id", "CLOUDFLARE_ACCOUNT_ID"),
        ("cloudflare_workers_ai_token", "CLOUDFLARE_WORKERS_AI_TOKEN"),
    ],
)
def test_workers_ai_without_its_configuration_is_refused(missing: str, expected: str):
    with pytest.raises(ConfigurationError, match=expected):
        build_llm_provider(workers_ai(**{missing: None}))


def test_the_vectorize_token_does_not_stand_in_for_the_workers_ai_one():
    """Two products, two tokens. A Vectorize token must not spend Workers AI."""
    settings = workers_ai(cloudflare_workers_ai_token=None, cloudflare_api_token=FAKE_KEY)

    with pytest.raises(ConfigurationError, match="CLOUDFLARE_WORKERS_AI_TOKEN"):
        build_llm_provider(settings)


def test_the_target_stack_is_mistral_embeddings_with_workers_ai_generation():
    """The architecture this project migrated to, wired end to end."""
    settings = workers_ai(
        environment=Environment.PRODUCTION,
        embedding_provider=EmbeddingProviderName.MISTRAL,
        mistral_api_key=FAKE_KEY,
    )

    components = build_query_components(settings)

    assert isinstance(components.embeddings, MistralEmbeddingProvider)
    assert isinstance(components.llm, WorkersAIChatProvider)
    assert components.embeddings.spec.model == "mistral-embed"
    run(components.aclose())


def test_switching_generation_to_workers_ai_leaves_the_embedding_space_alone():
    """The migration must not invalidate a single vector in the index.

    Embeddings are compared by `EmbeddingSpec` identity, and that identity is
    what decides whether the existing Vectorize index is still usable. If
    changing the generation provider moved it, every chunk would have to be
    re-embedded.
    """
    mistral_generation = local(
        embedding_provider=EmbeddingProviderName.MISTRAL,
        llm_provider=LLMProviderName.MISTRAL,
        mistral_api_key=FAKE_KEY,
    )
    workers_ai_generation = workers_ai(
        embedding_provider=EmbeddingProviderName.MISTRAL, mistral_api_key=FAKE_KEY
    )

    before = build_embedding_provider(mistral_generation).spec
    after = build_embedding_provider(workers_ai_generation).spec

    assert before == after
    assert before.identity == after.identity


# --- the generation retry budget ---------------------------------------------


@pytest.mark.parametrize(
    "settings_for",
    [
        lambda: local(llm_provider=LLMProviderName.MISTRAL, mistral_api_key=FAKE_KEY),
        workers_ai,
    ],
    ids=["mistral", "workers_ai"],
)
def test_a_generation_adapter_keeps_its_own_retry_budget_by_default(settings_for):
    """What every production path gets: the adapter's own number, not a copy."""
    provider = build_llm_provider(settings_for())

    assert provider.max_attempts == 3  # type: ignore[attr-defined]
    run(provider.aclose())  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "settings_for",
    [
        lambda: local(llm_provider=LLMProviderName.MISTRAL, mistral_api_key=FAKE_KEY),
        workers_ai,
    ],
    ids=["mistral", "workers_ai"],
)
def test_the_transport_budget_can_be_switched_off_for_a_caller_that_owns_it(settings_for):
    """One attempt, so a retrying caller above it cannot multiply the budget."""
    provider = build_llm_provider(settings_for(), generation_attempts=1)

    assert provider.max_attempts == 1  # type: ignore[attr-defined]
    run(provider.aclose())  # type: ignore[attr-defined]


def test_the_override_reaches_the_adapter_through_the_whole_query_stack():
    components = build_query_components(
        workers_ai(embedding_provider=EmbeddingProviderName.MISTRAL, mistral_api_key=FAKE_KEY),
        generation_attempts=1,
    )

    assert components.llm.max_attempts == 1  # type: ignore[attr-defined]
    run(components.aclose())


def test_the_development_stub_ignores_a_budget_it_has_no_transport_for():
    """No HTTP, nothing to retry — and passing the knob must not break wiring."""
    provider = build_llm_provider(local(), generation_attempts=1)

    assert isinstance(provider, DeterministicLLMProvider)


# --- provider independence ---------------------------------------------------


def test_embedding_and_generation_models_are_configured_separately():
    settings = local(
        llm_provider=LLMProviderName.MISTRAL,
        mistral_api_key=FAKE_KEY,
        mistral_chat_model="mistral-medium-latest",
        mistral_embedding_model="mistral-embed",
    )

    provider = build_llm_provider(settings)

    assert provider.model == "mistral-medium-latest"
    run(provider.aclose())  # type: ignore[attr-defined]


def test_a_real_generation_provider_pairs_with_the_local_embedding_one():
    """The two halves are chosen independently; neither implies the other."""
    settings = local(llm_provider=LLMProviderName.MISTRAL, mistral_api_key=FAKE_KEY)

    components = build_query_components(settings)

    assert isinstance(components.embeddings, DeterministicEmbeddingProvider)
    assert isinstance(components.llm, MistralChatProvider)
    run(components.aclose())


def test_mistral_generation_remains_available_after_the_workers_ai_migration():
    """The previous provider is an alternative, not a casualty."""
    settings = local(llm_provider=LLMProviderName.MISTRAL, mistral_api_key=FAKE_KEY)

    provider = build_llm_provider(settings)

    assert isinstance(provider, MistralChatProvider)
    run(provider.aclose())


def test_a_real_embedding_provider_pairs_with_the_local_generation_stub():
    settings = local(embedding_provider=EmbeddingProviderName.MISTRAL, mistral_api_key=FAKE_KEY)

    components = build_query_components(settings)

    assert isinstance(components.embeddings, MistralEmbeddingProvider)
    assert isinstance(components.llm, DeterministicLLMProvider)
    run(components.aclose())


# --- policy overrides --------------------------------------------------------


def test_the_policy_defaults_come_from_the_rag_layer_when_nothing_overrides_them():
    policy = build_retrieval_policy(local())

    assert policy.top_k == DEFAULT_TOP_K
    assert policy.min_similarity == DEFAULT_MIN_SIMILARITY


def test_the_environment_can_override_the_policy():
    policy = build_retrieval_policy(local(retrieval_top_k=3, retrieval_min_similarity=0.4))

    assert (policy.top_k, policy.min_similarity) == (3, 0.4)


# --- the corpus --------------------------------------------------------------


def test_a_corpus_that_cannot_be_loaded_stops_the_build():
    """A server missing three documents answers confidently and incompletely."""
    with pytest.raises(ConfigurationError, match="could not be loaded"):
        load_corpus_chunks(Path("tests/fixtures/knowledge/invalid"))


def test_a_missing_corpus_directory_stops_the_build(tmp_path: Path):
    with pytest.raises(ConfigurationError, match="could not be loaded"):
        load_corpus_chunks(tmp_path / "nowhere")


def test_an_empty_corpus_is_allowed_and_simply_answers_nothing(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()

    components = build_query_components(local(knowledge_root=empty))
    run(components.prepare())
    answer = run(components.answers.answer("anything at all"))

    assert answer.outcome is AnswerOutcome.NO_KNOWLEDGE


def test_every_chunk_of_the_corpus_is_resolvable_by_id():
    components = build_query_components(local())

    resolved = run(components.resolver.resolve([chunk.id for chunk in components.chunks]))

    assert len(resolved) == len(components.chunks)
