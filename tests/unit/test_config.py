"""Configuration parsing and its guardrails."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_rag.core.config import (
    EmbeddingProviderName,
    Environment,
    LLMProviderName,
    LogLevel,
    Settings,
    VectorStoreName,
    get_settings,
)


def _settings(**overrides: object) -> Settings:
    """Build settings without reading a developer's local `.env`."""
    defaults: dict[str, object] = {"_env_file": None}
    return Settings(**(defaults | overrides))  # type: ignore[arg-type]


def test_defaults_are_local_development_defaults():
    settings = _settings()

    assert settings.environment is Environment.LOCAL
    assert settings.log_level is LogLevel.INFO
    assert settings.is_production is False
    assert settings.service_name == "portfolio-rag-assistant"
    assert settings.allowed_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]


def test_origins_are_parsed_from_a_comma_separated_string():
    settings = _settings(allowed_origins=" https://a.example , https://b.example ")

    assert settings.allowed_origins == ["https://a.example", "https://b.example"]


def test_cors_wildcard_is_rejected():
    with pytest.raises(ValidationError, match="wildcard"):
        _settings(allowed_origins=["https://a.example", "*"])


def test_logs_are_json_outside_local_and_readable_inside_it():
    assert _settings(environment=Environment.LOCAL).use_json_logs is False
    assert _settings(environment=Environment.PRODUCTION).use_json_logs is True
    # An explicit setting always wins over the environment-derived default.
    assert _settings(environment=Environment.PRODUCTION, json_logs=False).use_json_logs is False


def test_unknown_environment_is_rejected():
    with pytest.raises(ValidationError):
        _settings(environment="prod")


def test_settings_are_read_from_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PORTFOLIO_RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("PORTFOLIO_RAG_LOG_LEVEL", "ERROR")
    monkeypatch.setenv("PORTFOLIO_RAG_ALLOWED_ORIGINS", "https://portfolio.example")
    get_settings.cache_clear()
    try:
        settings = get_settings()

        assert settings.environment is Environment.PRODUCTION
        assert settings.log_level is LogLevel.ERROR
        assert settings.allowed_origins == ["https://portfolio.example"]
        assert settings.is_production is True
    finally:
        get_settings.cache_clear()


def test_settings_are_immutable():
    settings = _settings()

    with pytest.raises(ValidationError):
        settings.environment = Environment.PRODUCTION  # type: ignore[misc]


def test_the_query_side_defaults_are_the_offline_ones():
    """A fresh checkout answers questions with no account and no key."""
    settings = _settings()

    assert settings.embedding_provider is EmbeddingProviderName.DETERMINISTIC
    assert settings.llm_provider is LLMProviderName.DETERMINISTIC
    assert settings.vector_store is VectorStoreName.MEMORY
    assert settings.knowledge_root == Path("knowledge")


def test_development_stand_ins_are_recognisable_as_such():
    assert _settings().uses_development_providers is True
    assert (
        _settings(embedding_provider="mistral", llm_provider="mistral").uses_development_providers
        is False
    )


def test_the_embedding_model_and_the_generation_model_are_separate_settings():
    """Sending a prompt to `mistral-embed` fails a long way from the mistake."""
    settings = _settings()

    assert settings.mistral_embedding_model == "mistral-embed"
    assert settings.mistral_chat_model == "mistral-small-latest"
    assert "embed" not in settings.mistral_chat_model


def test_workers_ai_is_a_selectable_generation_provider():
    settings = _settings(embedding_provider="mistral", llm_provider="cloudflare_workers_ai")

    assert settings.llm_provider is LLMProviderName.CLOUDFLARE_WORKERS_AI
    assert settings.uses_development_providers is False, "neither half is a stand-in"


def test_the_workers_ai_model_is_configuration_with_a_default():
    settings = _settings()

    assert settings.cloudflare_workers_ai_chat_model == "@cf/openai/gpt-oss-120b"
    assert _settings(cloudflare_workers_ai_chat_model="@cf/meta/llama-3.1-8b-instruct")
    with pytest.raises(ValidationError):
        _settings(cloudflare_workers_ai_chat_model="")


def test_the_two_cloudflare_products_share_an_account_and_not_a_token():
    """A token scoped to Vectorize should not be able to spend Workers AI."""
    settings = _settings()

    assert settings.cloudflare_account_id is None
    assert settings.cloudflare_api_token is None
    assert settings.cloudflare_workers_ai_token is None


def test_the_workers_ai_token_is_read_from_its_own_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_TOKEN",
        "test-token-not-a-real-credential",
    )
    get_settings.cache_clear()
    try:
        token = get_settings().cloudflare_workers_ai_token

        assert token is not None
        assert token.get_secret_value() == "test-token-not-a-real-credential"
    finally:
        get_settings.cache_clear()


def test_retrieval_overrides_default_to_absent_rather_than_to_a_second_copy():
    """`None` means "whatever `rag.policy` says", so the numbers live in one place."""
    settings = _settings()

    assert settings.retrieval_top_k is None
    assert settings.retrieval_min_similarity is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("retrieval_top_k", 0), ("retrieval_top_k", 500), ("retrieval_min_similarity", 2.0)],
)
def test_an_unusable_retrieval_override_is_rejected_at_startup(field: str, value: object):
    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_a_credential_is_never_rendered_in_the_settings_repr():
    settings = _settings(mistral_api_key="test-key-not-a-real-credential")

    assert "test-key-not-a-real-credential" not in repr(settings)
    assert "test-key-not-a-real-credential" not in str(settings)


def test_the_workers_ai_token_is_never_rendered_in_the_settings_repr():
    settings = _settings(
        cloudflare_workers_ai_token="test-token-not-a-real-credential"  # noqa: S106 - a fixture
    )

    assert "test-token-not-a-real-credential" not in repr(settings)
    assert "test-token-not-a-real-credential" not in str(settings)
