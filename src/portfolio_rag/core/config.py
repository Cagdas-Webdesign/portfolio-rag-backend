"""Typed application configuration.

Every setting is read from the environment (or a local ``.env`` file) and
validated once at startup. Nothing in the codebase should read ``os.environ``
directly — import :func:`get_settings` instead.

**Credentials are optional, and that is deliberate.** The default provider and
store are the local, offline ones, so a fresh checkout runs the whole pipeline
— ingestion, indexing, retrieval and a grounded answer — with no account and no
key. Real providers are selected explicitly and then require their credentials.
Nothing here holds a secret value in the repository; secrets come from the
environment as :class:`~pydantic.SecretStr` and are never logged.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Final

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from portfolio_rag import SERVICE_NAME, __version__

ENV_PREFIX: Final = "PORTFOLIO_RAG_"


class Environment(StrEnum):
    """Deployment environment the process is running in."""

    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class EmbeddingProviderName(StrEnum):
    """Which embedding adapter the composition root should build."""

    DETERMINISTIC = "deterministic"
    MISTRAL = "mistral"


class LLMProviderName(StrEnum):
    """Which generation adapter the composition root should build.

    ``DETERMINISTIC`` is a development stub that does not generate language.
    Selecting it in a production environment is refused by the composition
    root: a public site answering with a stub would look like a working
    assistant and be one only by accident.
    """

    DETERMINISTIC = "deterministic"
    MISTRAL = "mistral"
    CLOUDFLARE_WORKERS_AI = "cloudflare_workers_ai"


class VectorStoreName(StrEnum):
    """Which vector store adapter the composition root should build."""

    MEMORY = "memory"
    VECTORIZE = "vectorize"


class LogLevel(StrEnum):
    """Supported logging levels (mirrors the stdlib ``logging`` level names)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    log_level: LogLevel = LogLevel.INFO

    #: Version reported by ``/health`` and OpenAPI. Overridable so a build
    #: pipeline can stamp e.g. a git describe value without a code change.
    version: str = Field(default=__version__, min_length=1, max_length=64)

    #: Browser origins allowed to call the API. Comma-separated in the
    #: environment, e.g. ``http://localhost:5173,https://example.com``.
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"],
    )

    #: ``None`` means "decide from the environment" (see :attr:`use_json_logs`).
    json_logs: bool | None = None

    #: Directory the knowledge corpus is read from — by the CLI, and at startup
    #: to resolve retrieved passages back to their text.
    knowledge_root: Path = Path("knowledge")

    # --- providers and stores -----------------------------------------------
    # Defaults are the local ones: everything below runs with no account, no
    # key and no network, which is what keeps development and CI free.

    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.DETERMINISTIC
    llm_provider: LLMProviderName = LLMProviderName.DETERMINISTIC
    vector_store: VectorStoreName = VectorStoreName.MEMORY

    #: Seconds before an outbound provider or store request is abandoned.
    provider_timeout_seconds: float = Field(default=30.0, gt=0)

    #: The longest one chat request may take, end to end: retrieval, every
    #: generation, the grounding check and every transport retry beneath them.
    #: When it passes, the request is cancelled and ends as a technical error —
    #: in the application's own error envelope, before the platform's request
    #: timeout (recommended 90s on Cloud Run) cuts it off without one.
    #: **Provisional**: chosen below that timeout, not measured. To be re-set
    #: from unpaced production latencies; an evaluation's elapsed times include
    #: pacing waits and must not be used for it.
    request_deadline_seconds: float = Field(default=60.0, gt=0)

    mistral_api_key: SecretStr | None = None
    #: Embedding and generation are separate models, and mixing them up is a
    #: mistake that produces an error only at the far end of a pipeline.
    mistral_embedding_model: str = Field(default="mistral-embed", min_length=1)
    mistral_chat_model: str = Field(default="mistral-small-latest", min_length=1)

    #: Shared by both Cloudflare adapters — one account, two products.
    cloudflare_account_id: str | None = None
    #: Vectorize only. Workers AI gets its own token below, because a token
    #: scoped to one product should not be able to spend the other.
    cloudflare_api_token: SecretStr | None = None
    cloudflare_vectorize_index: str | None = None

    # --- the edge boundary ---------------------------------------------------
    # This service is meant to sit behind a gateway that does the bot, abuse
    # and rate-limit work ([ADR 0008](docs/adr/0008-abuse-boundary-at-the-edge.md)).
    # A gateway only helps if the origin refuses traffic that did not come
    # through it, which is what the shared secret is for: the gateway sets a
    # header, this service checks it, and a request that reaches the public
    # Cloud Run URL directly never gets as far as a provider call.

    #: Shared with the edge gateway, never with a browser. Kept as a
    #: ``SecretStr`` so it cannot be printed by accident.
    edge_shared_secret: SecretStr | None = None

    #: ``None`` means "decide from the environment" — see
    #: :attr:`edge_auth_required`. Set it explicitly to check the guard locally,
    #: or to run somewhere the gateway genuinely is not in front.
    require_edge_auth: bool | None = None

    cloudflare_workers_ai_token: SecretStr | None = None
    cloudflare_workers_ai_chat_model: str = Field(default="@cf/openai/gpt-oss-120b", min_length=1)

    # --- retrieval tuning ----------------------------------------------------
    # Overrides, not defaults: ``None`` means "whatever `rag.policy` says".
    # Keeping the numbers in one place stops a second copy from drifting, and
    # keeps `core` free of any import from the layers above it.

    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    retrieval_min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)

    @property
    def service_name(self) -> str:
        return SERVICE_NAME

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def uses_development_providers(self) -> bool:
        """Whether any selected adapter is a local stand-in rather than a real service."""
        return (
            self.embedding_provider is EmbeddingProviderName.DETERMINISTIC
            or self.llm_provider is LLMProviderName.DETERMINISTIC
        )

    @property
    def edge_auth_required(self) -> bool:
        """Whether the expensive route refuses requests without the edge header.

        On by default in production and off everywhere else, because that is
        where the bill and the public URL are. Local development and the test
        suite therefore keep working unchanged, and turning it on locally is a
        single setting rather than a different code path.
        """
        if self.require_edge_auth is not None:
            return self.require_edge_auth
        return self.environment is Environment.PRODUCTION

    @property
    def use_json_logs(self) -> bool:
        """Human-readable logs locally, machine-readable logs everywhere else."""
        if self.json_logs is not None:
            return self.json_logs
        return self.environment is not Environment.LOCAL

    @field_validator(
        "mistral_api_key",
        "cloudflare_account_id",
        "cloudflare_api_token",
        "cloudflare_vectorize_index",
        "edge_shared_secret",
        "cloudflare_workers_ai_token",
        mode="before",
    )
    @classmethod
    def _strip_credential(cls, value: object) -> object:
        """Drop whitespace around a credential or account identifier.

        None of these can contain it, and a secret read from a file or a
        secret manager often ends in a newline — which an HTTP client refuses
        to put in a header, so every request fails before it is sent.
        """
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string as well as a real list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("allowed_origins", mode="after")
    @classmethod
    def _reject_origin_wildcards(cls, value: list[str]) -> list[str]:
        """A CORS wildcard is never an acceptable default — fail loudly instead."""
        if "*" in value:
            raise ValueError(
                "CORS wildcard '*' is not allowed. List the concrete origins that "
                f"may call this API in {ENV_PREFIX}ALLOWED_ORIGINS."
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so configuration is parsed and validated exactly once. Tests that
    manipulate the environment must call ``get_settings.cache_clear()``.
    """
    return Settings()
