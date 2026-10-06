"""Mistral embeddings, over the HTTP API directly.

**Why not the official SDK.** This adapter uses one endpoint with one request
shape. The SDK would add a dependency tree and its own response models, every
one of which would be translated straight back into the port's types — paying
for an abstraction to immediately undo it. A ~150-line adapter against a stable
JSON endpoint is easier to read, to test without a network, and to keep out of
the core. If the adapter ever needs several Mistral APIs, that trade may flip
and the SDK may become the better answer.

**Failure handling.** Everything that can go wrong on the way to a vector —
authentication, rate limits, timeouts, network errors, 5xx, malformed bodies,
the wrong number of embeddings, the wrong dimensionality — is turned into
:class:`~portfolio_rag.ports.errors.EmbeddingProviderError`. Nothing from
``httpx`` and nothing from Mistral's JSON crosses this boundary.

**What is sent.** Only the embedding text composed by
:mod:`portfolio_rag.ingestion.embedding`: document title, heading path, chunk
content. No visibility, no trust level, no source path, no fingerprints, no
ids. That boundary is deliberate and documented in ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

import httpx2 as httpx

from portfolio_rag.domain.embedding import EmbeddingSpec, EmbeddingVector
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION
from portfolio_rag.ports.embeddings import EmbeddingInput, EmbeddingResult
from portfolio_rag.ports.errors import EmbeddingProviderError, ProviderFailureKind

PROVIDER_NAME: Final = "mistral"

#: Current general-purpose embedding model, 1024 dimensions.
DEFAULT_MODEL: Final = "mistral-embed"
DEFAULT_DIMENSIONS: Final = 1024
DEFAULT_BASE_URL: Final = "https://api.mistral.ai"

#: How many inputs go into one request. A provider-side limit, so it lives with
#: the provider: the application hands over the whole sequence and never counts.
DEFAULT_BATCH_SIZE: Final = 64

#: Transient failures are retried a couple of times and then given up on.
#: Deliberately not a retry framework — no backoff library, no circuit breaker,
#: no queue. Two extra attempts is the difference between surviving a blip and
#: hammering a rate-limited endpoint.
DEFAULT_MAX_ATTEMPTS: Final = 3
DEFAULT_RETRY_DELAY_SECONDS: Final = 1.0

_RETRYABLE_STATUS: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_AUTH_STATUS: Final = frozenset({401, 403})

#: Injected so tests can exercise retry behaviour without sleeping.
Sleeper = Callable[[float], Awaitable[None]]


class MistralEmbeddingProvider:
    """Embeds text with Mistral's embeddings endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("a Mistral API key is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self._spec = EmbeddingSpec(
            provider=PROVIDER_NAME,
            model=model,
            dimensions=dimensions,
            representation_version=EMBEDDING_REPRESENTATION_VERSION,
        )
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay_seconds
        self._sleep: Sleeper = sleeper or asyncio.sleep
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec

    async def aclose(self) -> None:
        """Release the HTTP client, if this adapter created it."""
        if self._owns_client:
            await self._client.aclose()

    async def embed(self, inputs: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        """Embed every input, splitting into provider-sized batches."""
        results: list[EmbeddingResult] = []
        for start in range(0, len(inputs), self._batch_size):
            batch = list(inputs[start : start + self._batch_size])
            results.extend(await self._embed_batch(batch))
        return results

    async def _embed_batch(self, batch: Sequence[EmbeddingInput]) -> list[EmbeddingResult]:
        payload = {"model": self._spec.model, "input": [item.text for item in batch]}
        body = await self._post_with_retries("/v1/embeddings", payload, size=len(batch))
        vectors = _parse_embeddings(body, expected=len(batch), dimensions=self._spec.dimensions)
        # Positional mapping is safe only because the response was validated to
        # carry exactly one entry per input, in the order Mistral reports.
        return [
            EmbeddingResult(id=item.id, vector=vector)
            for item, vector in zip(batch, vectors, strict=True)
        ]

    async def _post_with_retries(
        self, path: str, payload: dict[str, Any], *, size: int
    ) -> dict[str, Any]:
        """Post once, and retry a couple of times if the failure looks transient.

        Only 429, a handful of 5xx and network-level errors are retried. A 401
        will not become a 200 by asking again, and retrying it would just be a
        slower way to fail.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(path, json=payload)
            except httpx.TimeoutException:
                error = EmbeddingProviderError(
                    f"Mistral request timed out after {self._timeout_seconds}s "
                    f"while embedding {size} input(s).",
                    retryable=True,
                    kind=ProviderFailureKind.TIMEOUT,
                )
            except httpx.RequestError:
                # The exception text can carry the full URL; the message is
                # written here instead so nothing unexpected reaches a log.
                error = EmbeddingProviderError(
                    f"Mistral could not be reached while embedding {size} input(s).",
                    retryable=True,
                    kind=ProviderFailureKind.UNREACHABLE,
                )
            else:
                if response.status_code < 400:
                    return _decode_json(response)
                error = _status_error(response.status_code, size)

            if not error.retryable or attempt == self._max_attempts:
                raise error
            await self._sleep(self._retry_delay)

        raise EmbeddingProviderError(  # pragma: no cover - the loop always exits above
            "Mistral request could not be completed."
        )


def _status_error(status_code: int, size: int) -> EmbeddingProviderError:
    """Map an HTTP status to a message. The response body is never included."""
    if status_code in _AUTH_STATUS:
        return EmbeddingProviderError(
            f"Mistral rejected the credentials (HTTP {status_code}).",
            retryable=False,
            kind=ProviderFailureKind.HTTP_STATUS,
            status_code=status_code,
        )
    if status_code == 429:
        return EmbeddingProviderError(
            f"Mistral rate limit reached while embedding {size} input(s).",
            retryable=True,
            kind=ProviderFailureKind.RATE_LIMITED,
            status_code=status_code,
        )
    return EmbeddingProviderError(
        f"Mistral returned HTTP {status_code} while embedding {size} input(s).",
        retryable=status_code in _RETRYABLE_STATUS,
        kind=ProviderFailureKind.HTTP_STATUS,
        status_code=status_code,
    )


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise EmbeddingProviderError("Mistral returned a body that is not JSON.") from exc
    if not isinstance(body, dict):
        raise EmbeddingProviderError("Mistral returned JSON that is not an object.")
    return body


def _parse_embeddings(
    body: dict[str, Any], *, expected: int, dimensions: int
) -> list[EmbeddingVector]:
    """Validate the response shape before a single number is trusted."""
    data = body.get("data")
    if not isinstance(data, list):
        raise EmbeddingProviderError("Mistral response is missing its `data` array.")
    if len(data) != expected:
        raise EmbeddingProviderError(
            f"Mistral returned {len(data)} embeddings for {expected} inputs."
        )

    vectors: list[EmbeddingVector] = []
    for position, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise EmbeddingProviderError(f"Mistral entry {position} is not an object.")
        raw = entry.get("embedding")
        if not isinstance(raw, list) or not raw:
            raise EmbeddingProviderError(f"Mistral entry {position} has no embedding.")
        if len(raw) != dimensions:
            raise EmbeddingProviderError(
                f"Mistral entry {position} has {len(raw)} dimensions, expected {dimensions}."
            )
        vector = _as_finite_floats(raw, position)
        vectors.append(vector)
    return vectors


def _as_finite_floats(raw: list[Any], position: int) -> EmbeddingVector:
    values: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EmbeddingProviderError(f"Mistral entry {position} contains a non-numeric value.")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingProviderError(f"Mistral entry {position} contains a non-finite value.")
        values.append(number)
    return tuple(values)
