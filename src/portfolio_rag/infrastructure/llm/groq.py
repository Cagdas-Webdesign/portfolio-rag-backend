"""Groq chat completions, for the grounding check — over the REST API directly.

    POST https://api.groq.com/openai/v1/chat/completions
    {"model": ..., "messages": [...], "max_completion_tokens": N,
     "temperature": F, "response_format": {"type": "json_object"},
     "include_reasoning": false}

The same trade as the other chat adapters: one OpenAI-compatible endpoint with
a stable shape does not justify an SDK whose types would be translated straight
back into the port's.

**What it is for.** The composition root builds it only as the *grounding*
provider (`PORTFOLIO_RAG_GROUNDING_PROVIDER=groq`); generation is never routed
here. It is a plain `LLMProvider` nonetheless: the grounding check asks a model
one question through the same port, and nothing in this module knows what the
question is.

**Reasoning never leaves Groq.** ``include_reasoning: false`` asks for the
answer only, and the one thing read from a response is
``choices[0].message.content``. A ``reasoning`` field, if one arrived anyway,
is never looked at.

**Failures, and what is deliberately not retried.** Every failure becomes an
:class:`~portfolio_rag.ports.errors.LLMProviderError` whose message is composed
here; no response body, key or prompt reaches an exception or a log.

* A timeout, a connection failure or HTTP 500/502/503/504 is retried a bounded
  number of times beneath the port — the convention the other adapters keep.
* **HTTP 429 is not retried here.** The free tier's per-minute token budget is
  small; asking again a second later spends it, and the request's deadline,
  for nothing. The pipeline ends the request as a technical error.
* **Every other 4xx — HTTP 400 included — is a refused request.** Groq's JSON
  mode may refuse a reply that is not valid JSON, but no error code for that
  is documented and none has been observed here; telling it apart from a
  malformed request would be guessing. So a 400 is never recovered: it fails
  closed, as an unknown failure must. Should a code ever be documented and
  observed, it is the one place to map onto ``MALFORMED_RESPONSE``.

A reply whose content is missing or empty is ``MALFORMED_RESPONSE`` — the same
category the Workers AI adapter uses — so the pipeline's existing, single
recovery applies exactly as it does today.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from typing import Any, Final

import httpx2 as httpx

from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import (
    GenerationRequest,
    GenerationResponse,
    ResponseFormat,
    TokenUsage,
)

PROVIDER_NAME: Final = "groq"

#: The model the grounding comparison of 2026-10-10 measured: 20 of 20 gold
#: cases, no false accept (`tests/grounding_compare/`). Configured in one place:
#: ``PORTFOLIO_RAG_GROQ_GROUNDING_MODEL``.
DEFAULT_MODEL: Final = "openai/gpt-oss-20b"
DEFAULT_BASE_URL: Final = "https://api.groq.com/openai/v1"

#: One retry for a transport failure or a 5xx. Never for 429, never for 4xx.
DEFAULT_MAX_ATTEMPTS: Final = 2
DEFAULT_RETRY_DELAY_SECONDS: Final = 1.0

#: The longest ``Retry-After`` this adapter will report. Upstream input.
MAX_RETRY_AFTER_SECONDS: Final = 300.0

_RETRYABLE_STATUS: Final = frozenset({500, 502, 503, 504})
_AUTH_STATUS: Final = frozenset({401, 403})

Sleeper = Callable[[float], Awaitable[None]]


class GroqChatProvider:
    """Asks a Groq-hosted model one question, for the grounding check."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        max_attempts: int | None = None,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("a Groq API key is required")
        if not model:
            raise ValueError("a Groq model is required")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self._model = model
        self._max_attempts = DEFAULT_MAX_ATTEMPTS if max_attempts is None else max_attempts
        self._retry_delay = retry_delay_seconds
        self._sleep: Sleeper = sleeper or asyncio.sleep
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
    def model(self) -> str:
        return self._model

    @property
    def max_attempts(self) -> int:
        """How many transport attempts one :meth:`generate` may make."""
        return self._max_attempts

    async def aclose(self) -> None:
        """Release the HTTP client, if this adapter created it."""
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        body, attempts = await self._post_with_retries(_payload(self._model, request))
        try:
            response = _parse_completion(body, fallback_model=self._model)
        except LLMProviderError as exc:
            exc.attempts = exc.transport_attempts = attempts
            raise
        return response.model_copy(
            update={"transport_attempts": attempts, "http_attempts": attempts}
        )

    async def _post_with_retries(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                error = LLMProviderError(
                    "Groq request timed out.", retryable=True, kind=ProviderFailureKind.TIMEOUT
                )
            except (httpx.LocalProtocolError, httpx.UnsupportedProtocol) as exc:
                # Nothing reached the network; the text could quote the header.
                error = LLMProviderError(
                    "Groq request could not be sent: it is invalid locally "
                    f"({type(exc).__name__}). Check the configured API key.",
                    retryable=False,
                    kind=ProviderFailureKind.UNSPECIFIED,
                )
            except httpx.RequestError as exc:
                error = LLMProviderError(
                    f"Groq could not be reached ({type(exc).__name__}).",
                    retryable=True,
                    kind=ProviderFailureKind.UNREACHABLE,
                )
            else:
                if response.status_code < 400:
                    try:
                        return _decode_json(response), attempt
                    except LLMProviderError as exc:
                        exc.attempts = exc.transport_attempts = attempt
                        raise
                error = _status_error(
                    response.status_code, retry_after_seconds=_retry_after_seconds(response)
                )

            if not error.retryable or attempt == self._max_attempts:
                error.attempts = error.transport_attempts = attempt
                raise error
            await self._sleep(self._retry_delay)

        raise LLMProviderError(  # pragma: no cover - the loop always exits above
            "Groq request could not be completed."
        )


def _payload(model: str, request: GenerationRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": message.role.value, "content": message.content} for message in request.messages
        ],
        # The answer only: reasoning stays on Groq's side.
        "include_reasoning": False,
    }
    if request.max_output_tokens is not None:
        payload["max_completion_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.response_format is ResponseFormat.JSON_OBJECT:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _status_error(
    status_code: int, *, retry_after_seconds: float | None = None
) -> LLMProviderError:
    """Map an HTTP status to a message. The response body is never read here."""
    if status_code in _AUTH_STATUS:
        return LLMProviderError(
            f"Groq rejected the credentials (HTTP {status_code}).",
            retryable=False,
            kind=ProviderFailureKind.HTTP_STATUS,
            status_code=status_code,
        )
    if status_code == 429:
        return LLMProviderError(
            "Groq rate limit reached (HTTP 429).",
            # Not retried beneath the port — see the module docstring.
            retryable=False,
            retry_after_seconds=retry_after_seconds,
            kind=ProviderFailureKind.RATE_LIMITED,
            status_code=status_code,
        )
    return LLMProviderError(
        f"Groq returned HTTP {status_code}.",
        retryable=status_code in _RETRYABLE_STATUS,
        retry_after_seconds=retry_after_seconds,
        kind=ProviderFailureKind.HTTP_STATUS,
        status_code=status_code,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """``Retry-After`` as seconds when numeric, finite and sane; otherwise nothing."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0 or seconds > MAX_RETRY_AFTER_SECONDS:
        return None
    return seconds


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMProviderError(
            "Groq returned a body that is not JSON.",
            kind=ProviderFailureKind.MALFORMED_RESPONSE,
        ) from exc
    if not isinstance(body, dict):
        raise LLMProviderError(
            "Groq returned JSON that is not an object.",
            kind=ProviderFailureKind.MALFORMED_RESPONSE,
        )
    return body


def _parse_completion(body: dict[str, Any], *, fallback_model: str) -> GenerationResponse:
    """Validate the shape before a character is trusted; read the answer text only."""
    usage = _parse_usage(body.get("usage"))

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _malformed("Groq response contains no choices.", usage=usage)
    first = choices[0]
    if not isinstance(first, dict):
        raise _malformed("Groq choice is not an object.", usage=usage)

    raw_finish = first.get("finish_reason")
    finish_reason = raw_finish if isinstance(raw_finish, str) else None

    message = first.get("message")
    if not isinstance(message, dict):
        raise _malformed(
            "Groq choice carries no message.", finish_reason=finish_reason, usage=usage
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _malformed(
            "Groq message carries no text content.", finish_reason=finish_reason, usage=usage
        )

    return GenerationResponse(
        text=content, model=fallback_model, finish_reason=finish_reason, usage=usage
    )


def _malformed(
    message: str, *, finish_reason: str | None = None, usage: TokenUsage | None = None
) -> LLMProviderError:
    return LLMProviderError(
        message,
        kind=ProviderFailureKind.MALFORMED_RESPONSE,
        finish_reason=finish_reason,
        usage=usage,
    )


def _parse_usage(raw: object) -> TokenUsage | None:
    """Token usage when present and sane. Diagnostics only."""
    if not isinstance(raw, dict):
        return None
    prompt_tokens = raw.get("prompt_tokens")
    completion_tokens = raw.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    if isinstance(prompt_tokens, bool) or isinstance(completion_tokens, bool):
        return None
    if prompt_tokens < 0 or completion_tokens < 0:
        return None
    return TokenUsage(input_tokens=prompt_tokens, output_tokens=completion_tokens)
