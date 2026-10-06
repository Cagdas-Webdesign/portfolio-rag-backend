"""Mistral chat completions, over the HTTP API directly.

Same trade as the embedding adapter, for the same reasons: one endpoint, one
request shape, and every SDK response model would be translated straight back
into the port's types. The official SDK becomes the better answer when this
needs streaming or several Mistral APIs — not before.

**What crosses this boundary.** Outbound: the messages the query side composed
— grounding instructions, the selected public passages under their labels, and
the question. Nothing else. Not the corpus, not internal documents, not
fingerprints, not source paths, not ids, not settings. Inbound: text. Every
provider failure — auth, rate limit, timeout, network, 5xx, malformed body,
missing content — becomes
:class:`~portfolio_rag.ports.errors.LLMProviderError`, whose message is
composed here from a status code rather than echoed from upstream, so no
provider payload can reach a log or a client. A ``Retry-After`` header is read
off the response and carried on the error as a number; this adapter's own
retry budget is unchanged by it, because a request with a client waiting
cannot sit out a rate-limit window. Whoever can wait that long decides.

**The adapter does not interpret the answer.** It returns the text the model
produced. Whether that text satisfies the grounded answer contract is decided
in :mod:`portfolio_rag.rag.generation`, where the prompt that asked for it
lives.
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

PROVIDER_NAME: Final = "mistral"

#: Small, fast, and available on Mistral's free tier — which is the constraint
#: this project actually has. Configurable, and configured in exactly one
#: place: ``PORTFOLIO_RAG_MISTRAL_CHAT_MODEL``.
DEFAULT_MODEL: Final = "mistral-small-latest"
DEFAULT_BASE_URL: Final = "https://api.mistral.ai"

#: Bounded retries, matching the embedding adapter. Not a resilience framework:
#: no backoff library, no circuit breaker, no queue.
DEFAULT_MAX_ATTEMPTS: Final = 3
DEFAULT_RETRY_DELAY_SECONDS: Final = 1.0

#: The longest ``Retry-After`` this adapter will repeat to a caller. A header
#: asking for more than this is treated as absent: it is upstream input, and
#: "wait two hours" is not a number any caller here should act on.
MAX_RETRY_AFTER_SECONDS: Final = 300.0

_RETRYABLE_STATUS: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_AUTH_STATUS: Final = frozenset({401, 403})

#: Injected so tests can exercise retry behaviour without sleeping.
Sleeper = Callable[[float], Awaitable[None]]


class MistralChatProvider:
    """Generates completions with Mistral's chat endpoint."""

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
            raise ValueError("a Mistral API key is required")
        if not model:
            raise ValueError("a Mistral chat model is required")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self._model = model
        #: ``None`` means the adapter's own default, so the number lives in
        #: exactly one place and a caller that has no opinion cannot copy it.
        self._max_attempts = DEFAULT_MAX_ATTEMPTS if max_attempts is None else max_attempts
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
    def model(self) -> str:
        return self._model

    @property
    def max_attempts(self) -> int:
        """How many transport attempts one :meth:`generate` may make.

        Readable because it is a wiring decision, not an implementation
        detail: a caller that owns its own retry budget — the evaluation
        pacer — builds this adapter with ``1`` so that the two budgets cannot
        multiply, and that has to be checkable from outside.
        """
        return self._max_attempts

    async def aclose(self) -> None:
        """Release the HTTP client, if this adapter created it."""
        if self._owns_client:
            await self._client.aclose()

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Send one completion request and validate what comes back."""
        body, attempts = await self._post_with_retries(
            "/v1/chat/completions", _payload(self._model, request)
        )
        try:
            response = _parse_completion(body, fallback_model=self._model)
        except LLMProviderError as exc:
            exc.attempts = exc.transport_attempts = attempts
            raise
        # How many requests this response took — telemetry, never a decision.
        return response.model_copy(
            update={"transport_attempts": attempts, "http_attempts": attempts}
        )

    async def _post_with_retries(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Post once, and retry a couple of times if the failure looks transient.

        A 401 will not become a 200 by asking again; retrying it would just be
        a slower way to fail, and against a rate-limited endpoint a louder one.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(path, json=payload)
            except httpx.TimeoutException:
                error = LLMProviderError(
                    f"Mistral generation timed out after {self._timeout_seconds}s.",
                    retryable=True,
                    kind=ProviderFailureKind.TIMEOUT,
                )
            except httpx.RequestError:
                # The exception text can carry the full URL; the message is
                # composed here so nothing unexpected reaches a log.
                error = LLMProviderError(
                    "Mistral could not be reached for generation.",
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
                    response.status_code,
                    retry_after_seconds=_retry_after_seconds(response),
                )

            if not error.retryable or attempt == self._max_attempts:
                error.attempts = error.transport_attempts = attempt
                raise error
            await self._sleep(self._retry_delay)

        raise LLMProviderError(  # pragma: no cover - the loop always exits above
            "Mistral generation could not be completed."
        )


def _payload(model: str, request: GenerationRequest) -> dict[str, Any]:
    """Map the port's request onto Mistral's wire format."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": message.role.value, "content": message.content} for message in request.messages
        ],
    }
    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.response_format is ResponseFormat.JSON_OBJECT:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _status_error(
    status_code: int, *, retry_after_seconds: float | None = None
) -> LLMProviderError:
    """Map an HTTP status to a message. The response body is never included."""
    if status_code in _AUTH_STATUS:
        return LLMProviderError(
            f"Mistral rejected the credentials (HTTP {status_code}).",
            retryable=False,
            kind=ProviderFailureKind.HTTP_STATUS,
            status_code=status_code,
        )
    if status_code == 429:
        return LLMProviderError(
            "Mistral rate limit reached during generation.",
            retryable=True,
            retry_after_seconds=retry_after_seconds,
            kind=ProviderFailureKind.RATE_LIMITED,
            status_code=status_code,
        )
    return LLMProviderError(
        f"Mistral returned HTTP {status_code} during generation.",
        retryable=status_code in _RETRYABLE_STATUS,
        retry_after_seconds=retry_after_seconds,
        kind=ProviderFailureKind.HTTP_STATUS,
        status_code=status_code,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read ``Retry-After`` as a number of seconds, or report nothing.

    A value counts only when it is numeric, finite and not negative. Only the
    delta-seconds form is understood. The HTTP-date form is legal and
    is not parsed here: it would need a clock comparison against a header
    nobody in this project has seen, and a caller that waits a default instead
    is already correct. Anything unparsable, negative or absurd is reported as
    absent rather than trusted — a header is upstream input, and a caller
    should not be talked into sleeping for an hour by one.
    """
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw.strip())
    except ValueError:
        return None
    # `float()` accepts "nan" and "inf" as readily as "12". A NaN fails every
    # comparison below rather than being caught by one, so it has to be ruled
    # out first; the infinities would trip the ceiling, but relying on that is
    # incidental rather than intended.
    if not math.isfinite(seconds) or seconds < 0 or seconds > MAX_RETRY_AFTER_SECONDS:
        return None
    return seconds


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMProviderError(
            "Mistral returned a body that is not JSON.", kind=ProviderFailureKind.MALFORMED_RESPONSE
        ) from exc
    if not isinstance(body, dict):
        raise LLMProviderError(
            "Mistral returned JSON that is not an object.",
            kind=ProviderFailureKind.MALFORMED_RESPONSE,
        )
    return body


def _parse_completion(body: dict[str, Any], *, fallback_model: str) -> GenerationResponse:
    """Validate the response shape before a single character is trusted.

    Usage and, once a choice is readable, the finish reason are taken first,
    so that a response that is not a completion still reports them.
    """
    usage = _parse_usage(body.get("usage"))

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _malformed("Mistral response contains no choices.", usage=usage)

    first = choices[0]
    if not isinstance(first, dict):
        raise _malformed("Mistral choice is not an object.", usage=usage)

    raw_finish = first.get("finish_reason")
    finish_reason = raw_finish if isinstance(raw_finish, str) else None

    message = first.get("message")
    if not isinstance(message, dict):
        raise _malformed(
            "Mistral choice carries no message.", finish_reason=finish_reason, usage=usage
        )

    content = message.get("content")
    if not isinstance(content, str):
        raise _malformed(
            "Mistral message carries no text content.", finish_reason=finish_reason, usage=usage
        )

    model = body.get("model")

    return GenerationResponse(
        text=content,
        model=model if isinstance(model, str) and model else fallback_model,
        finish_reason=finish_reason,
        usage=usage,
    )


def _malformed(
    message: str, *, finish_reason: str | None = None, usage: TokenUsage | None = None
) -> LLMProviderError:
    """A response that is not a completion, with whatever metadata it did carry."""
    return LLMProviderError(
        message,
        kind=ProviderFailureKind.MALFORMED_RESPONSE,
        finish_reason=finish_reason,
        usage=usage,
    )


def _parse_usage(raw: object) -> TokenUsage | None:
    """Report token usage when it is present and sane, otherwise nothing.

    Diagnostics only: no decision in this system depends on these numbers, so a
    provider that omits or mangles them costs an observability field and never
    an answer.
    """
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
