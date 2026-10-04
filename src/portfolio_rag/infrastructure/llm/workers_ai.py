"""Cloudflare Workers AI chat completions, over the REST API directly.

Same trade as the Mistral chat adapter and the Vectorize store: one endpoint
with a stable shape does not justify an SDK whose models would be translated
straight back into the port's types.

    POST /accounts/{account_id}/ai/run/{model}
    {"messages": [...], "max_tokens": N, "temperature": F}

**Which response shape this expects, and why.** Workers AI answers in the shape
its *input* asked for: a ``prompt`` string comes back as ``result.response``, a
Responses-API input comes back as ``result.output``, and ``messages`` — what
the port carries, and therefore what this sends — comes back as
``result.choices[].message``. One request shape, one response shape, nothing
guessed at runtime.

**Reasoning never leaves this module, because it is never read.** Models such
as ``@cf/openai/gpt-oss-120b`` think out loud, and Workers AI can return that
thinking alongside the answer — in a ``reasoning`` field, in extra content
parts, in provider metadata. This adapter reads exactly one thing: the final
assistant text on the first choice. Everything else on the body, the choice and
the message is ignored rather than filtered, which is the difference between a
leak that cannot happen and one that a future field name could reintroduce.

**What crosses this boundary.** Outbound: the messages the query side composed
— grounding instructions, the selected public passages under their labels, and
the question. Nothing else. Inbound: text. Every provider failure — auth, rate
limit, timeout, network, 5xx, malformed body, missing content — becomes
:class:`~portfolio_rag.ports.errors.LLMProviderError`, whose message is
composed here from a status code rather than echoed from upstream, so no
provider payload and no credential can reach a log or a client.

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

PROVIDER_NAME: Final = "cloudflare-workers-ai"

#: OpenAI's open-weight 120B model, served on Workers AI. Configurable, and
#: configured in exactly one place:
#: ``PORTFOLIO_RAG_CLOUDFLARE_WORKERS_AI_CHAT_MODEL``.
DEFAULT_MODEL: Final = "@cf/openai/gpt-oss-120b"
DEFAULT_BASE_URL: Final = "https://api.cloudflare.com/client/v4"

#: Bounded retries, matching the Mistral adapters rather than inventing a
#: second convention. Not a resilience framework: no backoff library, no
#: circuit breaker, no queue. Three fast attempts is what a caller with a
#: browser open can afford; waiting out a rate-limit window is somebody else's
#: decision (see :mod:`portfolio_rag.evaluation.pacing`).
DEFAULT_MAX_ATTEMPTS: Final = 3
DEFAULT_RETRY_DELAY_SECONDS: Final = 1.0

#: The longest ``Retry-After`` this adapter will repeat to a caller. A header
#: asking for more than this is treated as absent: it is upstream input.
MAX_RETRY_AFTER_SECONDS: Final = 300.0

_RETRYABLE_STATUS: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_AUTH_STATUS: Final = frozenset({401, 403})

#: Content part types that carry answer text. A part of any other type —
#: ``reasoning`` above all — is dropped, not rendered.
_TEXT_PART_TYPES: Final = frozenset({"text", "output_text"})

#: Injected so tests can exercise retry behaviour without sleeping.
Sleeper = Callable[[float], Awaitable[None]]


class WorkersAIChatProvider:
    """Generates completions with a Workers AI chat model."""

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        max_attempts: int | None = None,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        client: httpx.AsyncClient | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        if not account_id:
            raise ValueError("a Cloudflare account id is required")
        if not api_token:
            raise ValueError("a Workers AI API token is required")
        if not model:
            raise ValueError("a Workers AI model is required")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

        self._model = model
        #: ``None`` means the adapter's own default, so the number lives in
        #: exactly one place and a caller that has no opinion cannot copy it.
        self._max_attempts = DEFAULT_MAX_ATTEMPTS if max_attempts is None else max_attempts
        self._retry_delay = retry_delay_seconds
        self._sleep: Sleeper = sleeper or asyncio.sleep
        self._timeout_seconds = timeout_seconds
        # The model id contains `@` and `/` and is a path segment, not a query
        # parameter. It is configuration, not user input: nothing reachable
        # from a request can put a value here.
        self._path = f"/accounts/{account_id}/ai/run/{model}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {api_token}",
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
        body = await self._post_with_retries(_payload(request))
        return _parse_completion(body, fallback_model=self._model)

    async def _post_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post once, and retry a couple of times if the failure looks transient.

        A 401 will not become a 200 by asking again; retrying it would just be
        a slower way to fail, and against a rate-limited endpoint a louder one
        that spends quota to do it.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.post(self._path, json=payload)
            except httpx.TimeoutException:
                error = LLMProviderError(
                    f"Workers AI generation timed out after {self._timeout_seconds}s.",
                    retryable=True,
                    kind=ProviderFailureKind.TIMEOUT,
                )
            except (httpx.LocalProtocolError, httpx.UnsupportedProtocol) as exc:
                # The request could not be built or sent from here — an
                # illegal header value (a credential with a stray newline or
                # space), a URL scheme the client cannot speak. Nothing reached
                # the network, so this is not "unreachable", and asking again
                # cannot change it. The exception text can quote the header —
                # that is, the token — so only its class is named.
                error = LLMProviderError(
                    "Workers AI request could not be sent: it is invalid locally "
                    f"({type(exc).__name__}). Check the configured account id and token.",
                    retryable=False,
                    kind=ProviderFailureKind.UNSPECIFIED,
                )
            except httpx.RequestError as exc:
                # The exception text can carry the full URL; the message is
                # composed here so nothing unexpected reaches a log.
                error = LLMProviderError(
                    f"Workers AI could not be reached for generation ({type(exc).__name__}).",
                    retryable=True,
                    kind=ProviderFailureKind.UNREACHABLE,
                )
            else:
                if response.status_code < 400:
                    return _decode_json(response)
                error = _status_error(
                    response.status_code,
                    retry_after_seconds=_retry_after_seconds(response),
                )

            if not error.retryable or attempt == self._max_attempts:
                error.attempts = attempt
                raise error
            await self._sleep(self._retry_delay)

        raise LLMProviderError(  # pragma: no cover - the loop always exits above
            "Workers AI generation could not be completed."
        )


def _payload(request: GenerationRequest) -> dict[str, Any]:
    """Map the port's request onto Workers AI's wire format.

    The model is in the URL, not the body — which is the one shape difference
    from an OpenAI-style chat endpoint.
    """
    payload: dict[str, Any] = {
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
            f"Workers AI rejected the credentials (HTTP {status_code}).",
            retryable=False,
            kind=ProviderFailureKind.HTTP_STATUS,
            status_code=status_code,
        )
    if status_code == 429:
        return LLMProviderError(
            "Workers AI rate limit reached during generation.",
            retryable=True,
            retry_after_seconds=retry_after_seconds,
            kind=ProviderFailureKind.RATE_LIMITED,
            status_code=status_code,
        )
    return LLMProviderError(
        f"Workers AI returned HTTP {status_code} during generation.",
        retryable=status_code in _RETRYABLE_STATUS,
        retry_after_seconds=retry_after_seconds,
        kind=ProviderFailureKind.HTTP_STATUS,
        status_code=status_code,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read ``Retry-After`` as a number of seconds, or report nothing.

    A value counts only when it is numeric, finite and not negative. Only the
    delta-seconds form is understood; anything unparsable, negative or
    absurd is reported as absent rather than trusted. Reported, not obeyed —
    this adapter's own delay is unchanged by it, and the batch runner that can
    afford to wait is the one that reads it.
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
            "Workers AI returned a body that is not JSON.",
            kind=ProviderFailureKind.MALFORMED_RESPONSE,
        ) from exc
    if not isinstance(body, dict):
        raise LLMProviderError(
            "Workers AI returned JSON that is not an object.",
            kind=ProviderFailureKind.MALFORMED_RESPONSE,
        )
    return body


def _parse_completion(body: dict[str, Any], *, fallback_model: str) -> GenerationResponse:
    """Validate the response shape before a single character is trusted.

    Cloudflare wraps every Workers AI result in its standard envelope, so the
    completion is under ``result``. A body whose ``success`` is explicitly
    false is a failure even when it arrived with HTTP 200.
    """
    if body.get("success") is False:
        raise LLMProviderError(
            "Workers AI reported a failed generation.", kind=ProviderFailureKind.MALFORMED_RESPONSE
        )

    result = body.get("result")
    if not isinstance(result, dict):
        raise LLMProviderError(
            "Workers AI response carries no result object.",
            kind=ProviderFailureKind.MALFORMED_RESPONSE,
        )

    # Read before the shape is judged, so that a response that turns out not
    # to be a completion still says what it cost and, once known, why it ended.
    usage = _parse_usage(result.get("usage"))

    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise _malformed("Workers AI response contains no choices.", usage=usage)

    first = choices[0]
    if not isinstance(first, dict):
        raise _malformed("Workers AI choice is not an object.", usage=usage)

    raw_finish = first.get("finish_reason")
    finish_reason = raw_finish if isinstance(raw_finish, str) else None

    message = first.get("message")
    if not isinstance(message, dict):
        raise _malformed(
            "Workers AI choice carries no message.", finish_reason=finish_reason, usage=usage
        )

    text = _answer_text(message)
    if text is None:
        raise _malformed(
            "Workers AI message carries no text content.",
            finish_reason=finish_reason,
            usage=usage,
        )

    return GenerationResponse(
        text=text,
        model=fallback_model,
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


def _answer_text(message: dict[str, Any]) -> str | None:
    """Take the final assistant text, and nothing that sits next to it.

    ``content`` is normally the string the model finished with. Reasoning
    models can instead split it into typed parts, and a thinking model's parts
    are not all meant for a reader — so only the text-carrying types are kept
    and every other part, ``reasoning`` included, is dropped.

    Sibling fields such as ``reasoning`` or ``reasoning_content`` are never
    looked at. That is deliberate: a filter has to be kept up to date with
    whatever a provider invents next, and not reading a field cannot fall
    behind.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") in _TEXT_PART_TYPES
            and isinstance(part.get("text"), str)
        ]
        if parts:
            return "".join(parts)
    return None


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
