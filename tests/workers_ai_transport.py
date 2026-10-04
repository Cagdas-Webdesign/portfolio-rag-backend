"""The real Workers AI adapter over an in-process transport, and the reply
shapes production produced.

Tests that need ``usage`` and ``finish_reason`` read the way production reads
them go through the real adapter; this serves it scripted bodies instead of
the network. No request leaves the process, and the credential is a
placeholder no service would accept.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx

from portfolio_rag.infrastructure.llm import WorkersAIChatProvider
from portfolio_rag.rag.policy import DEFAULT_OUTPUT_RESERVE_TOKENS

#: The output limit both provider calls are made with.
LIMIT = DEFAULT_OUTPUT_RESERVE_TOKENS

#: The recorded signature of every ``finish_reason=length`` failure: the limit
#: spent, one character per token, not JSON.
DEGENERATE = "{" + "\n" * (LIMIT - 1)
VALID = json.dumps(
    {"answer": "The service uses FastAPI [S1].", "sources": ["S1"], "support": "stated"}
)
SUPPORTED = json.dumps({"verdict": "supported"})

Served = list[dict[str, Any]]


async def _no_sleep(_: float) -> None:
    return None


def workers_ai(*responses: Any) -> tuple[WorkersAIChatProvider, Served]:
    """The real adapter over an in-process transport.

    Each response is a completion body, an ``httpx.Response`` or an exception
    to raise, served in order (the last one repeats). Every request body is
    recorded so a test can count generations and checks separately.
    """
    seen: Served = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        item = responses[min(len(seen) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json=item)

    provider = WorkersAIChatProvider(
        account_id="test-account-id",
        api_token="test-token-not-a-real-credential",  # noqa: S106 - a fixture value
        client=httpx.AsyncClient(
            base_url="https://api.cloudflare.test/client/v4",
            transport=httpx.MockTransport(handler),
        ),
        sleeper=_no_sleep,
    )
    return provider, seen
