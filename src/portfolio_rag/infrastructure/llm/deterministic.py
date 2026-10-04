"""A local generation provider for development. **Not a language model.**

It generates nothing. It reads the source labels out of the prompt it was
given and returns a fixed statement saying exactly that, citing those labels.
Its purpose is to make the query side runnable end to end — retrieval, context
building, the answer contract, citation validation, the HTTP response — with no
API key, no network and no cost, which is the same reason
:class:`~portfolio_rag.infrastructure.embedding.DeterministicEmbeddingProvider`
exists.

What it deliberately does **not** do is imitate an assistant. It writes no
prose about the passages, answers no questions and produces no sentence that
could be mistaken for a real answer in a screenshot. A stub that sounded
plausible would be worse than useless: every test of "does the pipeline work"
would silently become a test of "does the stub sound convincing", and the day
it reached a public deployment nobody would notice.

That is also why the composition root refuses to build this adapter in a
production environment.
"""

from __future__ import annotations

import json
import re
from typing import Final

from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse, MessageRole

PROVIDER_NAME: Final = "deterministic"
MODEL_NAME: Final = "context-echo-v1"

#: Labels as the context builder writes them.
_SOURCE_LABEL: Final = re.compile(r"\[SOURCE (S\d+)\]")

_STUB_ANSWER: Final = (
    "This is a development stub, not a generated answer. The configured "
    "generation provider is `deterministic`, which does not produce language. "
    "It reports the knowledge sources that were retrieved for this question so "
    "that the pipeline can be inspected end to end."
)

_NO_SOURCES_ANSWER: Final = (
    "This is a development stub, not a generated answer. No knowledge sources "
    "were present in the prompt."
)


class DeterministicLLMProvider:
    """Echoes the labels it was handed. Development and tests only."""

    @property
    def model(self) -> str:
        return MODEL_NAME

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Return the answer contract, filled in from the prompt itself."""
        labels = _labels_in(request)
        payload = {
            "answer": _STUB_ANSWER if labels else _NO_SOURCES_ANSWER,
            "sources": list(labels),
            # The stub's statement is about the labels it was handed, which is
            # exactly what the prompt contains: stated when there are any.
            "support": "stated" if labels else "none",
            # The pipeline asks a second time whether the cited passages carry
            # the answer. The stub cannot tell that request from the first, so
            # one reply satisfies both contracts: each reader takes its field
            # and ignores the other.
            "verdict": "supported" if labels else "not_supported",
        }
        return GenerationResponse(
            text=json.dumps(payload, ensure_ascii=False),
            model=MODEL_NAME,
            finish_reason="stop",
        )


def _labels_in(request: GenerationRequest) -> tuple[str, ...]:
    """Collect source labels from the user message, in order, without repeats.

    Only the user message: instructions are not knowledge, and a label found in
    the system role would mean the prompt builder had put a passage somewhere
    it must never be.
    """
    seen: list[str] = []
    for message in request.messages:
        if message.role is not MessageRole.USER:
            continue
        for label in _SOURCE_LABEL.findall(message.content):
            if label not in seen:
                seen.append(label)
    return tuple(seen)
