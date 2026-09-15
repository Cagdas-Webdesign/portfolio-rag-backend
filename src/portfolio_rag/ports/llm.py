"""Port for text generation providers.

Deliberately narrow: one non-streaming call in, one structured answer out.
Streaming and tool calling are absent because no caller consumes them.

**The port knows nothing about grounding.** It carries messages and returns
text. What those messages say, what shape the answer has to be in, and which
sources an answer is allowed to reference are decisions of the query side
(:mod:`portfolio_rag.rag`) — the prompt that asks for a contract and the parser
that enforces it belong together, and neither belongs in a vendor adapter.
:attr:`GenerationRequest.response_format` is the one exception: asking for JSON
is a transport capability every mainstream provider exposes, and it has to
cross the boundary to be used at all.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Role of a message in a generation request."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ResponseFormat(StrEnum):
    """How the provider should shape its reply.

    ``JSON_OBJECT`` asks for a single JSON document. Providers enforce this to
    differing degrees, so a caller still has to parse defensively — it removes
    most of the failure modes, not all of them.
    """

    TEXT = "text"
    JSON_OBJECT = "json_object"


class PromptMessage(BaseModel):
    """One message handed to the provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1)


class TokenUsage(BaseModel):
    """Token accounting, when the provider reports it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class GenerationRequest(BaseModel):
    """Everything a provider needs to produce one answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[PromptMessage, ...] = Field(min_length=1)
    max_output_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    response_format: ResponseFormat = ResponseFormat.TEXT


class GenerationResponse(BaseModel):
    """The provider's answer, normalized across vendors.

    :attr:`text` is exactly what the model produced, unparsed. A provider that
    interprets it has already made a decision that is not its to make.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    model: str = Field(description="Provider-specific model identifier that produced the text.")
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class LLMProvider(Protocol):
    """Turns a structured generation request into a structured response."""

    @property
    def model(self) -> str:
        """Model identifier this provider generates with. Logged, never guessed."""
        ...

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Produce one completion, using only documented bounded transport retries."""
        ...
