"""A reusable behavioural contract for :class:`LLMProvider` implementations.

Written against the port, never against an implementation's internals: any
adapter can be dropped in by subclassing the suite and supplying a provider.

Deliberately short, because the port is. What a caller is entitled to assume is
that a request is accepted, that some text comes back, that the model is named,
and that a failure arrives as :class:`LLMProviderError` rather than as an HTTP
library's exception. Everything about *what the text says* is the query side's
contract, not the port's.
"""

from __future__ import annotations

import pytest

from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import (
    GenerationRequest,
    LLMProvider,
    MessageRole,
    PromptMessage,
    ResponseFormat,
)
from tests.support import run

REQUEST = GenerationRequest(
    messages=(
        PromptMessage(role=MessageRole.SYSTEM, content="Answer only from the knowledge given."),
        PromptMessage(role=MessageRole.USER, content="KNOWLEDGE\n[SOURCE S1]\n…\n\nQUESTION\nWhy?"),
    ),
    max_output_tokens=256,
    temperature=0.2,
    response_format=ResponseFormat.JSON_OBJECT,
)


class LLMProviderContract:
    """Subclass and provide :meth:`make_provider`."""

    def make_provider(self) -> LLMProvider:  # pragma: no cover - overridden
        raise NotImplementedError

    def make_failing_provider(self) -> LLMProvider | None:
        """Return a provider whose next call fails, or ``None`` to skip that test."""
        return None

    @pytest.fixture
    def provider(self) -> LLMProvider:
        return self.make_provider()

    def test_a_request_is_accepted_and_answered(self, provider: LLMProvider):
        response = run(provider.generate(REQUEST))

        assert isinstance(response.text, str)
        assert response.text != ""

    def test_the_provider_names_the_model_it_generates_with(self, provider: LLMProvider):
        response = run(provider.generate(REQUEST))

        assert provider.model
        assert response.model

    def test_the_answer_is_returned_uninterpreted(self, provider: LLMProvider):
        """An adapter that parsed the answer would be making a decision it does not own."""
        response = run(provider.generate(REQUEST))

        assert not hasattr(response, "sources")
        assert not hasattr(response, "citations")

    def test_a_failure_arrives_as_a_port_error(self):
        failing = self.make_failing_provider()
        if failing is None:
            pytest.skip("this adapter has no failure mode to exercise")

        with pytest.raises(LLMProviderError):
            run(failing.generate(REQUEST))
