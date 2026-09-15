"""The development stub: what it does, and what it must never look like."""

from __future__ import annotations

import json

from portfolio_rag.infrastructure.llm import DeterministicLLMProvider
from portfolio_rag.ports.llm import (
    GenerationRequest,
    LLMProvider,
    MessageRole,
    PromptMessage,
)
from portfolio_rag.rag.generation import parse_generation
from tests.contracts.llm_provider import REQUEST, LLMProviderContract
from tests.doubles import FailingLLMProvider
from tests.support import run


class TestDeterministicLLMProviderContract(LLMProviderContract):
    def make_provider(self) -> LLMProvider:
        return DeterministicLLMProvider()

    def make_failing_provider(self) -> LLMProvider:
        return FailingLLMProvider()


def request_with(user_content: str) -> GenerationRequest:
    return GenerationRequest(
        messages=(
            PromptMessage(role=MessageRole.SYSTEM, content="instructions"),
            PromptMessage(role=MessageRole.USER, content=user_content),
        )
    )


def test_it_answers_in_the_shape_the_prompt_asked_for():
    response = run(DeterministicLLMProvider().generate(REQUEST))

    draft = parse_generation(response.text)
    assert draft.answer
    assert draft.source_labels == ("S1",)


def test_it_cites_every_label_it_was_given_in_order():
    response = run(
        DeterministicLLMProvider().generate(
            request_with("[SOURCE S1]\nfirst\n\n[SOURCE S2]\nsecond\n\n[SOURCE S3]\nthird")
        )
    )

    assert json.loads(response.text)["sources"] == ["S1", "S2", "S3"]


def test_it_never_invents_a_label():
    response = run(DeterministicLLMProvider().generate(request_with("[SOURCE S2]\nonly one")))

    assert json.loads(response.text)["sources"] == ["S2"]


def test_a_label_in_the_instructions_is_not_treated_as_knowledge():
    """Only the user message carries passages; the system role never does."""
    request = GenerationRequest(
        messages=(
            PromptMessage(role=MessageRole.SYSTEM, content="For example [SOURCE S7]"),
            PromptMessage(role=MessageRole.USER, content="[SOURCE S1]\nreal passage"),
        )
    )

    response = run(DeterministicLLMProvider().generate(request))

    assert json.loads(response.text)["sources"] == ["S1"]


def test_the_same_prompt_produces_the_same_reply():
    provider = DeterministicLLMProvider()

    assert run(provider.generate(REQUEST)).text == run(provider.generate(REQUEST)).text


def test_it_says_plainly_that_it_is_not_a_generated_answer():
    """A stub that read like an assistant would be worse than no stub at all."""
    answer = json.loads(run(DeterministicLLMProvider().generate(REQUEST)).text)["answer"]

    assert "development stub" in answer.lower()
    assert "not a generated answer" in answer.lower()


def test_it_answers_nothing_about_the_question_itself():
    response = run(
        DeterministicLLMProvider().generate(
            request_with("[SOURCE S1]\nThe sky is blue.\n\nQUESTION\nWhat colour is the sky?")
        )
    )

    assert "blue" not in json.loads(response.text)["answer"].lower()


def test_a_prompt_without_passages_is_handled_without_pretending():
    response = run(DeterministicLLMProvider().generate(request_with("KNOWLEDGE\n(none)")))

    payload = json.loads(response.text)
    assert payload["sources"] == []
    assert "no knowledge sources" in payload["answer"].lower()
