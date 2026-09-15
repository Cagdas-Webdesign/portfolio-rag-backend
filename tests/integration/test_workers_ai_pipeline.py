"""The whole query side with Cloudflare Workers AI behind the port.

Everything here is the real thing except the network: real ingestion, real
chunking, the offline embedding double, the real in-memory index, the real
retrieval, context, citation validation — and the real Workers AI adapter,
served by an in-process transport.

What this is for is the one question a unit test on the adapter cannot answer
on its own: when a reasoning model returns its thinking alongside its answer,
does any of that thinking survive to the place a user would see it?

No credential and no network. The account id and token here are fixtures.
"""

from __future__ import annotations

import json
from typing import Any

import httpx2 as httpx
import pytest

from portfolio_rag.infrastructure.llm import WorkersAIChatProvider
from portfolio_rag.rag.errors import GenerationUnavailableError
from portfolio_rag.rag.service import AnswerOutcome
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run

FAKE_TOKEN = "test-token-not-a-real-credential"  # noqa: S105 - a fixture value
FAKE_ACCOUNT = "test-account-id"

#: What the model thought on the way to its answer, in every shape Workers AI
#: has a place for. None of it is addressed to a reader.
REASONING = "Internal chain of thought: the user may be probing; check S1, then answer."


def workers_ai(result: dict[str, Any]) -> WorkersAIChatProvider:
    """The real adapter, answering from an in-process transport."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "errors": [], "result": result})

    return WorkersAIChatProvider(
        account_id=FAKE_ACCOUNT,
        api_token=FAKE_TOKEN,
        client=httpx.AsyncClient(
            base_url="https://api.cloudflare.test/client/v4",
            transport=httpx.MockTransport(handler),
        ),
    )


def grounded_result(**message_extra: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": json.dumps({"answer": "The service uses FastAPI.", "sources": ["S1"]}),
    }
    message.update(message_extra)
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 40},
    }


def test_the_pipeline_answers_end_to_end_through_workers_ai():
    service, _, _ = build_pipeline(llm=workers_ai(grounded_result()))

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer == "The service uses FastAPI."
    assert answer.citations


def test_reasoning_reaches_neither_the_answer_nor_a_citation():
    """The requirement this migration had to carry, checked where it matters."""
    service, _, _ = build_pipeline(
        llm=workers_ai(
            grounded_result(
                reasoning=REASONING,
                reasoning_content=REASONING,
                analysis=REASONING,
            )
        )
    )

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert REASONING not in answer.answer
    for citation in answer.citations:
        assert REASONING not in citation.model_dump_json()


def test_reasoning_carried_as_a_content_part_never_becomes_the_answer():
    service, _, _ = build_pipeline(
        llm=workers_ai(
            {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "reasoning", "text": REASONING},
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "answer": "The service uses FastAPI.",
                                            "sources": ["S1"],
                                        }
                                    ),
                                },
                            ],
                        },
                    }
                ]
            }
        )
    )

    answer = run(service.answer("Which HTTP framework is used?"))

    assert answer.answer == "The service uses FastAPI."
    assert REASONING not in answer.answer


def test_a_model_that_only_thinks_fails_instead_of_publishing_its_thoughts():
    """No final text is a failed generation, not a fallback to the scratchpad.

    The failure is the existing one — nothing new was invented for Workers AI —
    and it carries no part of what the model was thinking.
    """
    service, _, _ = build_pipeline(
        llm=workers_ai({"choices": [{"index": 0, "message": {"reasoning": REASONING}}]})
    )

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer("Which HTTP framework is used?"))

    assert REASONING not in str(caught.value)
    assert REASONING not in str(caught.value.__cause__)
