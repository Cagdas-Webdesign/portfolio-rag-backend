"""Pausing between evaluation questions — and nowhere else.

A recording sleeper stands in for time: every wait and every retrieval lands in
one event list, so the order is asserted directly and nothing actually sleeps.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any, cast

import pytest

from portfolio_rag.evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
    QuestionCategory,
    run_grounding_evaluation,
    run_retrieval_evaluation,
    validate_question_delay,
)
from portfolio_rag.rag.conversation import ConversationTurn
from portfolio_rag.rag.query import UserQuery
from portfolio_rag.rag.retrieval import PublicRetrievalService, RetrievalOutcome
from portfolio_rag.rag.service import GroundedAnswerService
from tests.doubles import ScriptedLLMProvider, grounded
from tests.integration.test_rag_pipeline import build_pipeline
from tests.support import run

EMPTY = RetrievalOutcome(
    chunks=(),
    matches_returned=0,
    below_threshold=0,
    unresolved=0,
    withheld=0,
    duration_seconds=0.0,
)


def dataset_of(count: int) -> EvaluationDataset:
    return EvaluationDataset(
        version=1,
        corpus="corpus",
        questions=tuple(
            EvaluationQuestion(
                id=f"q{index}", category=QuestionCategory.UNKNOWN, question=f"Question {index}?"
            )
            for index in range(1, count + 1)
        ),
    )


class Recorder:
    """One timeline for waits and requests."""

    def __init__(self) -> None:
        self.events: list[str] = []

    async def sleep(self, seconds: float) -> None:
        self.events.append(f"sleep {seconds}")

    async def retrieve(self, query: UserQuery) -> RetrievalOutcome:
        self.events.append(f"retrieve {query.text}")
        return EMPTY

    @property
    def sleeps(self) -> list[str]:
        return [event for event in self.events if event.startswith("sleep")]


def retrieval_run(count: int, delay: float) -> Recorder:
    recorder = Recorder()
    run(
        run_retrieval_evaluation(
            dataset_of(count),
            cast(PublicRetrievalService, recorder),
            delay_seconds=delay,
            sleeper=recorder.sleep,
        )
    )
    return recorder


def test_the_default_waits_nothing():
    """Case A, at the runner."""
    recorder = Recorder()

    run(
        run_retrieval_evaluation(
            dataset_of(5), cast(PublicRetrievalService, recorder), sleeper=recorder.sleep
        )
    )

    assert recorder.sleeps == []


@pytest.mark.parametrize("count", [1, 2, 12, 49])
def test_n_questions_wait_n_minus_one_times(count: int):
    """Case D."""
    assert len(retrieval_run(count, 3.0).sleeps) == count - 1


def test_the_first_question_is_asked_without_waiting():
    """Case E."""
    assert retrieval_run(3, 3.0).events[0] == "retrieve Question 1?"


def test_the_last_question_is_not_followed_by_a_wait():
    """Case F."""
    assert retrieval_run(3, 3.0).events[-1] == "retrieve Question 3?"


def test_every_wait_sits_between_two_questions():
    assert retrieval_run(3, 3.0).events == [
        "retrieve Question 1?",
        "sleep 3.0",
        "retrieve Question 2?",
        "sleep 3.0",
        "retrieve Question 3?",
    ]


def test_the_grounding_pass_is_paced_the_same_way():
    """Every answer embeds its question again, so it keeps the same gap."""
    recorder = Recorder()
    service, _, _ = build_pipeline(llm=ScriptedLLMProvider(grounded("A [S1].", "S1")))
    real_answer = service.answer

    async def answer(message: str, conversation: Sequence[ConversationTurn] = ()) -> Any:
        recorder.events.append(f"answer {message}")
        return await real_answer(message, conversation)

    service.answer = answer  # type: ignore[method-assign]

    run(run_grounding_evaluation(dataset_of(3), service, delay_seconds=2.0, sleeper=recorder.sleep))

    assert recorder.events == [
        "answer Question 1?",
        "sleep 2.0",
        "answer Question 2?",
        "sleep 2.0",
        "answer Question 3?",
    ]


@pytest.mark.parametrize("delay", [-1.0, -0.001, math.nan, math.inf])
def test_an_unusable_delay_is_refused_before_anything_is_asked(delay: float):
    """Case B, at the runner."""
    recorder = Recorder()

    with pytest.raises(ValueError, match="delay"):
        run(
            run_retrieval_evaluation(
                dataset_of(2),
                cast(PublicRetrievalService, recorder),
                delay_seconds=delay,
                sleeper=recorder.sleep,
            )
        )
    assert recorder.events == []


@pytest.mark.parametrize("delay", [0.0, 0.5, 3.0])
def test_a_usable_delay_is_accepted(delay: float):
    assert validate_question_delay(delay) == delay


def test_the_chat_path_is_never_paced(monkeypatch: pytest.MonkeyPatch):
    """Case I: answering questions outside a run makes no scheduling call at all."""
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def spy(seconds: float, *args: Any, **kwargs: Any) -> Any:
        slept.append(seconds)
        return await real_sleep(seconds, *args, **kwargs)

    monkeypatch.setattr(asyncio, "sleep", spy)
    service: GroundedAnswerService
    service, _, _ = build_pipeline(llm=ScriptedLLMProvider(grounded("A [S1].", "S1")))

    for _ in range(3):
        run(service.answer("Which framework serves the API?"))

    assert slept == []
