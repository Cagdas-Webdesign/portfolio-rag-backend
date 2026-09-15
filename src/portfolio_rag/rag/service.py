"""The query side, end to end.

    question
      → query boundary        normalize, bound, reject the unusable
      → retrieval             public passages, ranked, thresholded
      → [nothing found]       → honest answer, no provider call
      → context building      bounded, labelled, deterministic
      → generation            through the LLMProvider port
      → answer contract       parse what the model claimed
      → citation validation   keep only claims the backend can prove
      → [nothing proven]      → honest answer, no unproven claims published
      → presentation          renumber verified marks for a reader
      → GroundedAnswer

This class coordinates; it does not compute. Retrieval, context building,
prompting, parsing and citation mapping each live in their own module and are
tested there, which is what keeps this file readable and stops it from becoming
the 500-line object that every RAG codebase eventually grows.

It knows ports and its own package. It does not know FastAPI, Mistral,
Cloudflare or HTTP, and swapping any provider leaves it untouched.

**Two ways to have no answer, and both are successes.** A knowledge base that
does not cover a question, and a model that cannot ground an answer in what it
was given, both produce :data:`INSUFFICIENT_KNOWLEDGE_ANSWER` with no
citations and a ``200``. Neither is a server error, and neither is allowed to
become a plausible-sounding invention. There is no fallback to the model's
general knowledge: the authority in this system is the retrieved corpus, and an
assistant that answers from elsewhere when the corpus is silent is no longer a
portfolio assistant.

**The refusal is written in the language of the question.** Grounded answers
already are — the prompt says so — so a German question that the corpus cannot
answer came back in English, which reads as a broken assistant rather than an
honest one. :mod:`portfolio_rag.rag.language` picks between prepared sentences;
it chooses wording and nothing else, and no path here can be reached because of
it.

**Grounding is a process, not a proof.** Every answer is generated from
retrieved public passages and carries only sources the backend verified. That
reduces the room a model has to invent and makes what it says checkable. It is
not a guarantee of correctness, and nothing in this project claims one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from portfolio_rag.core.logging import get_logger
from portfolio_rag.domain.retrieval import SourceCitation
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import LLMProvider
from portfolio_rag.rag.citations import resolve_citations
from portfolio_rag.rag.context import GroundedContext, build_context
from portfolio_rag.rag.errors import GenerationUnavailableError
from portfolio_rag.rag.generation import GroundedAnswerDraft, parse_generation
from portfolio_rag.rag.language import (
    INSUFFICIENT_KNOWLEDGE_ANSWERS,
    AnswerLanguage,
    detect_language,
    insufficient_knowledge_answer,
)
from portfolio_rag.rag.policy import DEFAULT_CONTEXT_POLICY, ContextPolicy
from portfolio_rag.rag.presentation import present_answer
from portfolio_rag.rag.prompt import (
    GROUNDED_PROMPT_VERSION,
    available_context_tokens,
    build_generation_request,
)
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService, RetrievalOutcome

_logger = get_logger(__name__)

#: The English wording of the refusal, and the one used when the language of a
#: question cannot be told apart. Fixed wording rather than the model's own
#: phrasing: predictable for a client, identical in both insufficiency paths,
#: and incapable of containing a claim. The German wording, and the choice
#: between them, live in :mod:`portfolio_rag.rag.language`.
INSUFFICIENT_KNOWLEDGE_ANSWER: Final = INSUFFICIENT_KNOWLEDGE_ANSWERS[AnswerLanguage.ENGLISH]


class AnswerOutcome(StrEnum):
    """How an answer came about. Diagnostics — never part of the HTTP response."""

    ANSWERED = "answered"
    NO_KNOWLEDGE = "no_knowledge"
    """Retrieval found nothing above the threshold. No provider was called."""

    NOT_GROUNDED = "not_grounded"
    """A model produced no answer text, or cited nothing the backend could verify."""


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer, and everything the process knows about how it was produced.

    The API returns exactly two of these fields — :attr:`answer` and
    :attr:`citations` — by naming them, never by serializing this object. The
    rest is developer material: it is what the CLI shows, what the logs count,
    and what evaluation measures. Keeping it attached to the result
    rather than recomputing it is why ``query answer --show-context`` can print
    the context the model *actually* received instead of a second one built to
    look like it.
    """

    answer: str
    citations: tuple[SourceCitation, ...]
    outcome: AnswerOutcome
    retrieval: RetrievalOutcome
    context: GroundedContext | None
    """``None`` when retrieval short-circuited before a context was built."""

    unknown_labels: tuple[str, ...]
    generation_seconds: float
    total_seconds: float

    @property
    def is_grounded(self) -> bool:
        return self.outcome is AnswerOutcome.ANSWERED

    @property
    def retrieved_count(self) -> int:
        return len(self.retrieval.chunks)

    @property
    def context_source_count(self) -> int:
        return len(self.context.sources) if self.context is not None else 0

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("outcome", self.outcome.value),
            ("retrieved", str(self.retrieved_count)),
            ("context sources", str(self.context_source_count)),
            ("citations", str(len(self.citations))),
            ("unknown labels", str(len(self.unknown_labels))),
            ("retrieval seconds", f"{self.retrieval.duration_seconds:.4f}"),
            ("generation seconds", f"{self.generation_seconds:.4f}"),
            ("total seconds", f"{self.total_seconds:.4f}"),
        )


class GroundedAnswerService:
    """Answers one question from public knowledge, or honestly declines to."""

    def __init__(
        self,
        *,
        retrieval: PublicRetrievalService,
        llm: LLMProvider,
        context_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
    ) -> None:
        self._retrieval = retrieval
        self._llm = llm
        self._context_policy = context_policy

    @property
    def context_policy(self) -> ContextPolicy:
        return self._context_policy

    async def answer(self, message: str) -> GroundedAnswer:
        """Produce a grounded answer to *message*, or an honest refusal."""
        started = time.perf_counter()
        query = normalize_query(message)
        # Decided once, from the question as asked, and used only if one of the
        # two refusal paths below is taken.
        language = detect_language(query.text)

        retrieval = await self._retrieval.retrieve(query)
        if not retrieval.is_sufficient:
            # No provider call at all. Nothing was found to ground an answer
            # in, so there is nothing to ask a model about — and asking anyway
            # would spend a request to invite an invention.
            return self._insufficient(
                AnswerOutcome.NO_KNOWLEDGE,
                language=language,
                retrieval=retrieval,
                context=None,
                generation_seconds=0.0,
                started=started,
            )

        context = build_context(
            retrieval.chunks,
            available_tokens=available_context_tokens(query.text, self._context_policy),
        )

        generation_started = time.perf_counter()
        draft = await self._generate(query.text, context)
        generation_seconds = round(time.perf_counter() - generation_started, 4)

        citations = resolve_citations(context, draft.source_labels)
        if not draft.answer or not citations.is_grounded:
            # Three ways in, one safe way out: the model produced no answer
            # text at all, or declared the context insufficient (empty
            # sources), or answered without pointing at anything real. None of
            # them leaves evidence to publish on, so nothing is published.
            # Note the answer check stands on its own — an empty answer carrying
            # a resolvable label must not be published as a grounded blank.
            return self._insufficient(
                AnswerOutcome.NOT_GROUNDED,
                language=language,
                retrieval=retrieval,
                context=context,
                generation_seconds=generation_seconds,
                started=started,
                unknown_labels=citations.unknown_labels,
            )

        # Last, and only here: the answer has been generated, parsed and
        # verified, and every citation below was proven before this line. All
        # that is left is to renumber marks a reader should not have to decode.
        # Nothing in this step can admit a source — see `rag.presentation`.
        presented = present_answer(draft.answer, citations.citations, citations.cited)

        return self._finish(
            GroundedAnswer(
                answer=presented.answer,
                citations=presented.citations,
                outcome=AnswerOutcome.ANSWERED,
                retrieval=retrieval,
                context=context,
                unknown_labels=citations.unknown_labels,
                generation_seconds=generation_seconds,
                total_seconds=round(time.perf_counter() - started, 4),
            )
        )

    # --- steps --------------------------------------------------------------

    async def _generate(self, question: str, context: GroundedContext) -> GroundedAnswerDraft:
        request = build_generation_request(
            question=question,
            context=context,
            max_output_tokens=self._context_policy.output_reserve_tokens,
        )
        try:
            response = await self._llm.generate(request)
        except LLMProviderError as exc:
            _logger.warning("generation failed", extra={"reason": exc.describe()})
            raise GenerationUnavailableError from exc
        return parse_generation(response.text)

    def _insufficient(
        self,
        outcome: AnswerOutcome,
        *,
        language: AnswerLanguage,
        retrieval: RetrievalOutcome,
        context: GroundedContext | None,
        generation_seconds: float,
        started: float,
        unknown_labels: tuple[str, ...] = (),
    ) -> GroundedAnswer:
        return self._finish(
            GroundedAnswer(
                answer=insufficient_knowledge_answer(language),
                citations=(),
                outcome=outcome,
                retrieval=retrieval,
                context=context,
                unknown_labels=unknown_labels,
                generation_seconds=generation_seconds,
                total_seconds=round(time.perf_counter() - started, 4),
            )
        )

    def _finish(self, answer: GroundedAnswer) -> GroundedAnswer:
        """Record what happened. Counts, durations and identifiers only.

        Never the question, never a passage, never the answer, never the
        prompt — the request id in the log line is how a specific request is
        found, not its content.
        """
        _logger.info(
            "chat answered",
            extra={
                "outcome": answer.outcome.value,
                "retrieved_count": answer.retrieved_count,
                "context_sources": answer.context_source_count,
                "citation_count": len(answer.citations),
                "unknown_label_count": len(answer.unknown_labels),
                "embedding_space": self._retrieval.spec.identity,
                "generation_model": self._llm.model,
                "prompt_version": GROUNDED_PROMPT_VERSION,
                "retrieval_seconds": answer.retrieval.duration_seconds,
                "generation_seconds": answer.generation_seconds,
                "total_seconds": answer.total_seconds,
            },
        )
        return answer
