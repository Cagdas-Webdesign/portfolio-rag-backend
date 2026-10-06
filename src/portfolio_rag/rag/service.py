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
      → grounding check       ask again whether the cited passages say it
      → [not confirmed]       → honest answer, nothing published
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

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Final

from portfolio_rag.core.logging import get_logger
from portfolio_rag.domain.retrieval import SourceCitation
from portfolio_rag.ports.errors import LLMProviderError
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse, LLMProvider
from portfolio_rag.rag.citations import resolve_citations
from portfolio_rag.rag.context import ContextSource, GroundedContext, build_context
from portfolio_rag.rag.conversation import (
    NO_CONVERSATION,
    Conversation,
    ConversationTurn,
    bound_conversation,
)
from portfolio_rag.rag.errors import (
    GenerationFailure,
    GenerationFailureCategory,
    GenerationUnavailableError,
    ProviderCallType,
    RetrievalUnavailableError,
    provider_failure,
    visible_characters,
)
from portfolio_rag.rag.failure_policy import (
    classify_unusable_reply,
    may_recover,
    needs_larger_output,
)
from portfolio_rag.rag.generation import (
    GroundedAnswerDraft,
    SupportVerdict,
    cut_off_json,
    parse_generation,
)
from portfolio_rag.rag.language import (
    INSUFFICIENT_KNOWLEDGE_ANSWERS,
    AnswerLanguage,
    detect_language,
    insufficient_knowledge_answer,
)
from portfolio_rag.rag.policy import DEFAULT_CONTEXT_POLICY, ContextPolicy
from portfolio_rag.rag.presentation import present_answer
from portfolio_rag.rag.prompt import (
    CONVERSATION_PROMPT_VERSION,
    GROUNDED_PROMPT_VERSION,
    available_context_tokens,
    build_generation_request,
)
from portfolio_rag.rag.query import normalize_query
from portfolio_rag.rag.retrieval import PublicRetrievalService, RetrievalOutcome
from portfolio_rag.rag.telemetry import ProviderCallRecord, failed_call, replied_call
from portfolio_rag.rag.verification import (
    GROUNDING_CHECK_VERSION,
    GroundingVerdict,
    build_grounding_check_request,
    read_grounding_check,
)

_logger = get_logger(__name__)

#: The English wording of the refusal, and the one used when the language of a
#: question cannot be told apart. Fixed wording rather than the model's own
#: phrasing: predictable for a client, identical in both insufficiency paths,
#: and incapable of containing a claim. The German wording, and the choice
#: between them, live in :mod:`portfolio_rag.rag.language`.
INSUFFICIENT_KNOWLEDGE_ANSWER: Final = INSUFFICIENT_KNOWLEDGE_ANSWERS[AnswerLanguage.ENGLISH]

#: How many times one answer may be generated. Two: the generation, and one
#: more when the provider answered and what it delivered could not be used for
#: a reason a fresh sample fixes — a reply cut off at the output limit, or a
#: response that was not a completion at all. Measured against the real
#: provider, both happened to questions the same model had answered in full,
#: under the same limit, in earlier runs: the fault was in the sample, not in
#: the question or the budget.
#:
#: Not a transport retry — the adapter owns those, for failures to *reach* the
#: provider — and not a second chance for anything else: which failures allow
#: one is decided in :mod:`portfolio_rag.rag.failure_policy`, nowhere else.
#: Bounded at one extra request, so a provider that keeps failing costs twice,
#: never more. When the first reply stopped at the output limit, the second
#: request may spend up to :attr:`ContextPolicy.recovery_output_tokens`; the
#: first never does.
MAX_GENERATION_ATTEMPTS: Final = 2

#: How many times the grounding check may be asked for one answer. Two, for
#: exactly the failures and with exactly the output cap a generation gets its
#: second request for: a response that was not a completion, or a reply cut
#: off at the output limit — both measured against the real provider, both
#: with the whole cap spent on reasoning. A verdict, ``supported`` or
#: ``not_supported``, is never asked for again: it is an answer, not a fault.
MAX_GROUNDING_CHECK_ATTEMPTS: Final = 2

#: What the deadline is reported as when it ends a request during a provider
#: call. A fixed word, like the parser's rules.
DEADLINE_EXCEEDED: Final = "deadline_exceeded"


class AnswerOutcome(StrEnum):
    """How an answer came about. Diagnostics — never part of the HTTP response."""

    ANSWERED = "answered"
    NO_KNOWLEDGE = "no_knowledge"
    """Retrieval found nothing above the threshold. No provider was called."""

    NOT_GROUNDED = "not_grounded"
    """A model did not declare its answer stated by the passages, produced no
    answer text, or cited nothing the backend could verify — or the grounding
    check read the cited passages and said they do not carry the answer.
    A grounding check that gave no readable verdict is not this: it is a
    technical failure, and the request ends as one."""


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

    support: SupportVerdict | None = None
    """What the model declared about its own answer. ``None`` when no model was
    asked, or when its reply carried no verdict. Diagnostics."""

    claimed_labels: tuple[str, ...] = ()
    """The source labels exactly as the model listed them, before validation —
    including on a refusal, where none of them is published. Diagnostics."""

    grounding_check: GroundingVerdict | None = None
    """What the grounding check said. ``None`` when it was never asked: the
    answer had already been refused, or no model was called. Diagnostics."""

    grounding_check_seconds: float = 0.0

    generation_attempts: int = 0
    """How many times the answer was generated: ``1`` normally, ``2`` when the
    first reply was unusable and a second one was asked for, ``0`` when no
    model was called. Diagnostics."""

    grounding_check_attempts: int = 0
    """How many times the grounding check was asked: ``1`` normally, ``2`` when
    its first reply was unusable and it was asked again, ``0`` when it was not
    asked. Diagnostics."""

    conversation: Conversation = NO_CONVERSATION
    """The earlier turns the prompt carried, and how many were left out.
    Diagnostics — never evidence, and never part of the HTTP response."""

    provider_calls: tuple[ProviderCallRecord, ...] = ()
    """Every call this answer made to the generation provider, in order:
    each generation attempt, then each grounding check attempt if one was
    asked. Metadata only — see `rag.telemetry`. Diagnostics."""

    @property
    def regeneration_cause(self) -> GenerationFailure | None:
        """Why the first reply could not be used, when the answer took a second
        generation. ``None`` when it took one."""
        generations = [
            call for call in self.provider_calls if call.call_type is ProviderCallType.GENERATION
        ]
        return generations[0].failure if len(generations) > 1 else None

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
            (
                "grounding check",
                self.grounding_check.value if self.grounding_check is not None else "not run",
            ),
            ("grounding check seconds", f"{self.grounding_check_seconds:.4f}"),
            ("total seconds", f"{self.total_seconds:.4f}"),
        )


@dataclass
class _Progress:
    """What one request has done so far. Created, used and discarded by one call
    of :meth:`GroundedAnswerService.answer` — never shared, never global — so a
    request ended by its deadline can still say how far it got."""

    calls: list[ProviderCallRecord] = field(default_factory=list)
    step: ProviderCallType | None = None
    """``None`` while retrieving; the provider step under way after that."""

    retrieval: RetrievalOutcome | None = None


class _CallFailedError(Exception):
    """One provider call that could not be used, already classified. Private:
    it never leaves the service."""

    def __init__(self, failure: GenerationFailure, *, elapsed_seconds: float, reason: str) -> None:
        super().__init__(failure.detail)
        self.failure = failure
        self.elapsed_seconds = elapsed_seconds
        self.reason = reason


class GroundedAnswerService:
    """Answers one question from public knowledge, or honestly declines to."""

    def __init__(
        self,
        *,
        retrieval: PublicRetrievalService,
        llm: LLMProvider,
        context_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
        deadline_seconds: float | None = None,
    ) -> None:
        if deadline_seconds is not None and deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive")
        self._retrieval = retrieval
        self._llm = llm
        self._context_policy = context_policy
        self._deadline_seconds = deadline_seconds

    @property
    def context_policy(self) -> ContextPolicy:
        return self._context_policy

    @property
    def deadline_seconds(self) -> float | None:
        """The longest one answer may take, or ``None`` for no limit."""
        return self._deadline_seconds

    async def answer(
        self, message: str, conversation: Sequence[ConversationTurn] = ()
    ) -> GroundedAnswer:
        """Produce a grounded answer to *message*, or an honest refusal.

        *conversation* is the turns immediately before *message*, oldest
        first. They are used for one thing — letting the model read what the
        question refers to — and reach only the generation prompt: retrieval
        searches for *message* as asked, the grounding check never sees them,
        and they can produce no citation. See `rag.conversation`. Without
        them, every step below is exactly the single-question path.

        **One deadline for the whole request**, when one is configured: every
        step — retrieval, each generation, the grounding check, and every
        transport retry an adapter makes beneath them — spends from it. When it
        passes, the step under way is cancelled by asyncio's own mechanism, no
        further provider call is started, and the request ends as a technical
        error. Cancellation itself is never caught here.
        """
        started = time.perf_counter()
        progress = _Progress()
        try:
            async with asyncio.timeout(self._deadline_seconds) as deadline:
                return await self._answer(message, conversation, started, progress)
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise self._deadline_exceeded(progress) from exc
        except GenerationUnavailableError as exc:
            # What was retrieved and which calls were made are facts about this
            # request whether or not a model answered. They travel with the
            # failure for whoever examines it; the client's response is the
            # shared envelope and nothing of this.
            exc.retrieval = progress.retrieval
            exc.provider_calls = tuple(progress.calls)
            raise

    async def _answer(
        self,
        message: str,
        conversation: Sequence[ConversationTurn],
        started: float,
        progress: _Progress,
    ) -> GroundedAnswer:
        query = normalize_query(message)
        earlier = bound_conversation(conversation)
        # Decided once, from the question as asked, and used only if one of the
        # refusal paths below is taken.
        language = detect_language(query.text)

        retrieval = await self._retrieval.retrieve(query)
        progress.retrieval = retrieval
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
                conversation=earlier,
            )

        context = build_context(
            retrieval.chunks,
            available_tokens=available_context_tokens(query.text, self._context_policy, earlier),
        )

        progress.step = ProviderCallType.GENERATION
        generation_started = time.perf_counter()
        draft = await self._generate(query.text, context, earlier, progress.calls)
        generation_seconds = round(time.perf_counter() - generation_started, 4)
        generation_attempts = len(progress.calls)

        citations = resolve_citations(context, draft.source_labels)
        if (
            draft.support is not SupportVerdict.STATED
            or not draft.answer
            or not citations.is_grounded
        ):
            # Four ways in, one safe way out: the model did not declare its
            # answer stated by the passages, or produced no answer text at all,
            # or declared the context insufficient (empty sources), or answered
            # without pointing at anything real. None of them leaves evidence
            # to publish on, so nothing is published.
            # The first check stands on its own and fails closed. An answer the
            # model calls inferred is refused however real its labels are, and
            # so is a reply with no verdict at all: a missing verdict is not a
            # favourable one.
            # So does the answer check — an empty answer carrying a resolvable
            # label must not be published as a grounded blank.
            return self._insufficient(
                AnswerOutcome.NOT_GROUNDED,
                language=language,
                retrieval=retrieval,
                context=context,
                generation_seconds=generation_seconds,
                started=started,
                unknown_labels=citations.unknown_labels,
                draft=draft,
                generation_attempts=generation_attempts,
                conversation=earlier,
                provider_calls=tuple(progress.calls),
            )

        # The citations are real. Whether they *say what the answer says* is a
        # different question, and the generating model has already answered it
        # in its own favour — so it is asked again, separately. Only an answer
        # that passed everything above gets here, which is what keeps this a
        # second call for answers and no call at all for refusals.
        progress.step = ProviderCallType.GROUNDING_CHECK
        check_started = time.perf_counter()
        verdict = await self._check_grounding(
            query.text, draft.answer, citations.cited, progress.calls
        )
        check_seconds = round(time.perf_counter() - check_started, 4)
        check_attempts = len(progress.calls) - generation_attempts

        if verdict is not GroundingVerdict.SUPPORTED:
            # The check read the cited passages and said they do not carry the
            # answer. A quality finding, not a fault: the answer existed and is
            # not published. (A check that gave no verdict never gets here.)
            return self._insufficient(
                AnswerOutcome.NOT_GROUNDED,
                language=language,
                retrieval=retrieval,
                context=context,
                generation_seconds=generation_seconds,
                started=started,
                unknown_labels=citations.unknown_labels,
                draft=draft,
                generation_attempts=generation_attempts,
                grounding_check=verdict,
                grounding_check_seconds=check_seconds,
                grounding_check_attempts=check_attempts,
                conversation=earlier,
                provider_calls=tuple(progress.calls),
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
                support=draft.support,
                claimed_labels=draft.source_labels,
                grounding_check=verdict,
                grounding_check_seconds=check_seconds,
                generation_attempts=generation_attempts,
                grounding_check_attempts=check_attempts,
                conversation=earlier,
                provider_calls=tuple(progress.calls),
            )
        )

    # --- steps --------------------------------------------------------------

    async def _generate(
        self,
        question: str,
        context: GroundedContext,
        conversation: Conversation,
        calls: list[ProviderCallRecord],
    ) -> GroundedAnswerDraft:
        """Generate a draft, recording every call into *calls*.

        One request normally. A second one only when the failure policy allows
        it for what went wrong — see `rag.failure_policy` — and never a third.
        The second asks the same thing, with the larger output cap when the
        first stopped at the limit. Every other failure is raised on the spot.
        """
        request = build_generation_request(
            question=question,
            context=context,
            max_output_tokens=self._context_policy.output_reserve_tokens,
            conversation=conversation,
        )
        requests_made = 0
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            try:
                response, elapsed = await self._send(request, ProviderCallType.GENERATION)
            except _CallFailedError as failed:
                requests_made += failed.failure.attempts
                failure = replace(failed.failure, attempts=requests_made)
                calls.append(self._observe(self._failed(request, attempt, failed, failure)))
                if may_recover(failure) and attempt < MAX_GENERATION_ATTEMPTS:
                    _logger.warning(
                        "generation regenerated",
                        extra={"reason": failed.reason, **failure.fields()},
                    )
                    request = self._recovery_request(request, failure)
                    continue
                _logger.warning(
                    "generation failed", extra={"reason": failed.reason, **failure.fields()}
                )
                raise GenerationUnavailableError(failure=failure) from failed.__cause__

            requests_made += 1
            try:
                draft = parse_generation(response.text)
            except GenerationUnavailableError as exc:
                # The provider answered and the reply did not satisfy the answer
                # contract. Placed by the classification precedence — cut off at
                # the limit, or not — and logged with what can be said about a
                # reply without quoting it.
                failure = classify_unusable_reply(
                    _with_reply_facts(
                        replace(
                            exc.failure
                            or GenerationFailure(
                                category=GenerationFailureCategory.UNPARSEABLE_OUTPUT,
                                detail="unspecified",
                            ),
                            attempts=requests_made,
                        ),
                        response,
                    ),
                    finish_reason=response.finish_reason,
                    cut_off=cut_off_json(response.text),
                )
                calls.append(
                    self._observe(
                        self._replied(
                            request,
                            ProviderCallType.GENERATION,
                            attempt,
                            elapsed,
                            response,
                            failure,
                        )
                    )
                )
                if may_recover(failure) and attempt < MAX_GENERATION_ATTEMPTS:
                    # Nothing of the rejected reply is kept: the next one is read
                    # from scratch, by the same strict parser, and then has to
                    # pass every check this one would have.
                    _logger.warning("generation regenerated", extra=failure.fields())
                    request = self._recovery_request(request, failure)
                    continue
                _logger.warning("generation unusable", extra=failure.fields())
                raise GenerationUnavailableError(failure=failure) from exc

            calls.append(
                self._observe(
                    self._replied(
                        request, ProviderCallType.GENERATION, attempt, elapsed, response, None
                    )
                )
            )
            return draft

        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    async def _check_grounding(
        self,
        question: str,
        answer: str,
        cited: tuple[ContextSource, ...],
        calls: list[ProviderCallRecord],
    ) -> GroundingVerdict:
        """Ask whether *cited* says what *answer* says. See `rag.verification`.

        ``supported`` or ``not_supported`` is returned, and either ends the
        check. A reply that is no verdict at all, and a provider that could not
        be reached, are technical failures: asked once more when the failure
        policy allows it for what went wrong — the same rule, and the same
        larger output cap, as a regeneration — and raised otherwise, or when
        the second request fails too. The check then said nothing about the
        passages, so nothing is published and nothing is claimed about the
        knowledge base either.
        """
        request = build_grounding_check_request(
            question=question,
            answer=answer,
            cited=cited,
            max_output_tokens=self._context_policy.output_reserve_tokens,
        )
        requests_made = 0
        for attempt in range(1, MAX_GROUNDING_CHECK_ATTEMPTS + 1):
            try:
                response, elapsed = await self._send(request, ProviderCallType.GROUNDING_CHECK)
            except _CallFailedError as failed:
                requests_made += failed.failure.attempts
                failure = replace(failed.failure, attempts=requests_made)
                calls.append(self._observe(self._failed(request, attempt, failed, failure)))
                if may_recover(failure) and attempt < MAX_GROUNDING_CHECK_ATTEMPTS:
                    _logger.warning(
                        "grounding check repeated",
                        extra={"reason": failed.reason, **failure.fields()},
                    )
                    request = self._recovery_request(request, failure)
                    continue
                _logger.warning(
                    "grounding check failed",
                    extra={"reason": failed.reason, **failure.fields()},
                )
                raise GenerationUnavailableError(failure=failure) from failed.__cause__

            requests_made += 1
            verdict, rule = read_grounding_check(response.text)
            if verdict is not GroundingVerdict.UNUSABLE:
                calls.append(
                    self._observe(
                        self._replied(
                            request,
                            ProviderCallType.GROUNDING_CHECK,
                            attempt,
                            elapsed,
                            response,
                            None,
                        )
                    )
                )
                return verdict

            failure = classify_unusable_reply(
                _with_reply_facts(
                    GenerationFailure(
                        category=GenerationFailureCategory.UNPARSEABLE_OUTPUT,
                        detail=rule or "unspecified",
                        step=ProviderCallType.GROUNDING_CHECK,
                        attempts=requests_made,
                    ),
                    response,
                ),
                finish_reason=response.finish_reason,
                cut_off=cut_off_json(response.text),
            )
            calls.append(
                self._observe(
                    self._replied(
                        request,
                        ProviderCallType.GROUNDING_CHECK,
                        attempt,
                        elapsed,
                        response,
                        failure,
                    )
                )
            )
            # Nothing of the reply is logged but which rule it broke, its size and
            # how it ended.
            if may_recover(failure) and attempt < MAX_GROUNDING_CHECK_ATTEMPTS:
                _logger.warning("grounding check repeated", extra=failure.fields())
                request = self._recovery_request(request, failure)
                continue
            _logger.warning("grounding check unusable", extra=failure.fields())
            raise GenerationUnavailableError(failure=failure)

        raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover

    def _recovery_request(
        self, request: GenerationRequest, failure: GenerationFailure
    ) -> GenerationRequest:
        """The second request of a step: the same messages, and the larger
        output cap when the first stopped at the limit. Nothing else changes."""
        if not needs_larger_output(failure):
            return request
        return request.model_copy(
            update={"max_output_tokens": self._context_policy.recovery_output_tokens}
        )

    async def _send(
        self, request: GenerationRequest, step: ProviderCallType
    ) -> tuple[GenerationResponse, float]:
        """The provider boundary: one call through the port, and nothing else.

        Every way the call can fail leaves as :class:`_CallFailedError`, already
        classified, with the original exception as its cause:

        * an :class:`LLMProviderError` — the port's own contract — is placed by
          :func:`~portfolio_rag.rag.errors.provider_failure`;
        * anything else an adapter raises is a breach of that contract and is
          ``unclassified``, named by its exception class and never by its text.

        That second clause is deliberately broad and deliberately narrow: it
        covers exactly this one awaited call, so an adapter's bug is a 503 with
        its cause chained rather than a bare 500 — and it cannot catch a defect
        anywhere else in the pipeline. Cancellation is a ``BaseException`` and
        passes through untouched.
        """
        started = time.perf_counter()
        try:
            response = await self._llm.generate(request)
        except LLMProviderError as exc:
            raise _CallFailedError(
                provider_failure(exc, step=step),
                elapsed_seconds=_seconds_since(started),
                reason=exc.describe(),
            ) from exc
        except Exception as exc:  # the provider boundary — see the docstring
            raise _CallFailedError(
                GenerationFailure(
                    category=GenerationFailureCategory.UNCLASSIFIED,
                    detail=type(exc).__name__,
                    step=step,
                ),
                elapsed_seconds=_seconds_since(started),
                reason=type(exc).__name__,
            ) from exc
        return response, _seconds_since(started)

    def _deadline_exceeded(self, progress: _Progress) -> Exception:
        """The technical error a request ends as when its deadline passes.

        Named after the stage it interrupted, with the existing public codes:
        retrieval unavailable before any provider call, generation unavailable
        during one. The calls already completed are kept; the one that was
        cancelled left no record, because it delivered nothing.
        """
        _logger.warning(
            "request deadline exceeded",
            extra={
                "deadline_seconds": self._deadline_seconds,
                "step": progress.step.value if progress.step else "retrieval",
                "provider_calls": len(progress.calls),
            },
        )
        if progress.step is None:
            return RetrievalUnavailableError()
        error = GenerationUnavailableError(
            failure=GenerationFailure(
                category=GenerationFailureCategory.TIMEOUT,
                detail=DEADLINE_EXCEEDED,
                step=progress.step,
                attempts=1 + sum(1 for call in progress.calls if call.call_type is progress.step),
            )
        )
        error.retrieval = progress.retrieval
        error.provider_calls = tuple(progress.calls)
        return error

    def _replied(
        self,
        request: GenerationRequest,
        step: ProviderCallType,
        attempt: int,
        elapsed: float,
        response: GenerationResponse,
        failure: GenerationFailure | None,
    ) -> ProviderCallRecord:
        return replied_call(
            call_type=step,
            attempt=attempt,
            model=self._llm.model,
            response_format=request.response_format,
            elapsed_seconds=elapsed,
            response=response,
            failure=failure,
            max_output_tokens=request.max_output_tokens,
        )

    def _failed(
        self,
        request: GenerationRequest,
        attempt: int,
        failed: _CallFailedError,
        failure: GenerationFailure,
    ) -> ProviderCallRecord:
        return failed_call(
            call_type=failure.step,
            attempt=attempt,
            model=self._llm.model,
            response_format=request.response_format,
            elapsed_seconds=failed.elapsed_seconds,
            failure=failure,
            max_output_tokens=request.max_output_tokens,
            call_failure=failed.failure,
        )

    @staticmethod
    def _observe(call: ProviderCallRecord) -> ProviderCallRecord:
        """Log one provider call — metadata only — and hand the record back.

        One line per call, successful or not, so that an ordinary call has a
        baseline to be compared against. The request id the logging filter
        adds is what ties it to its request, or to its evaluation run.
        """
        _logger.info("provider call", extra=call.fields())
        return call

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
        draft: GroundedAnswerDraft | None = None,
        grounding_check: GroundingVerdict | None = None,
        grounding_check_seconds: float = 0.0,
        generation_attempts: int = 0,
        grounding_check_attempts: int = 0,
        conversation: Conversation = NO_CONVERSATION,
        provider_calls: tuple[ProviderCallRecord, ...] = (),
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
                support=draft.support if draft is not None else None,
                claimed_labels=draft.source_labels if draft is not None else (),
                grounding_check=grounding_check,
                grounding_check_seconds=grounding_check_seconds,
                generation_attempts=generation_attempts,
                grounding_check_attempts=grounding_check_attempts,
                conversation=conversation,
                provider_calls=provider_calls,
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
                "conversation_prompt_version": (
                    CONVERSATION_PROMPT_VERSION if not answer.conversation.is_empty else None
                ),
                "conversation_turns": len(answer.conversation.turns),
                "conversation_turns_dropped": answer.conversation.dropped,
                "grounding_check_version": GROUNDING_CHECK_VERSION,
                "grounding_check": (
                    answer.grounding_check.value if answer.grounding_check is not None else None
                ),
                "grounding_check_seconds": answer.grounding_check_seconds,
                "retrieval_seconds": answer.retrieval.duration_seconds,
                "generation_seconds": answer.generation_seconds,
                "generation_attempts": answer.generation_attempts,
                "grounding_check_attempts": answer.grounding_check_attempts,
                "provider_calls": len(answer.provider_calls),
                "regeneration_cause": (
                    answer.regeneration_cause.detail if answer.regeneration_cause else None
                ),
                "total_seconds": answer.total_seconds,
            },
        )
        return answer


def _with_reply_facts(
    failure: GenerationFailure, response: GenerationResponse
) -> GenerationFailure:
    """*failure*, with what can be said about the reply without quoting it."""
    return replace(
        failure,
        finish_reason=response.finish_reason,
        reply_characters=len(response.text),
        reply_visible_characters=visible_characters(response.text),
        input_tokens=response.usage.input_tokens if response.usage else None,
        output_tokens=response.usage.output_tokens if response.usage else None,
    )


def _seconds_since(started: float) -> float:
    return round(time.perf_counter() - started, 4)
