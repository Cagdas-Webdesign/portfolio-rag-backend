"""The end-to-end evaluation: one pass per question, and what it is held to.

A small real stack — two chunks, the in-memory index, the real answering
service — with the model's decision scripted. What is checked is the scoring
around that decision: which outcome counts as which failure, which failures are
gates, and that the export and the summary say exactly what the records do.

The states the pipeline is built to make impossible (a leaked label, a citation
with no passage behind it) are produced by editing a real answer, because the
point is that the evaluation would notice them, not that the pipeline produces
them.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.domain.retrieval import SourceCitation
from portfolio_rag.evaluation import (
    E2E_FORMAT_VERSION,
    E2EFailure,
    E2EReport,
    EvaluationQuestion,
    QuestionCategory,
    corpus_identity,
    export_e2e,
    render_summary,
    run_e2e_evaluation,
    score_answer,
)
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ports.errors import LLMProviderError, ProviderFailureKind
from portfolio_rag.ports.llm import LLMProvider
from portfolio_rag.rag.errors import GenerationFailureCategory
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswer, GroundedAnswerService
from portfolio_rag.rag.verification import GROUNDING_CHECK_VERSION
from tests.doubles import (
    FailingEmbeddingProvider,
    FailingLLMProvider,
    LexicalEmbeddingProvider,
    ScriptedLLMProvider,
    ScriptedReply,
    checked,
    grounded,
)
from tests.e2e_stack import (
    ANSWERABLE,
    CHUNKS,
    dataset_of,
    e2e_retrieval,
    e2e_service,
    label_of,
    run_metadata,
)
from tests.support import run

UNKNOWN = EvaluationQuestion(
    id="q-salary",
    category=QuestionCategory.UNKNOWN,
    question="What is the salary of the backend API author?",
)
DECLINED = ScriptedReply(answer="The passages do not cover this.", sources=())


def _answer(question: EvaluationQuestion, llm: LLMProvider) -> GroundedAnswer:
    service = GroundedAnswerService(
        retrieval=e2e_retrieval(), llm=llm, context_policy=ContextPolicy()
    )
    return run(service.answer(question.question))


def _scored(question: EvaluationQuestion, *replies: ScriptedReply):
    answer = _answer(question, ScriptedLLMProvider(*replies))
    return score_answer(question, answer, duration_seconds=0.5)


# --- what counts as what -------------------------------------------------------


def test_a_grounded_answer_from_the_expected_source_has_no_failure():
    label = label_of("stack", ANSWERABLE)

    record = _scored(ANSWERABLE, grounded(f"FastAPI [{label}].", label))

    assert record.outcome is AnswerOutcome.ANSWERED
    assert record.failures == ()
    assert record.verified_citations == len(record.citations) == 1
    assert record.cites_expected_source
    assert record.retrieval.first_relevant_rank is not None


def test_citing_only_a_source_the_ground_truth_does_not_list_is_measured_not_failed():
    """A fitting passage outside the ground truth is not waved through as correct."""
    label = label_of("deploy", ANSWERABLE)

    record = _scored(ANSWERABLE, grounded(f"Docker [{label}].", label))

    assert record.failures == ()
    assert not record.cites_expected_source


def test_an_answerable_question_the_model_declines_is_a_quality_finding():
    record = _scored(ANSWERABLE, DECLINED)

    assert record.outcome is AnswerOutcome.NOT_GROUNDED
    assert record.failures == (E2EFailure.NOT_ANSWERED,)
    assert E2EReport(records=(record,)).passed


def test_an_unanswerable_question_that_is_refused_is_a_controlled_refusal():
    record = _scored(UNKNOWN, DECLINED)

    assert record.is_controlled_refusal
    assert record.citations == ()
    assert record.failures == ()


def test_an_unanswerable_question_that_gets_answered_breaks_a_gate():
    record = _scored(UNKNOWN, grounded("It is stated here [S1].", "S1"))
    report = E2EReport(records=(record,))

    assert record.failures == (E2EFailure.ANSWERED_UNKNOWN,)
    assert not record.is_controlled_refusal
    assert report.gates["unanswerable_refused"] is False
    assert not report.passed


def test_a_label_the_model_invented_is_a_failure_even_though_it_was_dropped():
    label = label_of("stack", ANSWERABLE)

    record = _scored(ANSWERABLE, grounded(f"FastAPI [{label}].", label, "S9"))

    assert record.unknown_labels == ("S9",)
    assert record.verified_citations == len(record.citations) == 1
    assert record.failures == (E2EFailure.INVENTED_LABEL,)
    assert E2EReport(records=(record,)).gates["citations_verified"] is False


def test_states_the_pipeline_prevents_would_still_be_noticed():
    label = label_of("stack", ANSWERABLE)
    answer = _answer(ANSWERABLE, ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label)))
    forged = SourceCitation(document_id="elsewhere", title="Elsewhere", source="x.md")

    def failures(changed: GroundedAnswer) -> tuple[E2EFailure, ...]:
        return score_answer(ANSWERABLE, changed, duration_seconds=0.1).failures

    assert failures(replace(answer, answer="FastAPI [S2].")) == (E2EFailure.LABEL_LEAK,)
    assert failures(replace(answer, answer="FastAPI [1] [4].")) == (E2EFailure.DANGLING_MARK,)
    assert failures(replace(answer, citations=(forged,))) == (E2EFailure.UNVERIFIED_CITATION,)
    assert failures(replace(answer, answer="  ")) == (E2EFailure.EMPTY_ANSWER,)
    assert failures(replace(answer, citations=())) == (
        E2EFailure.NO_CITATION,
        E2EFailure.DANGLING_MARK,
    )


# --- the run -------------------------------------------------------------------


def test_every_question_is_asked_exactly_once():
    label = label_of("stack", ANSWERABLE)
    llm = ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label), DECLINED)

    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE, UNKNOWN), e2e_service(llm)))

    assert llm.call_count == 2
    assert [record.question.id for record in report.records] == ["q-api", "q-salary"]
    assert report.passed
    assert report.retrieval.hit_rates[5].correct == 1


def test_a_provider_failure_is_recorded_and_the_run_goes_on():
    report = run(
        run_e2e_evaluation(dataset_of(ANSWERABLE, UNKNOWN), e2e_service(FailingLLMProvider()))
    )

    assert [record.errored for record in report.records] == [True, True]
    assert report.records[0].error_code == "GENERATION_UNAVAILABLE"
    assert report.records[1].failures == (E2EFailure.PIPELINE_ERROR,)
    assert report.gates["no_pipeline_errors"] is False


def test_a_failed_generation_does_not_turn_a_found_passage_into_a_miss():
    """The provider not answering says nothing about what retrieval found."""
    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(FailingLLMProvider())))
    record = report.records[0]

    assert record.retrieval_observed
    assert record.retrieval.first_relevant_rank is not None
    # One root cause, one failure: not also a quality finding (e2e-eval-v3).
    assert record.failures == (E2EFailure.PIPELINE_ERROR,)
    assert report.retrieval.hit_rates[5].correct == 1
    assert report.retrieval.hit_rates[5].total == 1


def test_a_failed_generation_says_why_in_technical_terms_only():
    refused = LLMProviderError(
        "Provider returned HTTP 400 during generation.",
        kind=ProviderFailureKind.HTTP_STATUS,
        status_code=400,
    )
    report = run(
        run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(FailingLLMProvider(refused)))
    )
    error = report.records[0].error

    assert error is not None
    assert error.category is GenerationFailureCategory.PROVIDER_STATUS
    assert (error.status_code, error.retryable, error.attempts) == (400, False, 1)


def test_an_unusable_reply_is_recorded_with_how_it_ended():
    llm = ScriptedLLMProvider(ScriptedReply(raw_text="Sure! The answer is FastAPI."))
    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(llm)))
    error = report.records[0].error

    assert error is not None
    assert error.category is GenerationFailureCategory.UNPARSEABLE_OUTPUT
    assert (error.detail, error.finish_reason) == ("reply_not_json", "stop")
    assert error.reply_characters == len("Sure! The answer is FastAPI.")


def test_a_question_whose_retrieval_never_happened_is_left_out_not_counted_as_a_miss():
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(CHUNKS))
    broken = PublicRetrievalService(
        embeddings=FailingEmbeddingProvider(embeddings.spec),
        store=store,
        resolver=InMemoryChunkResolver(CHUNKS),
    )
    service = GroundedAnswerService(
        retrieval=broken, llm=ScriptedLLMProvider(), context_policy=ContextPolicy()
    )

    report = run(run_e2e_evaluation(dataset_of(ANSWERABLE), service))
    record = report.records[0]

    assert record.error_code == "RETRIEVAL_UNAVAILABLE"
    assert record.error is None
    assert not record.retrieval_observed
    # One root cause, one failure: not also a quality finding (e2e-eval-v3).
    assert record.failures == (E2EFailure.PIPELINE_ERROR,)
    assert report.retrieval.hit_rates[5].total == 0
    assert [r.question.id for r in report.retrieval_not_observed] == ["q-api"]


def test_the_pause_is_kept_between_questions_only():
    slept: list[float] = []

    async def sleeper(seconds: float) -> None:
        slept.append(seconds)

    llm = ScriptedLLMProvider(DECLINED)
    run(
        run_e2e_evaluation(
            dataset_of(ANSWERABLE, UNKNOWN, UNKNOWN.model_copy(update={"id": "q-other"})),
            e2e_service(llm),
            delay_seconds=3.0,
            sleeper=sleeper,
        )
    )

    assert slept == [3.0, 3.0]


# --- export and summary --------------------------------------------------------


def _exported(*, notes: tuple[str, ...] = ()) -> dict[str, Any]:
    label = label_of("stack", ANSWERABLE)
    dataset = dataset_of(ANSWERABLE, UNKNOWN)
    llm = ScriptedLLMProvider(
        grounded(f"FastAPI [{label}].", label), grounded("It says so [S1].", "S1")
    )
    report = run(run_e2e_evaluation(dataset, e2e_service(llm)))
    return export_e2e(report, run_metadata(dataset, notes=notes))


def test_the_export_names_its_run_and_every_question():
    payload = _exported()

    assert payload["format"] == E2E_FORMAT_VERSION
    assert payload["run"]["generation"] == {
        "provider": "scripted",
        "model": ScriptedLLMProvider.MODEL,
        "prompt_version": "grounded-answer-v3",
        "grounding_check_version": GROUNDING_CHECK_VERSION,
        "response_format": "json_object",
        "grounding_check_response_format": "json_object",
        "max_prompt_tokens": ContextPolicy().max_prompt_tokens,
        "output_reserve_tokens": ContextPolicy().output_reserve_tokens,
        "max_output_tokens": ContextPolicy().output_reserve_tokens,
        "generation_attempt_limit": 2,
        "transport_attempts": None,
        "delay_seconds": 8.0,
    }
    assert payload["run"]["corpus"]["documents"] == 2
    assert payload["run"]["corpus"]["chunks"] == 2
    assert payload["run"]["dataset"]["sha256"] == "f" * 64

    answered, refused = payload["questions"]
    assert answered["id"] == "q-api"
    assert answered["outcome"] == "answered"
    assert answered["first_relevant_rank"] is not None
    assert answered["hits"][0]["similarity"] is not None
    assert answered["citations"]["published"] == [{"document_id": "stack", "section": "API"}]
    assert answered["citations"]["verified"] == 1
    assert answered["failures"] == []
    assert refused["failures"] == ["answered_unknown"]


def test_the_export_records_the_verdict_and_the_labels_the_model_claimed():
    """What two diagnoses had to infer: what the model declared about its answer."""
    label = label_of("stack", ANSWERABLE)
    dataset = dataset_of(ANSWERABLE, UNKNOWN, UNKNOWN.model_copy(update={"id": "q-silent"}))
    llm = ScriptedLLMProvider(
        grounded(f"FastAPI [{label}].", label),
        ScriptedReply(answer="Nobody is paid [S1].", sources=("S1", "S2"), support="inferred"),
        ScriptedReply(answer="Nobody is paid [S1].", sources=("S1",), support=None),
    )
    report = run(run_e2e_evaluation(dataset, e2e_service(llm)))

    answered, inferred, silent = export_e2e(report, run_metadata(dataset))["questions"]

    assert answered["support"] == "stated"
    assert answered["citations"]["claimed_source_labels"] == [label]
    # Refused, and the artifact still says what the model pointed at.
    assert inferred["outcome"] == "not_grounded"
    assert inferred["support"] == "inferred"
    assert inferred["citations"]["claimed_source_labels"] == ["S1", "S2"]
    assert inferred["citations"]["published"] == []
    assert inferred["controlled_refusal"] is True
    # No verdict is recorded as no verdict, never as a default.
    assert silent["outcome"] == "not_grounded"
    assert silent["support"] is None
    assert silent["citations"]["claimed_source_labels"] == ["S1"]
    assert silent["citations"]["published"] == []


def test_the_export_records_what_the_grounding_check_said():
    """A refusal by the check is told apart from a refusal by the model."""
    label = label_of("stack", ANSWERABLE)
    dataset = dataset_of(ANSWERABLE, UNKNOWN, UNKNOWN.model_copy(update={"id": "q-declined"}))
    confirmed = ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label))
    refusing = ScriptedLLMProvider(
        grounded("Nobody is paid [S1].", "S1"),
        ScriptedReply(answer="", sources=(), support="none"),
        grounding_check=checked("not_supported"),
    )

    (answered,) = export_e2e(
        run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(confirmed))),
        run_metadata(dataset_of(ANSWERABLE)),
    )["questions"]
    payload = export_e2e(
        run(run_e2e_evaluation(dataset_of(*dataset.questions[1:]), e2e_service(refusing))),
        run_metadata(dataset),
    )
    stopped, declined = payload["questions"]

    assert payload["run"]["generation"]["grounding_check_version"] == GROUNDING_CHECK_VERSION
    assert answered["outcome"] == "answered"
    assert answered["grounding_check"] == "supported"
    # Stopped by the check: the model said stated and named a real label.
    assert stopped["outcome"] == "not_grounded"
    assert stopped["support"] == "stated"
    assert stopped["grounding_check"] == "not_supported"
    assert stopped["citations"]["claimed_source_labels"] == ["S1"]
    assert stopped["citations"]["published"] == []
    assert stopped["controlled_refusal"] is True
    assert stopped["failures"] == []
    # Declined by the model itself: the check was never asked.
    assert declined["grounding_check"] is None
    assert declined["grounding_check_seconds"] == 0.0


def test_the_export_says_how_many_generations_an_answer_took():
    """A retry that succeeded is otherwise invisible in a run's results."""
    label = label_of("stack", ANSWERABLE)
    cut_off = ScriptedReply(raw_text='{"answer": "FastAPI serves', finish_reason="length")
    good = grounded(f"FastAPI [{label}].", label)

    (once,) = export_e2e(
        run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(ScriptedLLMProvider(good)))),
        run_metadata(dataset_of(ANSWERABLE)),
    )["questions"]
    (twice,) = export_e2e(
        run(
            run_e2e_evaluation(
                dataset_of(ANSWERABLE), e2e_service(ScriptedLLMProvider(cut_off, good))
            )
        ),
        run_metadata(dataset_of(ANSWERABLE)),
    )["questions"]
    (failed,) = export_e2e(
        run(run_e2e_evaluation(dataset_of(ANSWERABLE), e2e_service(ScriptedLLMProvider(cut_off)))),
        run_metadata(dataset_of(ANSWERABLE)),
    )["questions"]

    assert once["outcome"] == "answered" and once["generation_attempts"] == 1
    assert twice["outcome"] == "answered" and twice["generation_attempts"] == 2
    assert twice["failures"] == []
    # Still an error, still counted as one: nothing is hidden by the retry.
    assert failed["outcome"] == "error"
    assert failed["error"]["failure_detail"] == "reply_not_json"
    assert failed["error"]["finish_reason"] == "length"
    assert failed["error"]["attempts"] == 2
    assert "pipeline_error" in failed["failures"]


def test_a_question_that_errored_has_no_verdict_in_the_export():
    dataset = dataset_of(ANSWERABLE)
    report = run(run_e2e_evaluation(dataset, e2e_service(FailingLLMProvider())))

    (errored,) = export_e2e(report, run_metadata(dataset))["questions"]

    assert errored["outcome"] == "error"
    assert errored["support"] is None
    assert errored["citations"]["claimed_source_labels"] == []


def test_the_aggregates_are_the_records_counted():
    metrics = _exported()["metrics"]

    assert metrics["retrieval"]["hit_rates"]["hit@5"] == {"correct": 1, "total": 1}
    assert metrics["answers"]["outcomes"]["answered"] == 2
    assert metrics["answers"]["answerable_answered"] == {"correct": 1, "total": 1}
    assert metrics["answers"]["answered_citing_expected_source"] == {"correct": 1, "total": 1}
    assert metrics["citations"]["validity"]["correct"] == metrics["citations"]["published"] == 2
    assert metrics["refusals"]["controlled_refusals"] == {"correct": 0, "total": 1}
    assert metrics["refusals"]["not_refused"] == ["q-salary"]
    assert metrics["robustness"]["pipeline_errors"] == 0


def test_a_broken_gate_fails_the_export_and_names_the_question():
    payload = _exported()

    assert payload["passed"] is False
    assert payload["gates"]["unanswerable_refused"] is False
    assert payload["gates"]["citations_verified"] is True
    assert payload["failures"] == {"answered_unknown": ["q-salary"]}


def test_the_export_is_plain_json_and_carries_no_passage_text(tmp_path: Path):
    text = json.dumps(_exported(), ensure_ascii=False, allow_nan=False)

    assert json.loads(text)["format"] == E2E_FORMAT_VERSION
    assert "FastAPI serves the HTTP API of the backend." not in text


def test_the_summary_says_what_the_export_says():
    payload = _exported(notes=("the index was synchronized on 2026-10-01",))

    summary = render_summary(payload)

    assert "**Gates: FAIL**" in summary
    assert "| unanswerable_refused | FAIL |" in summary
    assert "| Retrieval hit@5 | 1/1 (100.0%) |" in summary
    assert "| Unanswerable questions refused | 0/1 (0.0%) |" in summary
    assert "| `q-salary` | unknown | answered | answered_unknown |" in summary
    assert "`q-api`" not in summary.split("## Failures")[1]  # only failing questions there
    assert "the index was synchronized on 2026-10-01" in summary
    assert "What this does not measure" in summary


def _exported_with_errors() -> dict[str, Any]:
    label = label_of("stack", ANSWERABLE)
    unusable = ScriptedReply(raw_text="Sure! The answer is FastAPI.")
    dataset = dataset_of(ANSWERABLE, ANSWERABLE.model_copy(update={"id": "q-api-again"}), UNKNOWN)
    llm = ScriptedLLMProvider(grounded(f"FastAPI [{label}].", label), unusable, DECLINED)
    report = run(run_e2e_evaluation(dataset, e2e_service(llm)))
    return export_e2e(report, run_metadata(dataset))


def test_the_export_keeps_retrieval_and_availability_apart():
    payload = _exported_with_errors()
    metrics, errored = payload["metrics"], payload["questions"][1]

    assert metrics["availability"] == {
        "completed": {"correct": 2, "total": 3},
        "errors_by_category": {"unparseable_output": ["q-api-again"]},
    }
    assert metrics["retrieval"]["hit_rates"]["hit@5"] == {"correct": 2, "total": 2}
    assert metrics["retrieval"]["observed"] == {"correct": 3, "total": 3}
    assert metrics["retrieval"]["not_observed"] == []
    assert payload["failures"]["pipeline_error"] == ["q-api-again"]
    assert "retrieval_miss" not in payload["failures"]

    assert errored["outcome"] == "error"
    assert errored["retrieval_observed"] is True
    assert errored["first_relevant_rank"] is not None
    assert errored["hits"]
    assert errored["error"] == {
        "failure_category": "unparseable_output",
        "failure_detail": "reply_not_json",
        "failure_step": "generation",
        "status_code": None,
        "retryable": None,
        "attempts": 1,
        "retry_after_seconds": None,
        "finish_reason": "stop",
        "reply_characters": len("Sure! The answer is FastAPI."),
        "reply_visible_characters": len("Sure!TheanswerisFastAPI."),
        "input_tokens": None,
        "output_tokens": None,
    }
    assert payload["questions"][0]["error"] is None


def test_the_export_never_holds_the_reply_that_could_not_be_used():
    text = json.dumps(_exported_with_errors(), ensure_ascii=False)

    assert "Sure!" not in text


def test_the_summary_lists_each_error_with_its_cause():
    summary = render_summary(_exported_with_errors())

    assert "| Questions the pipeline completed | 2/3 (66.7%) |" in summary
    assert "| Questions with an observed retrieval | 3/3 (100.0%) |" in summary
    assert "| Retrieval hit@5 | 2/2 (100.0%) |" in summary
    assert "## Pipeline errors" in summary
    assert (
        "| `q-api-again` | GENERATION_UNAVAILABLE | generation | unparseable_output | "
        "reply_not_json | — | 1 | stop |"
    ) in summary


def test_the_corpus_identity_follows_the_document_fingerprints():
    same = corpus_identity(tuple(reversed(CHUNKS)))
    other = corpus_identity(CHUNKS[:1])

    assert corpus_identity(CHUNKS) == same
    assert other.sha256 != same.sha256
    assert (other.documents, other.chunks) == (1, 1)


def test_the_summary_lists_every_question_compactly_without_answer_text():
    payload = _exported()
    payload["questions"][0]["question"] = "Which | framework\nserves the API?"

    summary = render_summary(payload)

    assert "## Questions" in summary
    assert (
        "| `q-api` | Which / framework serves the API? | direct | answer | answered "
        "| 1/1 verified | — |" in summary
    )
    assert "| `q-salary` | What is the salary of the backend API author? | unknown | refuse |" in (
        summary
    )
    for question in payload["questions"]:
        assert question["answer"] not in summary.split("## Questions")[1].split("## Failures")[0]
