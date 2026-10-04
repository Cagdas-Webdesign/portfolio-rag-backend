"""An answer is published only when the model calls it stated — and proves it.

Prompt version 6 asks the model how the answer it wrote relates to the passages
it was given: ``stated``, ``inferred`` or ``none``. Versions 4 and 5 asked
whether the *question* was answerable, and a model that believed "none" was the
helpful answer to a question with a false premise said yes, twice, with a real
label attached.

What is tested here is the backend's half of that contract: what it does with
each thing a model can say. Whether a model *classifies* an answer correctly is
a property of the model and is measured by the end-to-end evaluation; a
scripted reply cannot stand in for it, and these tests do not pretend to.

The routing is deliberately one-sided. ``stated`` earns nothing — the answer
still has to cite something real. Everything else, a missing verdict included,
publishes nothing.

The passages are neutral fixtures shaped like the measured cases: a project
that deliberately does not use a technology, a project that does use one, and
studies that a passage explicitly says were not completed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from portfolio_rag.application.indexing import IndexingService
from portfolio_rag.domain.embedding import VectorIndexSpec
from portfolio_rag.evaluation import (
    QuestionCategory,
    load_dataset,
    resolve_corpus,
    run_grounding_evaluation,
)
from portfolio_rag.infrastructure.knowledge import InMemoryChunkResolver
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ingestion import load_knowledge_base
from portfolio_rag.ingestion.chunking import chunk_knowledge_base
from portfolio_rag.ports.llm import GenerationRequest, GenerationResponse, MessageRole
from portfolio_rag.rag.errors import GenerationUnavailableError
from portfolio_rag.rag.generation import SupportVerdict, parse_generation
from portfolio_rag.rag.language import AnswerLanguage, insufficient_knowledge_answer
from portfolio_rag.rag.policy import ContextPolicy
from portfolio_rag.rag.retrieval import PublicRetrievalService
from portfolio_rag.rag.service import AnswerOutcome, GroundedAnswer, GroundedAnswerService
from tests.doubles import (
    LexicalEmbeddingProvider,
    ScriptedLLMProvider,
    ScriptedReply,
    checked,
    is_grounding_check,
    make_chunk,
)
from tests.support import run
from tests.unit.test_answer_service import build_service

PROJECT_LIMITS = make_chunk(
    "project--0000",
    "Das Projekt setzt bewusst kein Kubernetes, kein Redis und keine Queues ein, "
    "solange der Anwendungsfall sie nicht braucht.",
    document_id="project",
    title="Projekt",
    heading_path=("Bewusste Grenzen",),
    language="de",
)
PROJECT_STACK = make_chunk(
    "project--0001",
    "Alex hat das Backend des Projekts mit FastAPI umgesetzt.",
    document_id="project",
    title="Projekt",
    heading_path=("Technologie-Stack",),
    language="de",
    ordinal=1,
)
SCHOOLING = make_chunk(
    "education--0000",
    "Alex hat das Abitur abgeschlossen.",
    document_id="education",
    title="Ausbildung",
    heading_path=("Schulbildung",),
    language="de",
)
STUDIES = make_chunk(
    "education--0001",
    "Alex hat fünf Semester Informatik studiert. Das Studium wurde nicht als "
    "abgeschlossenes Hochschulstudium fortgeführt.",
    document_id="education",
    title="Ausbildung",
    heading_path=("Studium",),
    language="de",
    ordinal=1,
)
CORPUS: list[object] = [PROJECT_LIMITS, PROJECT_STACK, SCHOOLING, STUDIES]

GERMAN_REFUSAL = insufficient_knowledge_answer(AnswerLanguage.GERMAN)


def _reply(**fields: Any) -> ScriptedReply:
    return ScriptedReply(raw_text=json.dumps(fields))


def _ask(question: str, reply: ScriptedReply) -> GroundedAnswer:
    service, _ = build_service(CORPUS, llm=ScriptedLLMProvider(reply))
    return run(service.answer(question))


def _label_of(question: str, chunk: Any) -> str:
    """The label *chunk* gets in the context built for *question*."""
    service, _ = build_service(CORPUS, llm=ScriptedLLMProvider())
    context = run(service.answer(question)).context
    assert context is not None
    return next(source.label for source in context.sources if source.chunk_id == chunk.id)


def _assert_controlled_refusal(answer: GroundedAnswer) -> None:
    assert answer.outcome is AnswerOutcome.NOT_GROUNDED
    assert answer.answer == GERMAN_REFUSAL
    assert answer.citations == ()


# --- reading the verdict -------------------------------------------------------


@pytest.mark.parametrize("verdict", list(SupportVerdict))
def test_each_verdict_is_read_from_the_reply(verdict: SupportVerdict):
    draft = parse_generation(json.dumps({"answer": "x", "sources": ["S1"], "support": verdict}))

    assert draft.support is verdict


@pytest.mark.parametrize("written", ["Stated", " stated ", "STATED"])
def test_case_and_blanks_around_a_verdict_are_forgiven(written: str):
    draft = parse_generation(json.dumps({"answer": "x", "sources": [], "support": written}))

    assert draft.support is SupportVerdict.STATED


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param('{"answer": "x", "sources": ["S1"]}', id="missing"),
        pytest.param('{"answer": "x", "sources": ["S1"], "support": null}', id="null"),
        pytest.param('{"answerable": true, "answer": "x", "sources": ["S1"]}', id="old contract"),
    ],
)
def test_a_reply_without_a_verdict_carries_none(reply: str):
    """No verdict is not a favourable verdict — and not a parse error either."""
    assert parse_generation(reply).support is None


@pytest.mark.parametrize(
    "value",
    ['"yes"', '"true"', '"grounded"', '"stated, mostly"', '""', "true", "1", '["stated"]', "{}"],
)
def test_a_verdict_that_is_not_one_of_the_three_is_a_broken_reply(value: str):
    """Guessing which way an unknown word was meant is how a refusal gets published."""
    with pytest.raises(GenerationUnavailableError) as caught:
        parse_generation(f'{{"answer": "x", "sources": ["S1"], "support": {value}}}')

    assert caught.value.failure is not None
    assert caught.value.failure.detail == "support_not_recognized"


# --- routing -------------------------------------------------------------------

LIMITS_QUESTION = "Was setzt das Projekt bewusst nicht ein?"


def test_stated_with_a_real_citation_is_published():
    label = _label_of(LIMITS_QUESTION, PROJECT_LIMITS)

    answer = _ask(
        LIMITS_QUESTION,
        _reply(answer=f"Kein Kubernetes [{label}].", sources=[label], support="stated"),
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert [(c.document_id, c.section) for c in answer.citations] == [
        ("project", "Bewusste Grenzen")
    ]
    assert answer.support is SupportVerdict.STATED


def test_inferred_is_refused_however_real_its_citation():
    """The measured failure, classified honestly: a real label, and no answer."""
    answer = _ask(
        "Welche Kubernetes-Produktionssysteme hat Alex betrieben?",
        _reply(
            answer="Alex hat keine Kubernetes-Produktionssysteme betrieben [S1].",
            sources=["S1"],
            support="inferred",
        ),
    )

    _assert_controlled_refusal(answer)
    assert "Kubernetes" not in answer.answer
    assert answer.unknown_labels == ()


def test_none_is_refused():
    answer = _ask(
        "Welche medizinische Ausbildung hat Alex?",
        _reply(answer="", sources=[], support="none"),
    )

    assert answer.retrieved_count > 0
    _assert_controlled_refusal(answer)


def test_none_is_refused_even_with_the_neighbours_cited():
    """A refusal that names what it read is still a refusal."""
    answer = _ask(
        "Welche medizinische Ausbildung hat Alex?",
        _reply(
            answer="Eine medizinische Ausbildung wird nicht erwähnt. Die Ausbildung umfasst "
            "das Abitur und ein Studium [S1][S2], jedoch keine medizinischen Qualifikationen.",
            sources=["S1", "S2"],
            support="none",
        ),
    )

    _assert_controlled_refusal(answer)
    assert "medizinisch" not in answer.answer
    assert "Abitur" not in answer.answer


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({}, id="missing"),
        pytest.param({"support": None}, id="null"),
        pytest.param({"answerable": True}, id="old contract, answerable true"),
    ],
)
def test_no_verdict_fails_closed(fields: dict[str, Any]):
    """An answer and a real label, and nothing published: the verdict is required."""
    answer = _ask(
        LIMITS_QUESTION,
        _reply(answer="Kein Kubernetes [S1].", sources=["S1"], **fields),
    )

    _assert_controlled_refusal(answer)
    assert answer.support is None


def test_an_unrecognized_verdict_is_an_unusable_generation():
    service, _ = build_service(
        CORPUS,
        llm=ScriptedLLMProvider(
            _reply(answer="Kein Kubernetes [S1].", sources=["S1"], support="probably")
        ),
    )

    with pytest.raises(GenerationUnavailableError) as caught:
        run(service.answer(LIMITS_QUESTION))

    assert caught.value.failure is not None
    assert caught.value.failure.detail == "support_not_recognized"


# --- the class of mistake, not the topic -------------------------------------------

#: One reply per way a passage gets stretched past what it says. Each is the
#: shape of the measured failure: a confident denial with a real label
#: attached, and the verdict that says how it was arrived at.
STRETCHED_DENIALS = {
    "a project's scope is not the person's": (
        "Hat Alex jemals mit Redis gearbeitet?",
        "Nein, Alex hat nie mit Redis gearbeitet [S1].",
    ),
    "current use is not past experience": (
        "Hat das Projekt früher einmal Queues eingesetzt?",
        "Nein, das Projekt hat nie Queues eingesetzt [S1].",
    ),
    "a premise in the question is not a fact": (
        "Seit wann betreibt Alex Redis-Cluster in Produktion?",
        "Alex betreibt keine Redis-Cluster; Redis wird bewusst nicht eingesetzt [S1].",
    ),
    "silence is not a negative": (
        "Hat Alex eine Promotion abgeschlossen?",
        "Nein, Alex hat keine Promotion abgeschlossen [S1][S2].",
    ),
}


@pytest.mark.parametrize(
    ("question", "denial"), STRETCHED_DENIALS.values(), ids=STRETCHED_DENIALS.keys()
)
def test_a_denial_the_passages_do_not_carry_is_not_published(question: str, denial: str):
    """Declared inferred, it is refused: no sentence of it, no source of it."""
    answer = _ask(question, _reply(answer=denial, sources=["S1", "S2"], support="inferred"))

    assert answer.retrieved_count > 0
    _assert_controlled_refusal(answer)
    assert answer.unknown_labels == ()
    assert denial not in answer.answer


# --- what must keep working ------------------------------------------------------


def test_a_negative_the_corpus_states_is_still_an_answer():
    """Not every "no" is an inference: this one is written in the passage."""
    question = "Hat Alex einen abgeschlossenen Hochschulabschluss?"
    label = _label_of(question, STUDIES)

    answer = _ask(
        question,
        _reply(
            answer=f"Nein. Das Studium wurde nicht als abgeschlossenes Hochschulstudium "
            f"fortgeführt [{label}].",
            sources=[label],
            support="stated",
        ),
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert answer.answer.startswith("Nein.")
    assert [(c.document_id, c.section) for c in answer.citations] == [("education", "Studium")]


def test_a_project_passage_carries_a_question_about_the_person_when_it_states_the_answer():
    """The verdict is about the answer and the passage, never about the document type."""
    question = "Welche Technologien nutzt Alex?"
    label = _label_of(question, PROJECT_STACK)

    answer = _ask(
        question,
        _reply(
            answer=f"Alex hat das Backend des Projekts mit FastAPI umgesetzt [{label}].",
            sources=[label],
            support="stated",
        ),
    )

    assert answer.outcome is AnswerOutcome.ANSWERED
    assert "FastAPI" in answer.answer
    assert [(c.document_id, c.section) for c in answer.citations] == [
        ("project", "Technologie-Stack")
    ]


def test_stated_earns_nothing_the_citation_checks_did_not_already_require():
    """The existing checks, untouched: a real label, or nothing is published."""
    unverifiable = _ask(
        LIMITS_QUESTION,
        _reply(answer="Kein Kubernetes [S9].", sources=["S9"], support="stated"),
    )
    uncited = _ask(
        LIMITS_QUESTION,
        _reply(answer="Kein Kubernetes.", sources=[], support="stated"),
    )
    blank = _ask(
        LIMITS_QUESTION,
        _reply(answer="", sources=["S1"], support="stated"),
    )
    named = _ask(
        LIMITS_QUESTION,
        _reply(answer="Kein Kubernetes.", sources=["knowledge/project.md"], support="stated"),
    )

    _assert_controlled_refusal(unverifiable)
    assert unverifiable.unknown_labels == ("S9",)
    _assert_controlled_refusal(uncited)
    _assert_controlled_refusal(blank)
    _assert_controlled_refusal(named)
    assert named.unknown_labels == ("knowledge/project.md",)


# --- what a run can say afterwards --------------------------------------------------


def test_the_verdict_and_the_claimed_labels_travel_with_the_result():
    """Diagnostics: what the model declared, including on what was not published."""
    refused = _ask(
        "Hat Alex jemals mit Redis gearbeitet?",
        _reply(answer="Nein [S1].", sources=["S1", "S2"], support="inferred"),
    )
    answered = _ask(
        LIMITS_QUESTION,
        _reply(answer="Kein Kubernetes [S1].", sources=["S1"], support="stated"),
    )

    assert refused.support is SupportVerdict.INFERRED
    assert refused.claimed_labels == ("S1", "S2")
    assert refused.citations == ()
    assert answered.support is SupportVerdict.STATED
    assert answered.claimed_labels == ("S1",)


def test_a_question_no_model_was_asked_about_has_no_verdict():
    service, llm = build_service([], llm=ScriptedLLMProvider())

    answer = run(service.answer("Welche medizinische Ausbildung hat Alex?"))

    assert answer.outcome is AnswerOutcome.NO_KNOWLEDGE
    assert llm.call_count == 0
    assert answer.support is None
    assert answer.claimed_labels == ()


# --- across a whole dataset --------------------------------------------------------

DATASET_PATH = Path("evaluation/questions.yaml")
_LABEL = re.compile(r"\[SOURCE (S\d+)\]")


class _FlaggingProvider:
    """Answers with the v6 contract: a verdict on every reply, sources on every reply.

    Questions listed in *declines* are refused the way the measured replies
    were — with real labels attached — so that the refusal has to come from the
    verdict and cannot come from an empty source list.
    """

    def __init__(self, declines: tuple[str, ...]) -> None:
        self._declines = declines

    @property
    def model(self) -> str:
        return "flagging-test-model"

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        if is_grounding_check(request):
            return GenerationResponse(
                text=checked("supported").render(), model=self.model, finish_reason="stop"
            )
        user = "\n".join(m.content for m in request.messages if m.role is MessageRole.USER)
        question = user.rpartition("QUESTION\n")[2].strip()
        labels = _LABEL.findall(user)
        declined = any(text in question for text in self._declines)
        payload = {
            "answer": "Not covered, but related passages exist."
            if declined
            else "A scripted grounded answer.",
            "sources": labels[:2] if declined else labels[:1],
            "support": "inferred" if declined else "stated",
        }
        return GenerationResponse(text=json.dumps(payload), model=self.model, finish_reason="stop")


def test_refusing_what_is_not_covered_does_not_refuse_what_is():
    """Every answerable category is still answered; every unanswerable one is refused."""
    dataset = load_dataset(DATASET_PATH)
    chunks = chunk_knowledge_base(load_knowledge_base(resolve_corpus(DATASET_PATH, dataset)))
    embeddings = LexicalEmbeddingProvider()
    store = InMemoryVectorStore(VectorIndexSpec(embedding=embeddings.spec))
    run(IndexingService(embeddings, store).synchronize(chunks))
    retrieval = PublicRetrievalService(
        embeddings=embeddings, store=store, resolver=InMemoryChunkResolver(chunks)
    )
    declines = tuple(q.question for q in dataset.questions if q.must_retrieve_nothing)
    service = GroundedAnswerService(
        retrieval=retrieval, llm=_FlaggingProvider(declines), context_policy=ContextPolicy()
    )

    report = run(run_grounding_evaluation(dataset, service))

    for category in (
        QuestionCategory.DIRECT,
        QuestionCategory.SECTION,
        QuestionCategory.MULTI_SOURCE,
        QuestionCategory.PARAPHRASED,
        QuestionCategory.AMBIGUOUS,
    ):
        records = [r for r in report.records if r.question.category is category]
        assert records, category
        assert all(r.outcome is AnswerOutcome.ANSWERED and r.citation_count > 0 for r in records)

    refused = [r for r in report.records if r.question.must_retrieve_nothing]
    assert refused
    assert all(r.outcome is AnswerOutcome.NOT_GROUNDED and r.citation_count == 0 for r in refused)
    assert report.failures == ()
