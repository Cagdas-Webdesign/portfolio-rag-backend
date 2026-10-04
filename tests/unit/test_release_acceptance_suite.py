"""The release-acceptance suite: 24 of the 49 questions, chosen for risk coverage.

These pin the selection and what it must keep covering, so that the suite
cannot shrink, swap out a hard case or quietly lose a kind of question. The
reasoning for each of the 49 is in `evaluation/README.md`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from portfolio_rag.evaluation import (
    EvaluationQuestion,
    EvaluationSuite,
    QuestionCategory,
    load_dataset,
    select_suite,
    suite_identity,
    suite_path,
)
from portfolio_rag.evaluation.operations import TIER_SUITES, Tier

DATASET = Path("evaluation/portfolio-questions.yaml")
SUITE = EvaluationSuite.RELEASE_ACCEPTANCE

#: The extended validation dataset, byte for byte. Selecting a suite must not
#: have changed a question, its category or its ground truth; changing the
#: dataset on purpose means updating this deliberately.
EXTENDED_SHA256 = "54afed0b6798c7ac380373cfdace6cc99c0270077504d1fc2e0a793460a8f280"

SELECTED = (
    "direct-wordpress-experience",
    "direct-abitur",
    "direct-contact",
    "paraphrased-component-ui",
    "paraphrased-premium-plugins",
    "paraphrased-edge-backend",
    "paraphrased-scroll-animation",
    "paraphrased-root-cause",
    "multi-frontend-backend-ai",
    "multi-marketing-automation-web",
    "multi-api-backend-evidence",
    "section-prompt-injection",
    "section-degree-claim",
    "section-lead-flow",
    "ambiguous-frontend-technologies",
    "ambiguous-api-frontend",
    "unknown-phone-number",
    "unknown-major-clients",
    "unknown-kubernetes-production",
    "unknown-medical-training",
    "broad-deployment",
    "broad-ai-technologies",
    "broad-frontend-backend-in-portfolio",
    "broad-project-scope",
)

#: Questions the evaluation history shows to be hard for a real provider or
#: for retrieval. Removing one would make the suite easier, not smaller.
HISTORICAL_PROBLEM_CASES = {
    "broad-deployment",
    "broad-frontend-backend-in-portfolio",
    "broad-ai-technologies",
    "ambiguous-frontend-technologies",
    "multi-api-backend-evidence",
    "section-lead-flow",
    "unknown-kubernetes-production",
    "paraphrased-scroll-animation",
    "broad-project-scope",
}


def _suite() -> tuple[str, ...]:
    full = load_dataset(DATASET)
    return tuple(question.id for question in select_suite(DATASET, full, SUITE).questions)


def test_the_suite_is_exactly_these_24_questions():
    ids = _suite()

    assert len(ids) == 24
    assert len(set(ids)) == 24
    assert set(ids) == set(SELECTED)


def test_every_selected_id_exists_and_nothing_is_duplicated_in_the_file():
    full = load_dataset(DATASET)
    known = {question.id for question in full.questions}
    raw = suite_path(DATASET, SUITE).read_text(encoding="utf-8")

    assert set(SELECTED) <= known
    for question_id in SELECTED:
        assert raw.count(f"  - {question_id}\n") == 1, question_id


def test_the_suite_is_versioned_and_hashable():
    full = load_dataset(DATASET)
    identity = suite_identity(DATASET, full, SUITE)
    path = Path("evaluation/portfolio-questions.release-acceptance.yaml")

    assert identity.version == 1
    assert identity.path == path.as_posix()
    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity == suite_identity(DATASET, full, SUITE)  # deterministic


def test_the_acceptance_tier_asks_the_release_suite_and_nothing_else():
    assert TIER_SUITES[Tier.ACCEPTANCE] is EvaluationSuite.RELEASE_ACCEPTANCE
    assert TIER_SUITES[Tier.SMOKE] is EvaluationSuite.PROVIDER_SMOKE
    assert EvaluationSuite.FULL not in TIER_SUITES.values()
    assert EvaluationSuite.RELEASE_ACCEPTANCE.value == "release-acceptance"


def test_the_extended_validation_suite_stays_whole_and_unchanged():
    full = load_dataset(DATASET)

    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == EXTENDED_SHA256
    assert len(full.questions) == 49
    assert select_suite(DATASET, full, EvaluationSuite.FULL) == full


def test_a_selected_question_is_the_dataset_question_itself():
    full = load_dataset(DATASET)
    by_id = {question.id: question for question in full.questions}

    for question in select_suite(DATASET, full, SUITE).questions:
        assert question == by_id[question.id]


def test_every_historical_problem_case_is_kept():
    assert set(_suite()) >= HISTORICAL_PROBLEM_CASES


def test_the_kinds_of_question_are_all_covered_more_than_once():
    full = load_dataset(DATASET)
    selected = select_suite(DATASET, full, SUITE).questions
    categories = [question.category for question in selected]
    ids = [question.id for question in selected]

    assert categories.count(QuestionCategory.UNKNOWN) >= 3
    assert categories.count(QuestionCategory.AMBIGUOUS) >= 2
    assert categories.count(QuestionCategory.MULTI_SOURCE) >= 4
    assert categories.count(QuestionCategory.PARAPHRASED) >= 3
    assert categories.count(QuestionCategory.SECTION) >= 2
    assert categories.count(QuestionCategory.DIRECT) >= 2
    assert sum(1 for question_id in ids if question_id.startswith("broad-")) >= 4
    # Mostly real use: the answerable questions are the large majority.
    assert sum(1 for question in selected if question.expects_an_answer) >= 18


def test_multi_source_questions_are_kept():
    full = load_dataset(DATASET)
    selected = select_suite(DATASET, full, SUITE).questions

    multi = [q for q in selected if len({e.document for e in q.expected_sources}) >= 2]
    assert len(multi) >= 8


def test_every_knowledge_document_is_still_asked_about():
    full = load_dataset(DATASET)
    selected = select_suite(DATASET, full, SUITE).questions

    def documents(questions: Sequence[EvaluationQuestion]) -> set[str]:
        return {e.document for q in questions for e in q.expected_sources}

    assert documents(selected) == documents(full.questions)
    assert len(documents(selected)) == 12
