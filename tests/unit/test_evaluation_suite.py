"""Full and smoke: two runs over one dataset, with the questions stored once.

The 49-question real-corpus dataset is the baseline and must stay whole. The
smoke suite is a file of ids beside it, so it can pick questions but never
restate them. These tests read the real files — they are what a run uses.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

from portfolio_rag.evaluation import (
    EvaluationDatasetError,
    EvaluationSuite,
    QuestionCategory,
    load_dataset,
    select_suite,
    suite_path,
)

DATASET_PATH = Path("evaluation/portfolio-questions.yaml")
SMOKE_PATH = Path("evaluation/portfolio-questions.smoke.yaml")


@pytest.fixture(scope="module")
def full():
    return load_dataset(DATASET_PATH)


@pytest.fixture(scope="module")
def smoke(full):
    return select_suite(DATASET_PATH, full, EvaluationSuite.SMOKE)


def test_the_full_suite_is_the_whole_dataset_unchanged(full):
    """Case A."""
    selected = select_suite(DATASET_PATH, full, EvaluationSuite.FULL)

    assert selected is full
    assert len(selected.questions) == 49


def test_the_smoke_suite_has_twelve_questions(smoke):
    """Case B."""
    assert len(smoke.questions) == 12


def test_the_smoke_suite_has_two_questions_per_category(smoke):
    """Case C."""
    counts = Counter(question.category for question in smoke.questions)

    assert counts == {
        QuestionCategory.DIRECT: 2,
        QuestionCategory.SECTION: 2,
        QuestionCategory.MULTI_SOURCE: 2,
        QuestionCategory.PARAPHRASED: 2,
        QuestionCategory.AMBIGUOUS: 2,
        QuestionCategory.UNKNOWN: 2,
    }


def test_the_smoke_selection_is_deterministic(full, smoke):
    """Case D: the same ids, in the same order, every time."""
    again = select_suite(DATASET_PATH, full, EvaluationSuite.SMOKE)

    assert [q.id for q in again.questions] == [q.id for q in smoke.questions]


def test_the_smoke_selection_does_not_depend_on_file_order(full, tmp_path: Path):
    """Reversing the id file selects the same questions, in dataset order."""
    raw = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))
    raw["questions"] = list(reversed(raw["questions"]))
    dataset_copy = tmp_path / DATASET_PATH.name
    dataset_copy.write_bytes(DATASET_PATH.read_bytes())
    suite_path(dataset_copy, EvaluationSuite.SMOKE).write_text(yaml.safe_dump(raw))

    reversed_ = select_suite(dataset_copy, full, EvaluationSuite.SMOKE)
    forward = select_suite(DATASET_PATH, full, EvaluationSuite.SMOKE)

    assert [q.id for q in reversed_.questions] == [q.id for q in forward.questions]


def test_smoke_questions_are_the_full_datasets_own_objects(full, smoke):
    """Case E: selected, not copied — question text and ground truth exist once."""
    by_id = {question.id: question for question in full.questions}

    for question in smoke.questions:
        assert question is by_id[question.id]


def test_the_smoke_file_holds_ids_and_nothing_else():
    """Case E: no question text, no expected sources in the suite file."""
    raw = yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8"))

    assert set(raw) == {"version", "dataset", "questions"}
    assert all(isinstance(entry, str) for entry in raw["questions"])


def test_the_smoke_suite_keeps_the_datasets_identity(full, smoke):
    assert (smoke.version, smoke.corpus) == (full.version, full.corpus)


def test_the_smoke_file_sits_beside_its_dataset():
    assert suite_path(DATASET_PATH, EvaluationSuite.SMOKE) == SMOKE_PATH


# --- a suite that does not match its dataset is refused ------------------------


def _write_suite(tmp_path: Path, **fields: object) -> Path:
    dataset_copy = tmp_path / DATASET_PATH.name
    dataset_copy.write_bytes(DATASET_PATH.read_bytes())
    payload = {"version": 1, "dataset": DATASET_PATH.name, "questions": ["direct-languages"]}
    payload.update(fields)
    suite_path(dataset_copy, EvaluationSuite.SMOKE).write_text(yaml.safe_dump(payload))
    return dataset_copy


def test_an_unknown_id_is_an_error_not_a_skip(full, tmp_path: Path):
    dataset_copy = _write_suite(tmp_path, questions=["direct-languages", "no-such-question"])

    with pytest.raises(EvaluationDatasetError, match="no-such-question"):
        select_suite(dataset_copy, full, EvaluationSuite.SMOKE)


def test_a_duplicate_id_is_an_error(full, tmp_path: Path):
    dataset_copy = _write_suite(tmp_path, questions=["direct-languages", "direct-languages"])

    with pytest.raises(EvaluationDatasetError, match="Duplicate"):
        select_suite(dataset_copy, full, EvaluationSuite.SMOKE)


def test_a_suite_written_for_another_dataset_is_refused(full, tmp_path: Path):
    dataset_copy = _write_suite(tmp_path, dataset="questions.yaml")

    with pytest.raises(EvaluationDatasetError, match=r"questions\.yaml"):
        select_suite(dataset_copy, full, EvaluationSuite.SMOKE)


def test_a_suite_file_that_does_not_exist_is_an_error(tmp_path: Path):
    fixture_dataset = Path("evaluation/questions.yaml")

    with pytest.raises(EvaluationDatasetError, match="could not be read"):
        select_suite(fixture_dataset, load_dataset(fixture_dataset), EvaluationSuite.SMOKE)


def test_a_suite_file_with_extra_fields_is_refused(full, tmp_path: Path):
    """Extra fields are how question text would creep in; the schema forbids them."""
    dataset_copy = _write_suite(tmp_path, expected_sources=[{"document": "x"}])

    with pytest.raises(EvaluationDatasetError, match="invalid"):
        select_suite(dataset_copy, full, EvaluationSuite.SMOKE)
