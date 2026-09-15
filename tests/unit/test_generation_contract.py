"""Parsing what a model claimed, without believing any of it yet."""

from __future__ import annotations

import json

import pytest

from portfolio_rag.rag.errors import GenerationUnavailableError
from portfolio_rag.rag.generation import parse_generation


def reply(answer: str = "An answer.", sources: list[str] | None = None) -> str:
    return json.dumps({"answer": answer, "sources": sources if sources is not None else []})


def test_a_well_formed_reply_yields_the_answer_and_its_claims():
    draft = parse_generation(reply("The service uses FastAPI.", ["S1", "S2"]))

    assert draft.answer == "The service uses FastAPI."
    assert draft.source_labels == ("S1", "S2")


def test_an_answer_without_sources_is_still_an_answer():
    """Whether it may be published is decided later, not here."""
    draft = parse_generation(reply("I cannot tell from the passages.", []))

    assert draft.answer == "I cannot tell from the passages."
    assert draft.source_labels == ()


def test_a_missing_sources_field_means_no_claims():
    draft = parse_generation('{"answer": "Something."}')

    assert draft.source_labels == ()


def test_surrounding_whitespace_is_trimmed_from_the_answer():
    assert parse_generation(reply("  Padded.  ")).answer == "Padded."


def test_a_fenced_json_block_is_still_json():
    """Common enough to be worth forgiving, unambiguous enough to be safe."""
    draft = parse_generation('```json\n{"answer": "Fenced.", "sources": ["S1"]}\n```')

    assert draft.answer == "Fenced."
    assert draft.source_labels == ("S1",)


@pytest.mark.parametrize(
    "label",
    ["S1", "s1", " S1 ", "[S1]"],
)
def test_obvious_spelling_variants_of_a_label_are_the_same_claim(label: str):
    assert parse_generation(reply(sources=[label])).source_labels == ("S1",)


def test_a_label_shaped_like_nothing_in_the_context_is_kept_for_validation():
    """Dropping it here would make a badly behaved model look well behaved."""
    draft = parse_generation(reply(sources=["docs/secrets.md", "S1"]))

    assert draft.source_labels == ("docs/secrets.md", "S1")


def test_a_duplicate_claim_is_preserved_for_the_validator_to_decide_on():
    assert parse_generation(reply(sources=["S1", "S1"])).source_labels == ("S1", "S1")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "The service uses FastAPI.",  # prose, not the contract
        "{not json",
        "[]",
        '"a string"',
        "null",
        "123",
    ],
)
def test_a_reply_that_is_not_the_requested_object_is_a_failed_generation(text: str):
    with pytest.raises(GenerationUnavailableError):
        parse_generation(text)


@pytest.mark.parametrize("answer", ["", "   ", "\n"])
def test_an_empty_answer_parses_to_an_empty_draft(answer: str):
    """Blank is a decision, not an outage.

    The provider was reachable and returned the requested object; it simply had
    nothing to say. Raising here would publish a 503 for what is really a
    knowledge gap. The service turns this into a refusal — see
    `test_answer_service.py`.
    """
    draft = parse_generation(reply(answer))

    assert draft.answer == ""


@pytest.mark.parametrize("payload", ['{"answer": null}', '{"sources": ["S1"]}'])
def test_a_missing_or_null_answer_is_an_empty_answer(payload: str):
    """Absent and null say the same thing as blank, and are treated the same."""
    draft = parse_generation(payload)

    assert draft.answer == ""


@pytest.mark.parametrize("payload", ['{"answer": 42}', '{"answer": ["a"]}', '{"answer": {}}'])
def test_an_answer_of_the_wrong_type_is_still_a_failed_generation(payload: str):
    """A number is not a blank answer — it is a reply that broke the contract."""
    with pytest.raises(GenerationUnavailableError):
        parse_generation(payload)


@pytest.mark.parametrize(
    "payload",
    ['{"answer": "a", "sources": "S1"}', '{"answer": "a", "sources": [1, 2]}'],
)
def test_a_malformed_sources_field_is_a_failed_generation(payload: str):
    with pytest.raises(GenerationUnavailableError):
        parse_generation(payload)


def test_the_error_carries_nothing_from_the_provider():
    """A failed parse must not turn the provider's body into an error message."""
    with pytest.raises(GenerationUnavailableError) as caught:
        parse_generation('{"answer": 42, "debug": "sk-live-not-a-real-token"}')

    assert "sk-live" not in caught.value.message
    assert caught.value.message == "Answer generation is temporarily unavailable."


def test_unexpected_extra_fields_are_ignored_rather_than_fatal():
    """A provider adding a field is not a reason to fail a working answer."""
    draft = parse_generation('{"answer": "Fine.", "sources": ["S1"], "confidence": 0.9}')

    assert draft.answer == "Fine."
