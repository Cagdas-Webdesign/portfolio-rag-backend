"""The size budget and its invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from portfolio_rag.ingestion.chunking import (
    DEFAULT_CHUNKING_POLICY,
    MARKDOWN_CHUNKING_STRATEGY_VERSION,
    ChunkingPolicy,
)


def test_the_defaults_are_the_documented_starting_values():
    policy = ChunkingPolicy()

    assert policy.target_chars == 1200
    assert policy.max_chars == 1800
    assert policy.overlap_chars == 150
    assert policy == DEFAULT_CHUNKING_POLICY


def test_the_strategy_version_is_defined_once_and_is_stable():
    assert MARKDOWN_CHUNKING_STRATEGY_VERSION == "markdown-structure-v1"


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"target_chars": 0}, "a zero target chunks nothing"),
        ({"target_chars": -1}, "negative sizes are meaningless"),
        ({"max_chars": 0}, "a zero ceiling admits no chunk"),
        ({"max_chars": -5}, "negative ceiling"),
        ({"target_chars": 1000, "max_chars": 900}, "ceiling below the goal"),
        ({"overlap_chars": -1}, "negative overlap"),
        ({"target_chars": 100, "max_chars": 200, "overlap_chars": 100}, "overlap equals target"),
        ({"target_chars": 100, "max_chars": 200, "overlap_chars": 150}, "overlap above target"),
    ],
)
def test_incoherent_budgets_are_rejected(overrides: dict[str, int], why: str):
    with pytest.raises(ValidationError):
        ChunkingPolicy(**overrides)


def test_a_ceiling_equal_to_the_target_is_allowed():
    """`max == target` just means "never exceed the goal", which is coherent."""
    assert ChunkingPolicy(target_chars=500, max_chars=500, overlap_chars=0).max_chars == 500


def test_overlap_may_be_switched_off():
    assert ChunkingPolicy(overlap_chars=0).overlap_chars == 0


def test_unknown_settings_are_rejected():
    with pytest.raises(ValidationError):
        ChunkingPolicy(target_tokens=1200)  # type: ignore[call-arg]


def test_a_policy_is_immutable():
    policy = ChunkingPolicy()

    with pytest.raises(ValidationError):
        policy.target_chars = 900  # type: ignore[misc]


def test_policies_compare_by_value():
    assert ChunkingPolicy(target_chars=900) == ChunkingPolicy(target_chars=900)
    assert ChunkingPolicy(target_chars=900) != ChunkingPolicy(target_chars=800)


def test_the_policy_describes_itself_for_developer_output():
    described = dict(ChunkingPolicy().describe())

    assert described["strategy"] == MARKDOWN_CHUNKING_STRATEGY_VERSION
    assert described["target chars"] == "1200"
    assert described["max chars"] == "1800"
    assert described["overlap chars"] == "150"
