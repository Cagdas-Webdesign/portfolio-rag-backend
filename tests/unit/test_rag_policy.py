"""Retrieval and context policies, and the token estimate they are counted in."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from portfolio_rag.domain.knowledge import KnowledgeChunk
from portfolio_rag.domain.retrieval import RetrievedChunk
from portfolio_rag.ingestion import collect_knowledge_base
from portfolio_rag.ingestion.chunking import ChunkingPolicy, chunk_knowledge_base
from portfolio_rag.rag.context import GroundedContext, build_context
from portfolio_rag.rag.policy import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_RETRIEVAL_POLICY,
    DEFAULT_TOP_K,
    ContextPolicy,
    RetrievalPolicy,
)
from portfolio_rag.rag.prompt import available_context_tokens
from portfolio_rag.rag.tokens import CHARS_PER_TOKEN, estimate_tokens
from tests.doubles import make_chunk


def test_the_defaults_are_defined_once_and_used_by_the_default_policy():
    assert DEFAULT_RETRIEVAL_POLICY.top_k == DEFAULT_TOP_K
    assert DEFAULT_RETRIEVAL_POLICY.min_similarity == DEFAULT_MIN_SIMILARITY


def test_top_k_is_the_measured_value():
    """Seven, from the 2026-10-04 sweep recorded on the constant. A different
    value needs a new measurement, not an edit."""
    assert DEFAULT_TOP_K == 7


def _context_of(chunks: list[KnowledgeChunk]) -> GroundedContext:
    question = "Welche Technologien setzt er konkret ein?"
    return build_context(
        tuple(RetrievedChunk(chunk=chunk, similarity=0.9) for chunk in chunks),
        available_tokens=available_context_tokens(question, DEFAULT_CONTEXT_POLICY),
    )


def test_the_default_window_fits_the_budget_with_the_largest_passages_of_the_corpus():
    """The seven largest passages the knowledge base has — the worst case a
    question can retrieve today — all fit. A corpus that outgrows this shows
    here, before passages start being left out in production."""
    result = collect_knowledge_base(Path("knowledge"))
    chunked = chunk_knowledge_base(result.documents)
    largest = sorted(chunked, key=lambda c: len(c.content), reverse=True)

    context = _context_of(largest[:DEFAULT_TOP_K])

    assert len(context.sources) == DEFAULT_TOP_K
    assert context.skipped_for_budget == 0


def test_at_the_size_limit_a_passage_that_does_not_fit_is_left_out_and_counted():
    """Seven passages at the chunking maximum exceed the budget. The excess is
    skipped whole and counted — never cut to fit."""
    size = ChunkingPolicy().max_chars
    chunks = [
        make_chunk(f"doc-{i}--0000", f"{i} " + "x" * (size - 2), document_id=f"doc-{i}")
        for i in range(DEFAULT_TOP_K)
    ]

    context = _context_of(chunks)

    assert len(context.sources) + context.skipped_for_budget == DEFAULT_TOP_K
    assert context.skipped_for_budget >= 1
    assert all(len(s.retrieved.chunk.content) == size for s in context.sources)


def test_a_policy_is_frozen_so_it_cannot_be_edited_mid_request():
    with pytest.raises(ValidationError):
        DEFAULT_RETRIEVAL_POLICY.top_k = 99  # type: ignore[misc]


@pytest.mark.parametrize("top_k", [0, -1, 51])
def test_an_unusable_top_k_is_refused(top_k: int):
    with pytest.raises(ValidationError):
        RetrievalPolicy(top_k=top_k)


@pytest.mark.parametrize("similarity", [-1.5, 1.5])
def test_a_similarity_outside_the_cosine_range_is_refused(similarity: float):
    with pytest.raises(ValidationError):
        RetrievalPolicy(min_similarity=similarity)


def test_visibility_is_not_a_policy_field():
    """It is a boundary in the retrieval service, not a knob a caller can turn."""
    assert "visibility" not in RetrievalPolicy.model_fields


def test_the_policy_description_says_similarity_not_confidence():
    rendered = dict(DEFAULT_RETRIEVAL_POLICY.describe())

    assert "min similarity" in rendered
    assert not any("confidence" in label.lower() for label in rendered)
    assert rendered["visibility"] == "public (enforced)"


def test_a_context_policy_must_leave_room_for_an_answer():
    with pytest.raises(ValidationError):
        ContextPolicy(max_prompt_tokens=500, output_reserve_tokens=500)


def test_the_default_context_policy_leaves_room():
    assert DEFAULT_CONTEXT_POLICY.output_reserve_tokens < DEFAULT_CONTEXT_POLICY.max_prompt_tokens


# --- token estimation --------------------------------------------------------


def test_empty_text_costs_nothing():
    assert estimate_tokens("") == 0


def test_any_text_costs_at_least_one_token():
    assert estimate_tokens("a") == 1


def test_the_estimate_rounds_up_rather_than_down():
    """Rounding down would let a prompt overflow by a token at every boundary."""
    assert estimate_tokens("a" * 7) == 3  # 7 / 3.0 = 2.33…


def test_the_estimate_is_deliberately_pessimistic():
    """Below what real tokenizers achieve, so the error is always the safe one."""
    assert CHARS_PER_TOKEN <= 3.5
    assert estimate_tokens("x" * 300) >= 300 / 4


def test_the_estimate_is_monotonic_in_length():
    assert estimate_tokens("x" * 100) < estimate_tokens("x" * 200)
