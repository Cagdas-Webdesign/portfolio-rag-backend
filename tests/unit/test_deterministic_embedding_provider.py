"""The local embedding provider: deterministic, offline, and not a mock."""

from __future__ import annotations

import asyncio
import math

import pytest

from portfolio_rag.infrastructure.embedding import DeterministicEmbeddingProvider
from portfolio_rag.ingestion.embedding import EMBEDDING_REPRESENTATION_VERSION
from portfolio_rag.ports.embeddings import EmbeddingInput


def _embed(provider: DeterministicEmbeddingProvider, *texts: str) -> list[tuple[float, ...]]:
    inputs = [EmbeddingInput(id=f"i{index}", text=text) for index, text in enumerate(texts)]
    return [result.vector for result in asyncio.run(provider.embed(inputs))]


def test_the_provider_declares_a_complete_embedding_space():
    spec = DeterministicEmbeddingProvider(dimensions=32).spec

    assert spec.provider == "deterministic"
    assert spec.dimensions == 32
    assert spec.representation_version == EMBEDDING_REPRESENTATION_VERSION


def test_the_same_text_always_produces_the_same_vector():
    provider = DeterministicEmbeddingProvider(dimensions=16)

    assert _embed(provider, "hello") == _embed(provider, "hello")


def test_a_fresh_provider_object_produces_the_same_vectors():
    """Derived from SHA-256, so nothing is carried in instance state."""
    first = _embed(DeterministicEmbeddingProvider(dimensions=16), "hello")
    second = _embed(DeterministicEmbeddingProvider(dimensions=16), "hello")

    assert first == second


def test_the_derivation_is_stable_across_processes():
    """Pinned literally: Python's `hash()` is randomised per process, so a
    regression to it would break re-indexing in a way no in-process test sees."""
    (vector,) = _embed(DeterministicEmbeddingProvider(dimensions=4), "stability")

    assert [round(value, 12) for value in vector] == [
        -0.730007692835,
        0.277536108467,
        0.47227086954,
        0.408684111121,
    ]


def test_different_text_produces_different_vectors():
    provider = DeterministicEmbeddingProvider(dimensions=16)

    first, second = _embed(provider, "alpha", "beta")

    assert first != second


def test_the_vectors_are_not_all_zeros():
    """A zero provider would make every similarity test vacuously pass."""
    (vector,) = _embed(DeterministicEmbeddingProvider(dimensions=16), "content")

    assert any(value != 0.0 for value in vector)
    assert len({round(value, 6) for value in vector}) > 1


def test_vectors_have_the_declared_dimension_and_are_finite():
    for dimensions in (1, 8, 33, 256):
        (vector,) = _embed(DeterministicEmbeddingProvider(dimensions=dimensions), "text")

        assert len(vector) == dimensions
        assert all(math.isfinite(value) for value in vector)


def test_vectors_are_unit_length_so_cosine_scores_are_readable():
    (vector,) = _embed(DeterministicEmbeddingProvider(dimensions=64), "text")

    assert math.isclose(math.sqrt(sum(value**2 for value in vector)), 1.0, rel_tol=1e-9)


def test_a_batch_keeps_every_result_matched_to_its_input():
    provider = DeterministicEmbeddingProvider(dimensions=8)
    inputs = [EmbeddingInput(id=name, text=name.upper()) for name in ("a", "b", "c")]

    results = asyncio.run(provider.embed(inputs))

    assert [result.id for result in results] == ["a", "b", "c"]
    for result in results:
        (expected,) = _embed(provider, result.id.upper())
        assert result.vector == expected


def test_an_empty_batch_returns_nothing():
    assert asyncio.run(DeterministicEmbeddingProvider().embed([])) == []


def test_a_dimension_of_zero_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        DeterministicEmbeddingProvider(dimensions=0)


def test_similar_inputs_do_not_collide():
    """Not semantic, but distinguishable — which is what ordering tests need."""
    provider = DeterministicEmbeddingProvider(dimensions=32)

    vectors = _embed(provider, "text a", "text b", "text c")

    assert len(set(vectors)) == 3
