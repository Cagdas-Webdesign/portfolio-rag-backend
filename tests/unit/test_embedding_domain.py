"""Embedding spaces and vectors: what the domain refuses to represent."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    SimilarityMetric,
    VectorIndexSpec,
    VectorRecord,
)
from tests.factories import DEFAULT_SPEC, make_metadata, make_vector_record


def _spec(**overrides: object) -> EmbeddingSpec:
    fields: dict[str, object] = {
        "provider": "mistral",
        "model": "mistral-embed",
        "dimensions": 1024,
        "representation_version": "embedding-text-v1",
    }
    fields.update(overrides)
    return EmbeddingSpec.model_validate(fields)


# --- embedding space --------------------------------------------------------


def test_a_complete_spec_validates_and_describes_itself():
    spec = _spec()

    assert spec.identity == "mistral:mistral-embed:1024:embedding-text-v1"
    assert dict(spec.describe())["model"] == "mistral-embed"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider", ""),
        ("model", ""),
        ("representation_version", ""),
        ("dimensions", 0),
        ("dimensions", -1),
    ],
)
def test_an_incomplete_or_impossible_spec_is_rejected(field: str, value: object):
    with pytest.raises(ValidationError):
        _spec(**{field: value})


def test_specs_compare_by_value():
    assert _spec() == _spec()
    assert _spec() != _spec(model="other")


def test_matching_dimensions_alone_do_not_make_two_spaces_compatible():
    """The trap this whole model exists to prevent."""
    mistral = _spec(provider="mistral", model="mistral-embed", dimensions=1024)
    other = _spec(provider="other", model="other-embed", dimensions=1024)

    assert mistral.dimensions == other.dimensions
    assert not mistral.is_compatible_with(other)


def test_a_different_representation_version_is_a_different_space():
    """The same model over differently composed text is not comparable."""
    first = _spec(representation_version="embedding-text-v1")
    second = _spec(representation_version="embedding-text-v2")

    assert not first.is_compatible_with(second)


def test_a_spec_is_compatible_with_itself():
    assert _spec().is_compatible_with(_spec())


def test_a_spec_is_immutable():
    with pytest.raises(ValidationError):
        _spec().dimensions = 512  # type: ignore[misc]


# --- index spec -------------------------------------------------------------


def test_an_index_spec_defaults_to_cosine_and_inherits_its_dimensions():
    index_spec = VectorIndexSpec(embedding=_spec())

    assert index_spec.metric is SimilarityMetric.COSINE
    assert index_spec.dimensions == 1024
    assert dict(index_spec.describe())["metric"] == "cosine"


def test_an_unknown_metric_is_rejected():
    with pytest.raises(ValidationError):
        VectorIndexSpec(embedding=_spec(), metric="euclidean")


# --- vectors ----------------------------------------------------------------


def test_a_vector_of_the_declared_length_is_accepted():
    record = make_vector_record(embedding=(0.1, -0.2, 3.5))

    assert len(record.embedding) == 3


@pytest.mark.parametrize(
    "embedding",
    [(), (float("nan"), 0.0, 1.0), (float("inf"), 0.0, 1.0), (0.0, float("-inf"), 1.0)],
    ids=["empty", "nan", "+inf", "-inf"],
)
def test_unusable_vectors_are_refused_at_the_boundary(embedding: tuple[float, ...]):
    with pytest.raises(ValidationError):
        make_vector_record(embedding=embedding)


def test_values_are_not_squeezed_into_an_invented_range():
    """Real models do not promise [-1, 1]; imposing it would corrupt vectors."""
    record = make_vector_record(embedding=(-42.5, 0.0, 1_000.25))

    assert record.embedding == (-42.5, 0.0, 1_000.25)
    assert all(math.isfinite(value) for value in record.embedding)


def test_integers_are_accepted_as_floats():
    record = make_vector_record(embedding=(1, 0, -1))

    assert record.embedding == (1.0, 0.0, -1.0)


# --- records ----------------------------------------------------------------


def test_a_record_requires_every_fingerprint():
    with pytest.raises(ValidationError):
        VectorRecord(  # type: ignore[call-arg]
            id="doc--0000",
            spec=DEFAULT_SPEC,
            embedding=(0.1, 0.2, 0.3),
            metadata=make_metadata(),
            embedding_fingerprint="a" * 64,
            chunk_fingerprint="b" * 64,
        )


def test_fingerprints_must_look_like_sha256_digests():
    with pytest.raises(ValidationError):
        make_vector_record(embedding_fingerprint="not-a-digest")


def test_metadata_rejects_fields_it_was_not_designed_to_carry():
    """The metadata set is a decision, not a dumping ground."""
    with pytest.raises(ValidationError):
        make_metadata(content="the whole chunk text")


def test_metadata_keeps_the_fields_retrieval_will_filter_on():
    metadata = make_metadata(visibility="internal", trust_level="unverified", language="de")

    assert metadata.visibility.value == "internal"
    assert metadata.trust_level.value == "unverified"
    assert metadata.language == "de"
