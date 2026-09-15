"""The in-memory store, held to the full VectorStore contract.

The reference implementation: if this store accepts something, the contract
permits it. Everything below the contract suite covers behaviour specific to
this adapter rather than required of every one.
"""

from __future__ import annotations

import pytest

from portfolio_rag.domain.embedding import SimilarityMetric, VectorIndexSpec
from portfolio_rag.infrastructure.vector_store import InMemoryVectorStore
from portfolio_rag.ports.errors import VectorStoreError
from portfolio_rag.ports.vector_store import VectorQuery, VectorStore
from tests.contracts.vector_store import INDEX_SPEC, SPEC, VectorStoreContract, record
from tests.support import run


class TestInMemoryVectorStoreContract(VectorStoreContract):
    def make_store(self) -> VectorStore:
        return InMemoryVectorStore(INDEX_SPEC)


def _store() -> InMemoryVectorStore:
    return InMemoryVectorStore(INDEX_SPEC)


def test_the_store_reports_the_index_spec_it_was_built_with():
    store = _store()

    assert store.index_spec.embedding == SPEC
    assert store.index_spec.metric is SimilarityMetric.COSINE
    assert store.index_spec.dimensions == 3


def test_this_store_can_enumerate_itself():
    assert _store().supports_enumeration is True


def test_cosine_scores_are_orientation_based_not_magnitude_based():
    """A vector and its scaled twin point the same way, so they score the same."""
    store = _store()
    run(store.upsert([record("doc--0000", (2.0, 0.0, 0.0))]))

    (match,) = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=1)))

    assert match.score == pytest.approx(1.0)


def test_an_opposing_vector_scores_negative():
    store = _store()
    run(store.upsert([record("doc--0000", (-1.0, 0.0, 0.0))]))

    (match,) = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=1)))

    assert match.score == pytest.approx(-1.0)


def test_a_zero_vector_scores_zero_rather_than_dividing_by_zero():
    store = _store()
    run(store.upsert([record("doc--0000", (0.0, 0.0, 0.0))]))

    (match,) = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=1)))

    assert match.score == 0.0


def test_a_filter_on_an_unknown_field_matches_nothing():
    """Better to return nothing than to silently ignore the constraint."""
    store = _store()
    run(store.upsert([record("doc--0000")]))

    matches = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), filters={"nonexistent": "x"})))

    assert matches == []


def test_a_list_valued_metadata_field_matches_on_membership():
    from tests.factories import make_metadata

    store = _store()
    run(
        store.upsert(
            [record("doc--0000", metadata=make_metadata("doc--0000", topics=("backend", "api")))]
        )
    )

    matches = run(
        store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), filters={"topics": "backend"}))
    )

    assert [match.record.id for match in matches] == ["doc--0000"]


def test_two_stores_do_not_share_state():
    first, second = _store(), _store()
    run(first.upsert([record("doc--0000")]))

    assert run(second.list_state()) == []


def test_an_index_built_for_another_space_refuses_everything_from_this_one():
    other_spec = VectorIndexSpec(embedding=SPEC.model_copy(update={"dimensions": 8}))
    store = InMemoryVectorStore(other_spec)

    with pytest.raises(VectorStoreError, match=r"dimensions|embedding space"):
        run(store.upsert([record("doc--0000")]))
