"""A reusable behavioural contract for :class:`VectorStore` implementations.

Written against the port, never against an implementation's internals. Any
adapter — the in-memory reference, a future Vectorize one running against a
real test index — can be dropped in by subclassing the suite and supplying a
store.

The tests describe what a *caller* is entitled to assume: that an upsert is
visible, that a second upsert replaces rather than duplicates, that a query
comes back in a defined order, that a filter is honoured, and that a vector
from the wrong embedding space is refused rather than silently mixed in.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pytest

from portfolio_rag.domain.embedding import (
    EmbeddingSpec,
    VectorIndexSpec,
    VectorRecord,
)
from portfolio_rag.ports.errors import VectorStoreError
from portfolio_rag.ports.vector_store import VectorQuery, VectorStore
from tests.factories import make_metadata, make_vector_record
from tests.support import run

SPEC = EmbeddingSpec(
    provider="deterministic",
    model="sha256-derived-v1",
    dimensions=3,
    representation_version="embedding-text-v1",
)
INDEX_SPEC = VectorIndexSpec(embedding=SPEC)

OTHER_SPACE = SPEC.model_copy(update={"model": "some-other-model"})
"""Same dimensionality, different model — the case that must still be refused."""


def record(
    record_id: str, embedding: tuple[float, ...] = (1.0, 0.0, 0.0), **overrides: Any
) -> VectorRecord:
    return make_vector_record(record_id, spec=SPEC, embedding=embedding, **overrides)


class VectorStoreContract:
    """Subclass and provide :meth:`make_store`."""

    def make_store(self) -> VectorStore:  # pragma: no cover - overridden
        raise NotImplementedError

    @pytest.fixture
    def store(self) -> VectorStore:
        return self.make_store()

    # --- state ------------------------------------------------------------

    def test_a_new_store_is_empty(self, store: VectorStore):
        assert run(store.list_state()) == []
        assert run(store.fetch(["absent"])) == []

    def test_an_upserted_record_can_be_read_back_whole(self, store: VectorStore):
        original = record("doc--0000")

        run(store.upsert([original]))
        (fetched,) = run(store.fetch(["doc--0000"]))

        assert fetched == original

    def test_state_omits_the_vector_but_keeps_the_identity(self, store: VectorStore):
        original = record("doc--0000")
        run(store.upsert([original]))

        (state,) = run(store.fetch_states(["doc--0000"]))

        assert state == original.state()
        assert state.embedding_fingerprint == original.embedding_fingerprint

    def test_upserting_the_same_id_replaces_rather_than_duplicates(self, store: VectorStore):
        run(store.upsert([record("doc--0000", (1.0, 0.0, 0.0))]))
        run(store.upsert([record("doc--0000", (0.0, 1.0, 0.0))]))

        states = run(store.list_state())
        (fetched,) = run(store.fetch(["doc--0000"]))

        assert len(states) == 1
        assert fetched.embedding == (0.0, 1.0, 0.0)

    def test_metadata_survives_a_round_trip(self, store: VectorStore):
        metadata = make_metadata(
            "doc--0000",
            heading_path=("Backend", "APIs"),
            visibility="internal",
            trust_level="authoritative",
            topics=("backend", "integration"),
            technologies=("REST",),
        )
        run(store.upsert([record("doc--0000", metadata=metadata)]))

        (fetched,) = run(store.fetch(["doc--0000"]))

        assert fetched.metadata == metadata

    def test_several_records_are_all_kept(self, store: VectorStore):
        run(store.upsert([record(f"doc--{index:04d}") for index in range(3)]))

        assert len(run(store.list_state())) == 3

    def test_enumeration_is_ordered_deterministically(self, store: VectorStore):
        run(store.upsert([record("doc--0002"), record("doc--0000"), record("doc--0001")]))

        first = [state.id for state in run(store.list_state())]
        second = [state.id for state in run(store.list_state())]

        assert first == second
        assert first == sorted(first)

    def test_fetching_a_mix_of_known_and_unknown_ids_returns_the_known_ones(
        self, store: VectorStore
    ):
        run(store.upsert([record("doc--0000")]))

        fetched = run(store.fetch(["doc--0000", "absent"]))

        assert [item.id for item in fetched] == ["doc--0000"]

    # --- deletion ---------------------------------------------------------

    def test_a_deleted_record_is_gone(self, store: VectorStore):
        run(store.upsert([record("doc--0000"), record("doc--0001")]))

        run(store.delete(["doc--0000"]))

        assert [state.id for state in run(store.list_state())] == ["doc--0001"]

    def test_deleting_something_absent_is_not_an_error(self, store: VectorStore):
        run(store.delete(["never-existed"]))

        assert run(store.list_state()) == []

    # --- validation -------------------------------------------------------

    def test_a_record_from_another_embedding_space_is_refused(self, store: VectorStore):
        """Same dimensions, different model: the vectors are not comparable."""
        foreign = make_vector_record("doc--0000", spec=OTHER_SPACE, embedding=(1.0, 0.0, 0.0))

        with pytest.raises(VectorStoreError, match="embedding space"):
            run(store.upsert([foreign]))

    @pytest.mark.parametrize("embedding", [(1.0, 0.0), (1.0, 0.0, 0.0, 0.0)])
    def test_a_wrongly_sized_vector_is_refused(
        self, store: VectorStore, embedding: tuple[float, ...]
    ):
        with pytest.raises(VectorStoreError, match="dimensions"):
            run(store.upsert([record("doc--0000", embedding)]))

    def test_a_rejected_batch_leaves_the_index_untouched(self, store: VectorStore):
        run(store.upsert([record("doc--0000")]))

        with pytest.raises(VectorStoreError):
            run(store.upsert([record("doc--0001"), record("doc--0002", (1.0, 0.0))]))

        assert [state.id for state in run(store.list_state())] == ["doc--0000"]

    def test_a_query_vector_of_the_wrong_size_is_refused(self, store: VectorStore):
        with pytest.raises(VectorStoreError, match="dimensions"):
            run(store.query(VectorQuery(embedding=(1.0, 0.0))))

    # --- query ------------------------------------------------------------

    def test_an_empty_index_matches_nothing(self, store: VectorStore):
        assert run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0)))) == []

    def test_results_come_back_most_similar_first(self, store: VectorStore):
        run(
            store.upsert(
                [
                    record("doc--0000", (1.0, 0.0, 0.0)),
                    record("doc--0001", (0.0, 1.0, 0.0)),
                    record("doc--0002", (0.7071, 0.7071, 0.0)),
                ]
            )
        )

        matches = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=3)))

        assert [match.record.id for match in matches] == ["doc--0000", "doc--0002", "doc--0001"]
        assert math.isclose(matches[0].score, 1.0, rel_tol=1e-6)

    def test_top_k_limits_the_result(self, store: VectorStore):
        run(store.upsert([record(f"doc--{index:04d}") for index in range(5)]))

        assert len(run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=2)))) == 2

    def test_asking_for_more_than_the_index_holds_returns_what_there_is(self, store: VectorStore):
        run(store.upsert([record("doc--0000"), record("doc--0001")]))

        matches = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=50)))

        assert len(matches) == 2

    def test_identical_scores_are_broken_by_ascending_id(self, store: VectorStore):
        """Identical text really does produce identical vectors, so this is not
        a corner case — and an arbitrary order would make results irreproducible."""
        run(store.upsert([record("doc--0002"), record("doc--0000"), record("doc--0001")]))

        matches = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=3)))

        assert [match.record.id for match in matches] == ["doc--0000", "doc--0001", "doc--0002"]

    def test_a_query_is_repeatable(self, store: VectorStore):
        run(store.upsert([record(f"doc--{index:04d}") for index in range(4)]))
        query = VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=3)

        first = [match.record.id for match in run(store.query(query))]
        second = [match.record.id for match in run(store.query(query))]

        assert first == second

    def test_matches_carry_state_not_vectors(self, store: VectorStore):
        run(store.upsert([record("doc--0000")]))

        (match,) = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=1)))

        assert match.record.metadata.chunk_id == "doc--0000"
        assert "embedding" not in match.record.model_dump()

    # --- filtering --------------------------------------------------------

    def _seed_visibility(self, store: VectorStore) -> None:
        run(
            store.upsert(
                [
                    record("doc--0000", metadata=make_metadata("doc--0000", visibility="public")),
                    record("doc--0001", metadata=make_metadata("doc--0001", visibility="internal")),
                ]
            )
        )

    def test_an_empty_filter_matches_everything(self, store: VectorStore):
        self._seed_visibility(store)

        matches = run(store.query(VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=10, filters={})))

        assert len(matches) == 2

    def test_visibility_can_be_filtered(self, store: VectorStore):
        """The filter public retrieval relies on to keep internal knowledge private."""
        self._seed_visibility(store)

        matches = run(
            store.query(
                VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=10, filters={"visibility": "public"})
            )
        )

        assert [match.record.id for match in matches] == ["doc--0000"]

    def test_metadata_filters_apply_to_other_fields_too(self, store: VectorStore):
        run(
            store.upsert(
                [
                    record("doc--0000", metadata=make_metadata("doc--0000", language="en")),
                    record("doc--0001", metadata=make_metadata("doc--0001", language="de")),
                ]
            )
        )

        matches = run(
            store.query(
                VectorQuery(embedding=(1.0, 0.0, 0.0), top_k=10, filters={"language": "de"})
            )
        )

        assert [match.record.id for match in matches] == ["doc--0001"]

    def test_a_filter_that_matches_nothing_returns_nothing(self, store: VectorStore):
        self._seed_visibility(store)

        matches = run(
            store.query(
                VectorQuery(
                    embedding=(1.0, 0.0, 0.0), top_k=10, filters={"language": "does-not-exist"}
                )
            )
        )

        assert matches == []

    def test_filters_combine_as_and(self, store: VectorStore):
        self._seed_visibility(store)

        matches = run(
            store.query(
                VectorQuery(
                    embedding=(1.0, 0.0, 0.0),
                    top_k=10,
                    filters={"visibility": "public", "language": "en"},
                )
            )
        )

        assert [match.record.id for match in matches] == ["doc--0000"]


def all_contract_tests() -> Sequence[str]:
    """Names of the contract's tests, for a suite that wants to assert coverage."""
    return tuple(name for name in dir(VectorStoreContract) if name.startswith("test_"))
