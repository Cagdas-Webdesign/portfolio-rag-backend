"""Citation validation: the backend deciding what an answer may claim."""

from __future__ import annotations

from portfolio_rag.domain.retrieval import RetrievedChunk, SourceCitation
from portfolio_rag.rag.citations import resolve_citations
from portfolio_rag.rag.context import GroundedContext, build_context
from tests.unit.test_context_builder import GENEROUS, retrieved


def context_of(*passages: RetrievedChunk) -> GroundedContext:
    return build_context(list(passages), available_tokens=GENEROUS)


def three_sources() -> GroundedContext:
    return context_of(
        retrieved("aa--0000", "First fact.", document_id="aa", title="Doc A", source="a.md"),
        retrieved("bb--0000", "Second fact.", document_id="bb", title="Doc B", source="b.md"),
        retrieved("cc--0000", "Third fact.", document_id="cc", title="Doc C", source="c.md"),
    )


def test_one_valid_label_becomes_one_citation():
    outcome = resolve_citations(three_sources(), ["S1"])

    assert [citation.title for citation in outcome.citations] == ["Doc A"]
    assert outcome.is_grounded


def test_several_valid_labels_become_several_citations_in_the_order_claimed():
    outcome = resolve_citations(three_sources(), ["S3", "S1"])

    assert [citation.title for citation in outcome.citations] == ["Doc C", "Doc A"]


def test_a_label_that_was_never_in_the_context_is_refused():
    outcome = resolve_citations(three_sources(), ["S9"])

    assert outcome.citations == ()
    assert outcome.unknown_labels == ("S9",)
    assert not outcome.is_grounded


def test_an_invented_source_is_never_published():
    """A model naming a file is producing text, not provenance."""
    outcome = resolve_citations(three_sources(), ["https://example.test/secret", "docs/hidden.md"])

    assert outcome.citations == ()
    assert len(outcome.unknown_labels) == 2


def test_the_valid_half_of_a_mixed_claim_survives():
    outcome = resolve_citations(three_sources(), ["S1", "S9"])

    assert [citation.title for citation in outcome.citations] == ["Doc A"]
    assert outcome.unknown_labels == ("S9",)


def test_a_repeated_label_is_counted_once():
    outcome = resolve_citations(three_sources(), ["S1", "S1", "S1"])

    assert len(outcome.citations) == 1
    assert outcome.duplicate_labels == 2


def test_no_claims_at_all_means_nothing_is_grounded():
    outcome = resolve_citations(three_sources(), [])

    assert outcome.citations == ()
    assert not outcome.is_grounded


def test_two_passages_from_one_document_and_section_become_one_citation():
    """A reader wants the source, not the chunking."""
    context = context_of(
        retrieved("doc--0000", "First half.", heading_path=("APIs",)),
        retrieved("doc--0001", "Second half.", heading_path=("APIs",)),
    )

    outcome = resolve_citations(context, ["S1", "S2"])

    assert len(outcome.citations) == 1
    assert len(outcome.cited) == 2, "the granular trail is kept for diagnostics"


def test_two_sections_of_one_document_stay_two_citations():
    context = context_of(
        retrieved("doc--0000", "About APIs.", heading_path=("APIs",)),
        retrieved("doc--0001", "About storage.", heading_path=("Storage",)),
    )

    outcome = resolve_citations(context, ["S1", "S2"])

    assert [citation.section for citation in outcome.citations] == ["APIs", "Storage"]


def test_a_citation_is_built_from_the_retrieved_chunk_not_from_the_model():
    context = context_of(
        retrieved(
            "guide--0000",
            "Some passage.",
            document_id="guide",
            title="Integration Guide",
            source="docs/guide.md",
            heading_path=("Backend", "APIs"),
        )
    )

    (citation,) = resolve_citations(context, ["S1"]).citations

    assert citation == SourceCitation(
        document_id="guide",
        title="Integration Guide",
        source="docs/guide.md",
        section="APIs",
    )


def test_a_public_citation_carries_no_internal_fields():
    (citation,) = resolve_citations(three_sources(), ["S1"]).citations
    payload = citation.model_dump()

    assert set(payload) == {"document_id", "title", "source", "section"}
    forbidden_fields = ("chunk_id", "similarity", "fingerprint", "visibility", "trust_level")
    for forbidden in (*forbidden_fields, "label"):
        assert forbidden not in payload


def test_a_citation_never_carries_the_passage_text():
    (citation,) = resolve_citations(three_sources(), ["S1"]).citations

    assert "First fact." not in str(citation.model_dump())


def test_the_same_claims_resolve_the_same_way_every_time():
    context = three_sources()

    assert resolve_citations(context, ["S2"]) == resolve_citations(context, ["S2"])
