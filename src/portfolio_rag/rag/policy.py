"""The knobs retrieval and context building are allowed to have.

Small on purpose. Every field here is one a caller can genuinely act on today;
re-rankers, hybrid weights, BM25 mixing and multi-query settings are absent
because none of them is implemented, and a configuration field for a feature
that does not exist is a lie with a default value.

**What is not a setting: visibility.** A public request retrieves public
documents. That is not a policy field, not a flag and not a parameter — it is
built into the retrieval service, so no caller can forget it, override it, or
be talked into it. See :mod:`portfolio_rag.rag.retrieval`.

The numbers below have been **measured, not merely chosen** — see
`evaluation/` and the sweep recorded on :data:`DEFAULT_MIN_SIMILARITY`. What
the measurement showed is that they are also **provider-specific**: a cosine
threshold means something different under every embedding model, so these stay
conservative starting values and the harness exists so they can be re-measured
against whichever model actually runs.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: How many candidates a search asks for — and, since nothing reranks them,
#: how many passages the answer may draw on.
#:
#: **Seven, by measurement.** Swept on 2026-10-04 over the 24 release-acceptance
#: questions against the production path (`mistral-embed`, Vectorize), from one
#: top-30 retrieval export per question: five, seven and eight passages, and
#: per-document diversity caps over a 10- and 20-candidate window. Every
#: diversity cap cost `section-lead-flow` one of the three sections of one
#: document it needs. Seven lost no expected source on any question, kept
#: hit@1/3/5 (13/16/17 of 20), raised MRR 0.735 → 0.742 and expected-source
#: coverage 0.598 → 0.614, and brought a passage that answers a broad question
#: (rank 7) into the context. Eight added one more source at +66 % context
#: instead of +44 %. The record is in `evaluation/README.md`. Like the
#: threshold, this is a property of one embedding space: re-measure it there.
DEFAULT_TOP_K: Final = 7

#: Below this cosine similarity, a match is not treated as evidence.
#:
#: A *similarity*, never a confidence: it says how close two vectors point, not
#: how likely an answer is to be correct, and nothing renders it as a
#: percentage.
#:
#: **This number is specific to one embedding space, and the evaluation says so
#: with data.** Swept over the fixture corpus (`evaluation/`, 24 questions)
#: under a lexical embedding, the similarity ranges of answerable and
#: unanswerable questions overlap completely — answerable spans 0.257 to 0.791,
#: unanswerable 0.265 to 0.650 — so no value separates them. Raising the threshold
#: until out-of-scope questions are rejected costs real answers at roughly one
#: for one; lowering it to 0.15 recovers three answerable questions and rejects
#: no more noise.
#:
#: Two things follow, and both are load-bearing:
#:
#: 1. A threshold filters *noise*, not *topic*. What actually keeps the system
#:    from answering a question the corpus does not cover is the grounding path
#:    — a model that declines, and a backend that refuses to publish an answer
#:    with no verified citation. The evaluation is where that stopped being an
#:    assumption.
#: 2. Cosine distributions differ per provider, so a value tuned against one
#:    embedding model means nothing under another. This default is left at a
#:    conservative starting value rather than fitted to the offline double, and
#:    **re-measuring it against the production embedding model with
#:    `portfolio-rag eval run` is a documented deployment step**, not a
#:    refinement somebody may get around to.
DEFAULT_MIN_SIMILARITY: Final = 0.25

#: Ceiling on everything sent to the model in one request: instructions plus
#: context plus question. Well under any current model's window on purpose —
#: the budget is a cost and latency decision, not a limit discovered by hitting
#: one.
DEFAULT_MAX_PROMPT_TOKENS: Final = 6000

#: Room left for the answer, and simultaneously the cap requested from the
#: provider. One number, so the space reserved and the space allowed cannot
#: disagree.
DEFAULT_OUTPUT_RESERVE_TOKENS: Final = 800

#: The output cap of a recovery attempt — the one second request a step may
#: make — when the first reply stopped at the output limit. Never the cap of a
#: first attempt: the ordinary request stays at the reserve and costs what it
#: did.
#:
#: **1500, by measurement.** `@cf/openai/gpt-oss-120b` reasons before it
#: answers, and the reasoning counts against the cap: of 118 calls with
#: reported usage in `evaluation/results/` (all at 800), the parsed ones used
#: 116 to 786 tokens for replies of a few hundred characters, and four stopped at
#: exactly 800 with no or a cut-off reply — two generations, two grounding
#: checks. The share of calls reaching *n* tokens halves roughly every 95
#: tokens (≥ 500: 30.5 %, ≥ 600: 14.4 %, ≥ 700: 6.8 %, ≥ 800: 3.4 %). Extended
#: past 800 — an extrapolation, not an observation — a reply that already ran
#: past 800 runs past 1200 about one time in twenty, past 1500 about one in
#: 170, past 1600 about one in 340. 1500 is the smallest round cap that brings
#: a second failure under one percent.
DEFAULT_RECOVERY_OUTPUT_TOKENS: Final = 1500


class RetrievalPolicy(BaseModel):
    """How many candidates to consider, and how close is close enough."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=50,
        description="Maximum number of candidates the vector store is asked for.",
    )
    min_similarity: float = Field(
        default=DEFAULT_MIN_SIMILARITY,
        ge=-1.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for a match to count as evidence. "
            "A similarity, not a confidence."
        ),
    )

    def describe(self) -> tuple[tuple[str, str], ...]:
        """Label/value pairs for developer output."""
        return (
            ("top k", str(self.top_k)),
            ("min similarity", f"{self.min_similarity:.3f}"),
            ("visibility", "public (enforced)"),
        )


class ContextPolicy(BaseModel):
    """The token budget one answer is allowed to spend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_prompt_tokens: int = Field(
        default=DEFAULT_MAX_PROMPT_TOKENS,
        gt=0,
        description="Ceiling on instructions + context + question, in estimated tokens.",
    )
    output_reserve_tokens: int = Field(
        default=DEFAULT_OUTPUT_RESERVE_TOKENS,
        gt=0,
        description="Tokens held back for the answer, and the cap requested from the provider.",
    )
    recovery_output_tokens: int = Field(
        default=DEFAULT_RECOVERY_OUTPUT_TOKENS,
        gt=0,
        description=(
            "The cap of a recovery attempt after a reply stopped at the output limit. "
            "Never used for a first attempt."
        ),
    )

    @model_validator(mode="after")
    def _check_budget_leaves_room(self) -> ContextPolicy:
        if self.output_reserve_tokens >= self.max_prompt_tokens:
            raise ValueError(
                f"output_reserve_tokens ({self.output_reserve_tokens}) must be smaller "
                f"than max_prompt_tokens ({self.max_prompt_tokens})"
            )
        if self.recovery_output_tokens < self.output_reserve_tokens:
            raise ValueError(
                f"recovery_output_tokens ({self.recovery_output_tokens}) must not be smaller "
                f"than output_reserve_tokens ({self.output_reserve_tokens})"
            )
        return self

    def describe(self) -> tuple[tuple[str, str], ...]:
        return (
            ("max prompt tokens", str(self.max_prompt_tokens)),
            ("output reserve", str(self.output_reserve_tokens)),
            ("recovery output limit", str(self.recovery_output_tokens)),
        )


#: Used when no policy is supplied.
DEFAULT_RETRIEVAL_POLICY: Final = RetrievalPolicy()
DEFAULT_CONTEXT_POLICY: Final = ContextPolicy()
