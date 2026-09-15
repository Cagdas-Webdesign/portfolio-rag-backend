# ADR 0005 — Deterministic, structure-aware chunking

* **Status:** accepted
* **Date:** 2026-08-07
* **Phase:** 3

## Context

Retrieval never returns a document; it returns a chunk. Chunk boundaries
therefore decide what an answer can be grounded in, and no amount of embedding
or ranking quality repairs a unit that starts mid-sentence, ends before the
point, or mixes two subjects.

Three constraints shape the choice. There is no evaluation harness yet, so a
strategy has to be judged by reasoning rather than measurement — which argues
for one whose behaviour is obvious. No embedding model has been chosen, so
boundaries must not depend on a tokenizer. And the budget is €0, so nothing may
require a model call at ingestion time.

## Decision

Chunk on Markdown structure, with a character budget, deterministically.

* **Structure first.** markdown-it-py parses the body; chunks are sliced out of
  the source by the line ranges its tokens carry. Headings become a
  `heading_path`, not text in the chunk. A section is a hard boundary.
* **Blocks, not offsets.** Whole paragraphs, lists, quotes and code fences are
  packed together while the budget allows. Fenced code is atomic.
* **Character budget.** `target_chars` is a soft goal, `max_chars` a hard
  ceiling. When a single block busts the ceiling, a fallback ladder divides it:
  lines, then sentences, then whitespace, then — only for a token longer than
  the ceiling — a hard cut. An atomic block that cannot be divided is an error,
  not a silent oversized chunk.
* **Controlled overlap**, only between chunks of the same split section, snapped
  to a clean boundary, never breaking a fence, never exceeding the ceiling.
* **Stable identity.** `document-id--0000` for position; a SHA-256 fingerprint
  over strategy, policy, document id, heading path and content for change
  detection.
* **Versioned strategy.** `markdown-structure-v1`, recorded on every chunk.
* **Content stays source text.** Whether a title or heading path is prepended
  before embedding is Phase 4's decision, so the context is kept beside the
  content, not spliced into it.

## Alternatives considered

**Fixed-length character slicing.** Trivial and tokenizer-independent, but it
cuts mid-sentence and mid-fence, mixes subjects across headings, and destroys
the heading context that makes a citation specific. Rejected: it optimises the
one thing that does not matter (uniform size) at the cost of the thing that
does.

**Model-tokenizer-based chunking.** Fits an embedding model's window exactly.
Rejected for now: it couples the corpus to a vendor decision that has not been
taken, and would have to be redone if the model changes.
[ADR 0002](0002-provider-agnostic-core.md) exists to prevent exactly this. If
Phase 4 shows a model's window is the real constraint, a token-aware *check*
can be added on top of these boundaries without redesigning them.

**LLM or embedding-based semantic chunking.** Potentially better boundaries.
Rejected: it costs money per ingestion, needs a provider, is not reproducible
run to run, cannot be unit-tested, and its advantage over structure is an
assumption — nobody here has measured it. Markdown headings are already a human
semantic segmentation, written by the author.

## Consequences

**Good**

* Deterministic: the same document and policy produce the same chunks, ids and
  fingerprints on any machine, so chunking is testable by exact expectation.
* Transparent: every boundary is explainable by "a heading, a block edge, or the
  budget" — a developer can see why a cut happened, from the CLI.
* Free and offline: no provider, no network, no cost per ingestion.
* Provider-independent: the corpus does not encode a model choice.
* Structure preserved: fences stay intact, lists stay together, headings stay
  attached as data.

**Bad / accepted**

* A character budget is not a token budget. A chunk near `max_chars` may be
  larger or smaller in tokens than expected; Phase 4 must check this against the
  chosen model's limit rather than assume it.
* Structural heuristics are not comprehension. A section that changes subject
  without a heading is not detected.
* The defaults (1200/1800/150) are starting values, not measured optima. They
  are the first thing the evaluation pass should tune against real questions.
* Changing the policy changes chunk identity: a chunk is a unit produced by a
  recipe, so a different recipe means different units and re-embedding. That is
  correct, but it makes policy changes a corpus-wide operation, not a tweak.
* An oversized code fence fails ingestion instead of being cut. Deliberate — the
  alternative is broken Markdown in the index — but it puts an occasional
  authoring obligation on the knowledge base.

**Revisit when** evaluation exists (Phase 6) and can compare this strategy
against an alternative on real questions, or when an embedding model's token
window turns out to be the binding constraint. A second strategy would then
justify the abstraction that is deliberately absent today — there is one
strategy, so there is one implementation and no registry.
