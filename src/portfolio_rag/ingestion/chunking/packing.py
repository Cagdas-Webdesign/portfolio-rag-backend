"""Turning a section's blocks into size-bounded pieces of text.

Three jobs, in order:

1. **Packing** — fill a chunk with whole blocks while there is budget.
2. **Fallback** — divide a single block that busts the budget on its own,
   along the cleanest boundary available.
3. **Overlap** — repeat a little of the previous chunk at the start of the next
   one, but only inside the same section.

The packing rule, stated once: add whole blocks while the chunk is still below
``target_chars`` *and* the result would stay within ``max_chars``. Reaching the
target stops packing; exceeding the maximum is never allowed. That is why a
chunk usually lands a little above the target and never above the maximum.

Nothing here mixes blocks from two sections. A heading is the author saying
"this is a different subject", and packing across that boundary to fill a
budget would produce a unit that answers neither question well.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from portfolio_rag.ingestion.chunking.errors import UnsplittableBlockError
from portfolio_rag.ingestion.chunking.policy import ChunkingPolicy
from portfolio_rag.ingestion.chunking.structure import Block, Section

#: Blocks are rejoined the way Markdown separates them.
BLOCK_SEPARATOR: Final = "\n\n"

#: List items are rejoined line-wise so the list keeps its markers.
_ITEM_SEPARATOR: Final = "\n"

#: End of a sentence: terminal punctuation, optional closing quote or bracket,
#: then whitespace. Deliberately a small heuristic — a sentence tokenizer would
#: mean an NLP dependency for a fallback that only fires on runaway paragraphs.
_SENTENCE_END: Final = re.compile("[.!?\u2026]['\"\u00bb\u201d\u2019)\\]]*\\s+")

#: A line that opens or closes a fenced code block.
_FENCE_LINE: Final = re.compile(r"^\s{0,3}(?:`{3,}|~{3,})")


@dataclass(frozen=True, slots=True)
class PackedChunk:
    """One chunk's text, plus how much of its start is repeated context.

    ``overlap_prefix_chars`` counts repeated *source* characters: ``text[:n]``
    is exactly the text that also appears in the previous chunk. The blank line
    that joins it to the rest is structure, not repeated content, so it is not
    counted.
    """

    text: str
    overlap_prefix_chars: int = 0


def pack_section(
    section: Section,
    policy: ChunkingPolicy,
    *,
    document_id: str,
) -> list[PackedChunk]:
    """Pack one section into chunks, applying overlap where it was split."""
    pieces = _pack_blocks(section, policy, document_id=document_id)
    return _apply_overlap(pieces, policy)


def _pack_blocks(section: Section, policy: ChunkingPolicy, *, document_id: str) -> list[str]:
    """Greedily fill chunks with whole blocks; divide blocks that do not fit."""
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current, current_length
        if current:
            pieces.append(BLOCK_SEPARATOR.join(current))
            current = []
            current_length = 0

    for block in section.blocks:
        for fragment in _fit_block(block, policy, section=section, document_id=document_id):
            if not current:
                current = [fragment]
                current_length = len(fragment)
                continue

            combined = current_length + len(BLOCK_SEPARATOR) + len(fragment)
            if combined <= policy.max_chars and current_length < policy.target_chars:
                current.append(fragment)
                current_length = combined
            else:
                flush()
                current = [fragment]
                current_length = len(fragment)

    flush()
    return pieces


def _fit_block(
    block: Block,
    policy: ChunkingPolicy,
    *,
    section: Section,
    document_id: str,
) -> list[str]:
    """Return the block as one fragment, or as several that each fit."""
    if len(block) <= policy.max_chars:
        return [block.text]

    if block.is_atomic:
        raise UnsplittableBlockError(
            f"A {block.kind.value} block of {len(block)} characters exceeds the "
            f"{policy.max_chars}-character limit and cannot be divided without "
            "producing broken Markdown. Split it in the source document.",
            document_id=document_id,
            heading_path=section.heading_path,
        )

    fragments = (
        _pack_units(block.split_units, _ITEM_SEPARATOR, policy.max_chars)
        if block.split_units
        else _split_text(block.text, policy.max_chars)
    )

    # A fence nested inside a list item or a blockquote survives as long as the
    # cut lands between whole units. When it does not, the result would be an
    # unterminated fence, so this fails the same way an oversized fence does.
    if any(not _has_balanced_fences(fragment) for fragment in fragments):
        raise UnsplittableBlockError(
            f"A {block.kind.value} block of {len(block)} characters exceeds the "
            f"{policy.max_chars}-character limit and can only be divided by "
            "cutting through a fenced code block. Split it in the source document.",
            document_id=document_id,
            heading_path=section.heading_path,
        )

    return fragments


def _pack_units(units: tuple[str, ...] | list[str], separator: str, limit: int) -> list[str]:
    """Greedily group natural sub-units (list items, lines, sentences, words)."""
    packed: list[str] = []
    current: list[str] = []
    current_length = 0

    for unit in units:
        if not current:
            current = [unit]
            current_length = len(unit)
            continue
        combined = current_length + len(separator) + len(unit)
        if combined <= limit:
            current.append(unit)
            current_length = combined
        else:
            packed.append(separator.join(current))
            current = [unit]
            current_length = len(unit)

    if current:
        packed.append(separator.join(current))

    # A single unit may still be too large; refine only those.
    refined: list[str] = []
    for piece in packed:
        refined.extend(_split_text(piece, limit) if len(piece) > limit else [piece])
    return refined


def _split_text(text: str, limit: int) -> list[str]:
    """Divide oversized text along the cleanest boundary that exists.

    The ladder, coarsest first: line breaks, then sentence ends, then any
    whitespace, then — only if a single "word" is itself longer than the limit —
    a hard character cut. Each rung produces strictly shorter pieces than its
    input, so the recursion terminates; the hard cut always succeeds, so it
    cannot loop.
    """
    if len(text) <= limit:
        return [text]

    for units, separator in (
        (text.split("\n"), "\n"),
        (_split_sentences(text), " "),
        (text.split(" "), " "),
    ):
        if len(units) < 2:
            continue
        return _pack_units(units, separator, limit)

    return _hard_cut(text, limit)


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping the punctuation attached."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        sentences.append(text[start : match.end()].strip())
        start = match.end()
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return [sentence for sentence in sentences if sentence]


def _hard_cut(text: str, limit: int) -> list[str]:
    """Last resort for a single token longer than the limit.

    Reached only when there is no whitespace at all to cut on — a base64 blob,
    a minified line, a very long URL. Splitting mid-token is lossless for
    retrieval purposes and strictly better than emitting a unit that violates
    the hard ceiling.
    """
    return [text[start : start + limit] for start in range(0, len(text), limit)]


def _apply_overlap(pieces: list[str], policy: ChunkingPolicy) -> list[PackedChunk]:
    """Prepend a little of the previous chunk to every chunk after the first.

    Only within one section, and only when the section actually had to be
    split: a short standalone section gains nothing from repeating a neighbour
    it has no continuity with.

    Overlap is taken from the previous chunk's *own* text, before that chunk
    received its own overlap, so the repetition cannot compound across a long
    section.
    """
    if policy.overlap_chars <= 0 or len(pieces) < 2:
        return [PackedChunk(piece) for piece in pieces]

    chunks = [PackedChunk(pieces[0])]
    for index in range(1, len(pieces)):
        body = pieces[index]
        # The hard ceiling wins over the overlap budget: a nearly full chunk
        # gets a smaller prefix rather than no prefix, and never an oversized
        # chunk. Because the budget is capped here, the result always fits.
        budget = min(policy.overlap_chars, policy.max_chars - len(body) - len(BLOCK_SEPARATOR))
        prefix = _overlap_prefix(pieces[index - 1], budget)
        if prefix:
            chunks.append(
                PackedChunk(
                    text=prefix + BLOCK_SEPARATOR + body,
                    overlap_prefix_chars=len(prefix),
                )
            )
        else:
            chunks.append(PackedChunk(body))
    return chunks


def _overlap_prefix(text: str, budget: int) -> str:
    """Take up to *budget* trailing characters, snapped to a clean boundary.

    Preference order: paragraph break, line break, sentence end, word break.
    The result is usually shorter than the budget — a clean boundary beats an
    exact character count, because a fragment starting mid-sentence is noise in
    an embedding.

    Returns an empty string when nothing clean is available, or when the
    candidate would carry an unbalanced code fence into the next chunk.
    """
    if budget <= 0 or not text:
        return ""

    tail = text[-budget:]

    for marker in (BLOCK_SEPARATOR, "\n"):
        position = tail.find(marker)
        if position != -1:
            candidate = tail[position + len(marker) :].strip()
            if _is_usable_overlap(candidate):
                return candidate

    match = _SENTENCE_END.search(tail)
    if match:
        candidate = tail[match.end() :].strip()
        if _is_usable_overlap(candidate):
            return candidate

    started_mid_word = len(text) > budget and not text[-budget - 1].isspace()
    if started_mid_word:
        space = tail.find(" ")
        candidate = tail[space + 1 :].strip() if space != -1 else ""
    else:
        candidate = tail.strip()

    return candidate if _is_usable_overlap(candidate) else ""


def _is_usable_overlap(candidate: str) -> bool:
    return bool(candidate) and _has_balanced_fences(candidate)


def _has_balanced_fences(text: str) -> bool:
    """Reject a fragment that would open a code fence it never closes."""
    return sum(1 for line in text.split("\n") if _FENCE_LINE.match(line)) % 2 == 0
