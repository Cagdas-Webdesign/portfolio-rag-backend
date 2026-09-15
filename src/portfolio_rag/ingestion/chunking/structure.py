"""Reading a document's Markdown structure — headings and blocks.

markdown-it-py is used purely as a **parser**: ``.parse()`` gives a token
stream in which every block token carries the source line range it came from.
Chunk text is then sliced out of the original body by those line numbers, so
what ends up in a chunk is the author's Markdown, character for character.
Nothing is rendered, and the renderer is never called.

Writing this by hand was the alternative and it is a trap: fenced code blocks,
setext headings, lazy continuation lines and nested lists are exactly the cases
a naive line scanner gets wrong, and getting them wrong means a ``#`` inside a
code example silently becomes a section boundary.

Security: HTML is disabled and no plugins are loaded. Markdown is treated as
data — parsed for structure, never executed, never fetched, never rendered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from markdown_it import MarkdownIt
from markdown_it.token import Token

#: Parser instance: CommonMark, no raw HTML, no plugins. Stateless across
#: ``parse`` calls, so one shared instance is safe and avoids re-building the
#: rule chain for every document.
_PARSER: Final = MarkdownIt("commonmark", {"html": False})

_MAX_HEADING_LEVEL: Final = 6


class BlockKind(StrEnum):
    """What kind of Markdown block a piece of source is.

    The distinction only exists to answer two questions: may this block be
    split, and if so, along which boundaries?
    """

    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    QUOTE = "quote"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Block:
    """One top-level Markdown block, as it appears in the source."""

    kind: BlockKind
    text: str
    split_units: tuple[str, ...] = ()
    """Natural sub-units to fall back on when the block alone busts the budget.
    List items for lists; empty for everything else."""

    @property
    def is_atomic(self) -> bool:
        """Code must survive intact: a half fence is broken Markdown, and half
        a function is worse than no function."""
        return self.kind is BlockKind.CODE

    def __len__(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class Section:
    """A heading path and the blocks that live directly under it.

    A section is a hard boundary for chunking: its blocks are packed on their
    own and never mixed with another section's, because two headings mean the
    author considered the material separate.
    """

    heading_path: tuple[str, ...]
    blocks: tuple[Block, ...]


def analyze_structure(content: str) -> tuple[Section, ...]:
    """Split a document body into sections of blocks, in source order.

    Sections without blocks are dropped: a heading whose only child is another
    heading contributes to its children's heading path, not a chunk of its own.
    """
    lines = content.split("\n")
    tokens = _PARSER.parse(content)

    sections: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_path: tuple[str, ...] = ()
    current_blocks: list[Block] = []

    def close_section() -> None:
        if current_blocks:
            sections.append(Section(heading_path=current_path, blocks=tuple(current_blocks)))

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.level != 0:
            index += 1
            continue

        if token.type == "heading_open":
            close_section()
            current_blocks = []
            heading_text, index = _read_heading(tokens, index)
            _push_heading(heading_stack, _heading_level(token), heading_text)
            current_path = tuple(text for _, text in heading_stack)
            continue

        if token.nesting == 1:
            block, index = _read_container(tokens, index, lines)
            if block is not None:
                current_blocks.append(block)
            continue

        if token.nesting == 0:
            block = _read_leaf(token, lines)
            if block is not None:
                current_blocks.append(block)

        index += 1

    close_section()
    return tuple(sections)


def _heading_level(token: Token) -> int:
    """`h3` -> 3. Clamped, because the parser never emits anything else."""
    try:
        level = int(token.tag[1:])
    except ValueError:  # pragma: no cover - markdown-it always emits h1..h6
        return 1
    return min(max(level, 1), _MAX_HEADING_LEVEL)


def _read_heading(tokens: list[Token], index: int) -> tuple[str, int]:
    """Return the heading's source text and the index just past its close.

    The text is the inline source between the markers — no rendering, no
    stripping of emphasis. ``## API **v2**`` yields ``API **v2**``.
    """
    text = ""
    cursor = index + 1
    while cursor < len(tokens) and tokens[cursor].type != "heading_close":
        if tokens[cursor].type == "inline":
            text = tokens[cursor].content.strip()
        cursor += 1
    return text, cursor + 1


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
    """Apply the heading-stack rule.

    Everything at the same level or deeper is popped, then this heading is
    pushed. A jump from ``##`` straight to ``####`` therefore nests without
    inventing the ``###`` that the author did not write — the path records what
    the document says, not what it "should" have said.

    A heading with no text (``##`` on its own) closes the levels below it but
    contributes no path entry, because an empty string is not a section name.
    """
    while stack and stack[-1][0] >= level:
        stack.pop()
    if text:
        stack.append((level, text))


def _read_container(tokens: list[Token], index: int, lines: list[str]) -> tuple[Block | None, int]:
    """Consume a container block (paragraph, list, blockquote, table, …)."""
    opening = tokens[index]
    kind = _kind_of(opening.type)
    split_units: list[str] = []

    cursor = index + 1
    depth = 1
    while cursor < len(tokens) and depth > 0:
        inner = tokens[cursor]
        if inner.nesting == 1:
            depth += 1
            # Only direct children count: items of a *nested* list sit deeper
            # and must stay inside their parent item.
            if inner.type == "list_item_open" and inner.level == opening.level + 1:
                unit = _slice(lines, inner)
                if unit:
                    split_units.append(unit)
        elif inner.nesting == -1:
            depth -= 1
        cursor += 1

    text = _slice(lines, opening)
    if not text:
        return None, cursor
    return Block(kind=kind, text=text, split_units=tuple(split_units)), cursor


def _read_leaf(token: Token, lines: list[str]) -> Block | None:
    """Consume a self-contained block (fence, indented code, rule, HTML)."""
    text = _slice(lines, token)
    if not text:
        return None
    return Block(kind=_kind_of(token.type), text=text)


def _slice(lines: list[str], token: Token) -> str:
    """Cut a token's source out of the body by its line range."""
    if token.map is None:
        return ""
    start, end = token.map
    # Only the right edge is trimmed: leading spaces can start a code block or
    # mark a nested list item, so they are content.
    return "\n".join(lines[start:end]).rstrip()


def _kind_of(token_type: str) -> BlockKind:
    match token_type:
        case "paragraph_open":
            return BlockKind.PARAGRAPH
        case "bullet_list_open" | "ordered_list_open":
            return BlockKind.LIST
        case "blockquote_open":
            return BlockKind.QUOTE
        case "fence" | "code_block":
            return BlockKind.CODE
        case _:
            return BlockKind.OTHER
