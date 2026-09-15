"""Turning bytes on disk into one canonical text form.

The same document must produce the same result regardless of which editor,
operating system or keyboard wrote it. Three things get normalized:

* **Encoding.** UTF-8 only, with an optional byte-order mark removed. Anything
  else is rejected rather than guessed at — a mis-decoded document would ingest
  successfully and be subtly wrong, which is worse than failing.
* **Line endings.** CRLF and lone CR become LF.
* **Unicode composition.** NFC, so that ``ü`` written as one code point and
  ``ü`` written as ``u`` plus a combining diaeresis compare equal, hash equal
  and embed equal.

What is deliberately *not* done: nothing rewrites, reflows, re-cases or
"improves" the text. Markdown that survives this module means exactly what its
author wrote.
"""

from __future__ import annotations

import unicodedata
from typing import Final

from portfolio_rag.ingestion.errors import DocumentIngestionError, IngestionErrorCode

BYTE_ORDER_MARK: Final = "﻿"


def decode_source(raw: bytes) -> str:
    """Decode file bytes into canonical text, or fail loudly."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentIngestionError(
            IngestionErrorCode.INVALID_ENCODING,
            "File is not valid UTF-8.",
            reason=f"invalid byte at position {exc.start}",
        ) from exc
    return normalize_text(text)


def normalize_text(text: str) -> str:
    """Strip a leading BOM, unify line endings, normalize to Unicode NFC."""
    text = text.removeprefix(BYTE_ORDER_MARK)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text)


def normalize_body(body: str) -> str:
    """Remove file artifacts around the Markdown body, and nothing else.

    Only the edges are touched: blank lines left over from the frontmatter
    delimiter, and trailing whitespace at the end of the file. Line interiors
    stay untouched — trailing spaces mid-document are a Markdown hard line
    break, and leading spaces on the first line can start a code block.
    """
    return body.lstrip("\n").rstrip()
