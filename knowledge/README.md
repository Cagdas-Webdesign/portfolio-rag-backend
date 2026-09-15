# Knowledge base

Source documents the assistant will be grounded in. Markdown with YAML frontmatter, one topic per
file, reviewed like code.

Everything here is **read by `portfolio_rag.ingestion`** — the format below is enforced, not
advisory. Check your documents at any time:

```bash
uv run portfolio-rag knowledge validate
```

**This directory holds the 12 authorized documents of the portfolio corpus**, alongside the format
and a blank template. They are the first real knowledge base of this system and state facts about
Çağdaş Uçar and his work. Knowledge documents state facts about a real person and their work; they
are written by that person, never generated.

---

## Anatomy of a document

```markdown
---
schema_version: 1
id: api-integrations
title: API & Systemintegration
document_type: skill
language: de
topics:
  - backend
  - integration
technologies:
  - REST
  - JSON
source: portfolio
source_type: authored
version: 1
updated_at: 2026-08-07
visibility: public
trust_level: verified
---

# API & Systemintegration

The Markdown body starts here.
```

The frontmatter block must be the **first thing in the file**, opened and closed by a line
containing exactly `---`. Everything after the closing delimiter is the body. Later `---` lines are
ordinary horizontal rules and are left alone.

## Schema version

Every document declares `schema_version`. **The only version this build reads is `1`.**

A document declaring anything else is rejected with `UNSUPPORTED_SCHEMA_VERSION` rather than read
optimistically. That is the whole point: when version 2 exists and gives a field a new meaning, a
version-1 reader must refuse the document instead of quietly misinterpreting it. Which versions are
supported is decided in `portfolio_rag.ingestion.metadata` — one set, one place.

There is no migration engine and no version 2. There will be one when a second version genuinely
exists.

## Fields

Required unless marked optional. Unknown keys are **rejected** — a typo must fail loudly instead of
disappearing and leaving you believing a document is tagged when it is not.

| Field | Type | Notes |
| --- | --- | --- |
| `schema_version` | integer | Currently always `1`. |
| `id` | slug | Stable technical identity — see the rules below. |
| `title` | string, 1–200 chars | Human-readable heading. Shown in citations. |
| `document_type` | enum | `profile` · `skill` · `project` · `experience` · `education` · `article` · `faq` · `reference` |
| `language` | ISO 639-1 | `en`, `de`, … Lower case, two letters. One language per file; translations are separate documents. |
| `topics` | list of strings | Subject tags. Optional; defaults to empty. |
| `technologies` | list of strings | Named tools, languages, frameworks. Optional; defaults to empty. |
| `source` | string | Where the content came from: a URL, a repository path, or a name. |
| `source_type` | enum | `authored` · `repository` · `website` · `imported_file` |
| `version` | integer ≥ 1 | Bump on every material content change. Must be a real number, not `"1"`. |
| `updated_at` | `YYYY-MM-DD` | Date of that change. |
| `visibility` | enum | `public` or `internal`. Optional; defaults to `internal`. |
| `trust_level` | enum | `authoritative` · `verified` · `unverified`. Optional; defaults to `unverified`. |
| `license` | string | Optional. Required for anything not authored for this repository. |

The body must not be empty. Frontmatter without content is rejected with `EMPTY_DOCUMENT`.

### Document id rules

The id is a **stable technical identity**, not a title. Chunk ids and citations will derive from it.

* lower case ASCII letters and digits, hyphens between words
* no spaces, no slashes, no dots, no path segments, no underscores
* 2–100 characters, never starting or ending with a hyphen
* unique across the entire knowledge base — a collision is a hard error
* conventionally matches the filename, though nesting means that is not enforced

| Valid | Invalid | Why |
| --- | --- | --- |
| `profile` | `API Integrations` | spaces and capitals |
| `api-integrations` | `../profile` | path traversal |
| `project-ai-assistant` | `skills/backend` | slashes |
| | `my_profile` | underscore |
| | `trailing-` | trailing hyphen |

**An id is forever.** Renaming one orphans every citation that points at it. Deprecate a document
instead of renaming it, and never reuse an id for different content.

### Provenance is the point

`source`, `source_type`, `version`, `updated_at` and `trust_level` exist so that every future answer
can be traced back to a document, a revision of it, and a judgement about how much it should be
trusted:

* **`visibility` is a safety boundary, not a hint.** Retrieval for public clients will filter on it.
  A document that must not appear in a public answer is `internal`, whatever its content. Note that
  `internal` is not *secret* — this directory is as readable as the repository is.
* **`trust_level` ranks conflicts.** When two documents disagree, the more trusted one wins.
  `unverified` is the default so that trust is granted deliberately, not by omission.

Ingestion adds two derived fields you never write yourself: the document's path relative to this
directory, and a SHA-256 hash over its normalized metadata and body. The hash changes whenever
anything retrieval-relevant changes — including metadata, because changing `visibility` changes how
the document may be used.

## Which files are ingested

Everything under `knowledge/` matching `**/*.md`, recursively — **except**:

| Skipped | Rule |
| --- | --- |
| `README.md` | files named `readme`, any case: documentation *about* the base |
| `_template.md`, `_drafts/` | anything whose name starts with `_`: format artifacts |
| `.hidden.md`, `.obsidian/` | anything whose name starts with `.` |
| `notes.md~` | editor backups |
| `notes.txt`, `paper.pdf` | only `.md` is supported today |
| symlinks pointing outside `knowledge/` | the knowledge base cannot reach out of its own directory |

Results are sorted by relative path, so ingestion order is identical on every machine.

Subdirectories are free — group by topic if it helps:

```
knowledge/
├── profile.md
├── skills/
│   ├── backend.md
│   └── web.md
└── projects/
    └── assistant.md
```

## What ingestion changes, and what it never touches

Normalized, so that the same document written on any machine is the same document:

* the file is decoded as UTF-8 (a leading byte-order mark is dropped); anything else is rejected
  rather than guessed at
* CRLF and lone CR line endings become LF
* text is normalized to Unicode NFC, so `ü` typed as one character and `ü` typed as two compare,
  hash and embed identically
* blank lines left over around the frontmatter delimiter, and trailing whitespace at the end of the
  file, are trimmed

**Nothing else is touched.** Your prose is not reflowed, re-cased, re-punctuated or "improved".
Leading indentation on the first line is preserved (it may start a code block), and trailing spaces
inside the document are preserved (they are Markdown hard line breaks).

## Writing guidance

* One coherent topic per file — a document that answers three unrelated questions chunks badly.
* Use headings, and nest them sensibly. They become the heading path of every chunk beneath them,
  which is what makes a citation specific — and a heading is where the chunker is allowed to cut.
* Prefer specific, checkable statements over adjectives. Retrieval quality comes from content that
  actually answers a question.
* Write self-contained paragraphs. A chunk is retrieved without its neighbours, so a paragraph that
  only makes sense after the previous one will be retrieved without the context it needs.
* Keep code examples reasonably sized. A fenced block is never split — cutting one would produce an
  unterminated fence — so a single enormous example is rejected outright rather than chunked. Split
  it into the parts you would explain separately anyway.
* Keep it accurate and current. A wrong document produces a confidently wrong, well-cited answer —
  the worst failure mode this system has.

None of this is the algorithm leaking into your writing: it is the same advice that makes a document
readable. A document that a person can skim by its headings is a document that chunks well.

To see exactly where your document will be cut:

```bash
uv run portfolio-rag knowledge chunks <your-document-id> --show-content
```

## Template

Copy [`_template.md`](_template.md) and fill it in. It starts with `_`, so it is never ingested.

## When validation fails

`portfolio-rag knowledge validate` reports every problem it can find in one run, and exits non-zero
if any document is unusable:

| Code | Meaning |
| --- | --- |
| `INVALID_KNOWLEDGE_ROOT` | the directory does not exist or is not a directory |
| `UNREADABLE_SOURCE` | the file could not be read |
| `INVALID_ENCODING` | not valid UTF-8 |
| `MISSING_FRONTMATTER` | the file does not start with a `---` block |
| `INVALID_FRONTMATTER` | the block is unterminated, empty, not YAML, or not a mapping |
| `UNSUPPORTED_SCHEMA_VERSION` | written against a schema version this build does not read |
| `INVALID_METADATA` | a field is missing, has the wrong type, or is not a known field |
| `EMPTY_DOCUMENT` | valid frontmatter, no body |
| `DUPLICATE_DOCUMENT_ID` | two documents claim the same `id` |

A field written twice (`id:` on two lines) is an `INVALID_FRONTMATTER` error. YAML itself would
silently keep the last value; for authored metadata that is data loss, not a convenience.

Chunking reports one further problem, from `knowledge chunks`:

| Code | Meaning |
| --- | --- |
| `UNSPLITTABLE_BLOCK` | a code fence is larger than the chunk size limit and cannot be divided |

Inspect what ingestion made of a single document:

```bash
uv run portfolio-rag knowledge inspect api-integrations
```
