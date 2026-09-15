# ADR 0004 — Knowledge lives in versioned Markdown with YAML frontmatter

* **Status:** accepted
* **Date:** 2026-08-07
* **Phase:** 2

## Context

The corpus needs a home. The options were a database table, a headless CMS, or
files in this repository.

Whatever is chosen becomes the thing everything else is derived from, and it has
properties the rest of the system inherits. The corpus is small (dozens of
documents), authored by one person, changes rarely, and is reviewed rather than
submitted. Its accuracy matters more than its size: a wrong document produces a
confidently wrong, well-cited answer.

A second problem sits underneath the format choice. The frontmatter schema will
change — new fields, changed meanings, retired ones. Documents written today
will still be on disk then. Without a declared version, a reader has to guess
whether a missing field means "not set" or "written before that field existed",
and guessing wrong is silent.

## Decision

Knowledge documents are Markdown files with a YAML frontmatter block, stored in
`knowledge/` in this repository, and every document declares
`schema_version: 1`.

* **Format**: one topic per file, UTF-8, `kebab-case.md`. The frontmatter is
  validated against `DocumentMetadata`; unknown keys are rejected.
* **Versioning**: `portfolio_rag.ingestion.metadata` holds the set of versions
  this build can read. A document declaring anything else is refused with
  `UNSUPPORTED_SCHEMA_VERSION` — never read optimistically, because silently
  ignoring a field that version 2 gave a meaning to is precisely the failure
  versioning exists to prevent.
* **No migration engine.** One version exists. When a second one genuinely
  does, the set above is the single place that has to learn about it.
* **Derived, not authoritative**: the vector index is rebuildable from these
  files. The files are the record; the index is a cache.

## Consequences

**Good**

* Git is the audit trail: what the assistant knew in March, who changed it, and
  why, are already answered.
* A knowledge change is a pull request — reviewable by the same process as code.
* No database, no CMS, no service, no cost, and no credentials for local work.
* Ingestion is testable against a directory of fixtures, with no infrastructure.
* Portable: Markdown outlives any particular vector database.

**Bad / accepted**

* Editing requires a text editor and a commit — there is no web UI. Acceptable
  while the author is also the developer; a non-technical editor would need one.
* The corpus is public to anyone who can read the repository, so `visibility`
  controls what may be *answered with*, not what is secret. Genuinely
  confidential material does not belong in `knowledge/`.
* Scaling to thousands of documents would make full re-ingestion on every run
  wasteful. The document hash exists so that incremental re-indexing can be
  built when that day comes; it is not needed now.

**Revisit when** a non-technical author needs to edit the corpus, when
documents must be added without a deploy, or when the corpus grows past what a
directory listing can sensibly hold.
