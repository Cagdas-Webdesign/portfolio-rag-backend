---
schema_version: 1
id: storage
title: Storage
document_type: reference
language: en
topics:
  - storage
technologies:
  - Markdown
source: evaluation/corpus/storage.md
source_type: authored
version: 1
updated_at: 2026-08-11
visibility: public
trust_level: verified
---

# Storage

Fixture content describing an invented "Example Service".

## Corpus format

Knowledge documents are stored as Markdown files with YAML frontmatter.

## Vector index

Embeddings live in a vector index that is derived from the Markdown corpus and
can always be rebuilt from it.

## Backups

Because the index is derived data, losing it is an inconvenience rather than
data loss.
