---
schema_version: 1
id: quality
title: Quality Tooling
document_type: reference
language: en
topics:
  - quality
technologies:
  - pytest
  - mypy
  - Ruff
source: evaluation/corpus/quality.md
source_type: authored
version: 1
updated_at: 2026-08-11
visibility: public
trust_level: verified
---

# Quality Tooling

Fixture content describing an invented "Example Service".

## Test runner

Automated tests are executed with pytest.

## Type checking

Static types are checked with mypy in strict mode.

## Linting and formatting

Ruff handles both linting and formatting for the whole repository.
