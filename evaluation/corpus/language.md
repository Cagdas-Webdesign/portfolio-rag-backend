---
schema_version: 1
id: language
title: Language and Models
document_type: reference
language: en
topics:
  - backend
technologies:
  - Python
  - Pydantic
source: evaluation/corpus/language.md
source_type: authored
version: 1
updated_at: 2026-08-11
visibility: public
trust_level: verified
---

# Language and Models

Fixture content describing an invented "Example Service". This document exists
partly to create near-matches: it mentions Python and validation, which several
other fixture documents touch on too.

## Programming language

The Example Service is written in Python 3.13.

## Data models

Request, response and domain models are defined with Pydantic, which validates
every value at the boundary.
