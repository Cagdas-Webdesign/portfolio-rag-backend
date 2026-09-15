---
schema_version: 1
id: http-api
title: HTTP API
document_type: reference
language: en
topics:
  - backend
technologies:
  - FastAPI
  - Uvicorn
source: evaluation/corpus/http-api.md
source_type: authored
version: 1
updated_at: 2026-08-11
visibility: public
trust_level: verified
---

# HTTP API

Fixture content describing an invented "Example Service". Nothing here refers to
a real person, project, employer or organisation.

## Web framework

The Example Service exposes its HTTP API with FastAPI.

## Application server

Uvicorn runs the ASGI application in production and in development.

## Versioning

Public routes live under the `/api/v1` prefix. A breaking change gets a new
prefix rather than a mutated one.
