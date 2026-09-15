---
schema_version: 1
id: deployment
title: Deployment
document_type: reference
language: en
topics:
  - operations
technologies:
  - Docker
source: evaluation/corpus/deployment.md
source_type: authored
version: 1
updated_at: 2026-08-11
visibility: public
trust_level: verified
---

# Deployment

Fixture content describing an invented "Example Service".

## Container image

The Example Service ships as a Docker image built from a pinned base image.

## Runtime user

The container process runs as a non-root user and writes nothing to disk.

## Health checks

A liveness probe is served at the unversioned `/health` route.
