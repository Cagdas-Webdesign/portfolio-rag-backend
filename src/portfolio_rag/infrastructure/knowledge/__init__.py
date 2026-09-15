"""Adapters that make corpus text available at query time."""

from portfolio_rag.infrastructure.knowledge.memory import InMemoryChunkResolver

__all__ = ["InMemoryChunkResolver"]
