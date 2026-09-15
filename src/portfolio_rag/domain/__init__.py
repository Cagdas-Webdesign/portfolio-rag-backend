"""Domain model: the vocabulary the whole system agrees on.

These types are pure data with no I/O, no framework and no provider knowledge.
Both the ingestion and retrieval sides are expressed in terms of them, which
keeps the ports in ``portfolio_rag.ports`` provider-agnostic.

``domain`` may import from ``core`` only.
"""
