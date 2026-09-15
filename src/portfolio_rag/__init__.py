"""portfolio-rag-assistant — a provider-agnostic RAG backend platform.

The public surface of this package is the ASGI application exposed by
``portfolio_rag.main``. Everything else is internal and may change between
roadmap phases (see ``docs/ROADMAP.md``).
"""

__all__ = ["SERVICE_NAME", "__version__"]

__version__ = "0.1.0"

SERVICE_NAME = "portfolio-rag-assistant"
