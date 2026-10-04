"""portfolio-rag-assistant — a provider-agnostic RAG backend platform.

The public surface of this package is the ASGI application exposed by
``portfolio_rag.main``. Everything else is internal and may change between
roadmap phases (see ``docs/ROADMAP.md``).
"""

__all__ = ["SERVICE_NAME", "__version__"]

#: The project's one version. The package build reads it from here
#: (``[tool.hatch.version]`` in ``pyproject.toml``), ``/health`` and OpenAPI
#: default to it, and every evaluation export records it as
#: ``project_version``. A release tag is ``v`` + this value, set by hand.
__version__ = "1.1.0"

SERVICE_NAME = "portfolio-rag-assistant"
