"""HTTP transport layer: routing, request/response schemas, middleware.

This layer owns everything that is HTTP-specific — status codes, headers,
serialization, CORS — and nothing else. Business behaviour belongs in the
application and query layers, so route handlers stay thin.

``api`` may import from ``core``, ``domain`` and ``ports``. Nothing imports
``api`` except ``portfolio_rag.main``.
"""
