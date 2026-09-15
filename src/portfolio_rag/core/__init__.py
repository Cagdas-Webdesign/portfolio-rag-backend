"""Cross-cutting concerns: configuration, error taxonomy, logging, request context.

``core`` may not import from ``api``, ``domain`` or ``ports``. It is the
innermost layer and therefore has to stay dependency-free towards the rest of
the application.
"""
