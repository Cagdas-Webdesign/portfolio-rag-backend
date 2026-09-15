"""Use cases that coordinate several ports.

Indexing composes an embedding representation, an embedding provider, a vector
store and a change plan. That coordination belongs to none of those components
and stays outside both the CLI and concrete adapters.

``application`` may import from ``core``, ``domain``, ``ports`` and
``ingestion``. It must never import ``infrastructure``: it works against ports
and is handed concrete adapters by the composition root.
"""
