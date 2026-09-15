"""Ports: the contracts the application core depends on.

Each port is a :class:`typing.Protocol` plus the small set of value objects it
exchanges. Adapters in ``portfolio_rag.infrastructure`` implement these
protocols against concrete providers — Mistral, Workers AI, Vectorize, … — and
are wired in at the application edge. No provider SDK is imported here, and
none is imported by any module that consumes these ports.

Structural typing means adapters do not subclass anything: a class simply has
to match the signature. Fakes in tests are therefore trivial to write.

``ports`` may import from ``core`` and ``domain`` only.
"""
