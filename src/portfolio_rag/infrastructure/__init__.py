"""Adapters: the only place that knows a vendor's name.

Everything here implements a port from :mod:`portfolio_rag.ports` and is
selected at the composition root. Nothing above this layer imports it, and
nothing in it imports the application layer — which is what makes a provider
swap a one-line change at the composition root rather than a refactor.

Two kinds of adapter live side by side on purpose:

* **Reference implementations** — ``DeterministicEmbeddingProvider`` and
  ``InMemoryVectorStore``. They make the whole indexing pipeline runnable and
  testable with no account, no key, no network and no cost. They are real
  implementations of the contract, not mocks.
* **Real providers** — the Mistral embedding adapter and the Cloudflare
  Vectorize store. Both need credentials, both are opt-in, and neither is
  required for development or CI.
"""
