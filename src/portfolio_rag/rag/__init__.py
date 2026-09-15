"""RAG: the query side of the system.

Where :mod:`portfolio_rag.ingestion` turns source material into indexable units
once, this package turns one question into one grounded answer, every time it
is asked:

    UserQuery
      → query representation   deterministic, versioned, nothing invented
      → query embedding        the corpus's own EmbeddingProvider
      → public retrieval       visibility=public, by construction
      → RetrievedChunk[]       ranked, thresholded, resolved to real text
      → GroundedContext        bounded, labelled S1…Sn, deterministic
      → GenerationRequest      instructions and knowledge in separate roles
      → LLMProvider            through the port; no vendor is named here
      → citation validation    only what the backend can prove
      → GroundedAnswer

Everything in here works against ports and pure functions. No module in this
package imports FastAPI, an HTTP client or a provider SDK, and none of them
opens a file — which is why the whole pipeline is exercised offline, with no
credentials, in ``tests/integration/test_rag_pipeline.py``.

The corpus side does not import this package, and this package does not produce
chunks. Chunk boundaries are a property of the corpus, decided once at
ingestion time; re-deciding them per question would mean answering against text
that was never embedded.
"""

from portfolio_rag.rag.citations import CitationOutcome, resolve_citations
from portfolio_rag.rag.context import (
    CONTEXT_REPRESENTATION_VERSION,
    ContextSource,
    GroundedContext,
    build_context,
)
from portfolio_rag.rag.errors import (
    ContextBudgetError,
    EmbeddingSpaceMismatchError,
    GenerationUnavailableError,
    QueryEmbeddingError,
    QueryValidationError,
    RetrievalUnavailableError,
)
from portfolio_rag.rag.generation import GroundedAnswerDraft, parse_generation
from portfolio_rag.rag.language import (
    DEFAULT_ANSWER_LANGUAGE,
    INSUFFICIENT_KNOWLEDGE_ANSWERS,
    AnswerLanguage,
    detect_language,
    insufficient_knowledge_answer,
)
from portfolio_rag.rag.policy import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_RETRIEVAL_POLICY,
    ContextPolicy,
    RetrievalPolicy,
)
from portfolio_rag.rag.prompt import GROUNDED_PROMPT_VERSION, build_generation_request
from portfolio_rag.rag.query import (
    MAX_QUERY_LENGTH,
    QUERY_REPRESENTATION_VERSION,
    UserQuery,
    normalize_query,
)
from portfolio_rag.rag.retrieval import (
    PUBLIC_VISIBILITY_FILTER,
    PublicRetrievalService,
    RetrievalOutcome,
)
from portfolio_rag.rag.service import (
    INSUFFICIENT_KNOWLEDGE_ANSWER,
    AnswerOutcome,
    GroundedAnswer,
    GroundedAnswerService,
)

__all__ = [
    "CONTEXT_REPRESENTATION_VERSION",
    "DEFAULT_ANSWER_LANGUAGE",
    "DEFAULT_CONTEXT_POLICY",
    "DEFAULT_RETRIEVAL_POLICY",
    "GROUNDED_PROMPT_VERSION",
    "INSUFFICIENT_KNOWLEDGE_ANSWER",
    "INSUFFICIENT_KNOWLEDGE_ANSWERS",
    "MAX_QUERY_LENGTH",
    "PUBLIC_VISIBILITY_FILTER",
    "QUERY_REPRESENTATION_VERSION",
    "AnswerLanguage",
    "AnswerOutcome",
    "CitationOutcome",
    "ContextBudgetError",
    "ContextPolicy",
    "ContextSource",
    "EmbeddingSpaceMismatchError",
    "GenerationUnavailableError",
    "GroundedAnswer",
    "GroundedAnswerDraft",
    "GroundedAnswerService",
    "GroundedContext",
    "PublicRetrievalService",
    "QueryEmbeddingError",
    "QueryValidationError",
    "RetrievalOutcome",
    "RetrievalPolicy",
    "RetrievalUnavailableError",
    "UserQuery",
    "build_context",
    "build_generation_request",
    "detect_language",
    "insufficient_knowledge_answer",
    "normalize_query",
    "parse_generation",
    "resolve_citations",
]
