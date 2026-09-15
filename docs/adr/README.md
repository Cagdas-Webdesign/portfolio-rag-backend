# Architecture Decision Records

One file per decision that would be expensive to reverse or surprising to a newcomer. Short and
concrete: context, decision, consequences, and what would make us revisit it.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-modular-monolith.md) | Modular monolith instead of microservices | accepted |
| [0002](0002-provider-agnostic-core.md) | External AI and storage systems are reached through ports | accepted |
| [0003](0003-fastapi-portable-runtime.md) | FastAPI/ASGI is the portable core; Cloudflare is a deployment target | accepted |
| [0004](0004-versioned-markdown-knowledge-format.md) | Knowledge lives in versioned Markdown with YAML frontmatter | accepted |
| [0005](0005-deterministic-structure-aware-chunking.md) | Deterministic, structure-aware chunking | accepted |
| [0006](0006-versioned-embeddings-and-incremental-indexing.md) | Versioned embedding representation and incremental vector indexing | accepted |
| [0007](0007-grounded-retrieval-and-backend-owned-citations.md) | Grounded retrieval, public-by-construction, backend-owned citations | accepted |
| [0008](0008-abuse-boundary-at-the-edge.md) | The public abuse boundary lives at the deployment edge | accepted |

## Writing a new one

Number sequentially, name the file `NNNN-short-slug.md`, and keep the structure above. Record the
decision when it is made — an ADR written afterwards documents a rationalization, not a decision.

Superseding beats editing: leave the old ADR in place, set its status to `superseded by NNNN`, and
explain in the new one what changed. The value of this directory is the trail, not the snapshot.
