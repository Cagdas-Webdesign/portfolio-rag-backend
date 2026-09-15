# Query-side fixture corpus

Three neutral documents used by the Phase 5 tests. They describe nothing but
themselves: no real person, project, employer, date or claim appears here.

| Document | Visibility | Why it exists |
| --- | --- | --- |
| `http-stack.md` | public | one fact a question can be answered from |
| `storage-layer.md` | public | a second fact, in a second document, for the multi-source test |
| `internal-notes.md` | internal | must never be retrievable by a public request |

`INTERNAL-ONLY-SECRET-VALUE` in `internal-notes.md` is a marker string invented
for the leakage test — it is not a credential and grants nothing.

Nothing in this directory is part of the knowledge base the assistant answers
from; discovery excludes this README the same way it excludes any other.
