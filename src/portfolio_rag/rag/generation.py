"""Reading the model's reply back into something the backend can check.

The other half of :mod:`portfolio_rag.rag.prompt`. The prompt asks for one JSON
object with an answer and a list of source labels; this parses exactly that,
and lives beside the prompt because a contract with its two halves in different
layers is a contract that drifts.

**Structured output rather than regex archaeology.** Asking the provider for
JSON and parsing it turns "which sources did it claim?" into a field lookup.
Extracting citations from prose with a regular expression works until a model
writes ``(see S1 and S2)`` one day and ``[S1][S2]`` the next, and the failure
is silent: citations quietly stop being found and answers quietly stop being
grounded. When a provider cannot be asked for JSON, the honest fallback is to
fail the parse — not to guess.

**Nothing here is trusted.** A parsed label is a *claim*. Whether it refers to
a passage that actually existed is decided in :mod:`portfolio_rag.rag.citations`
against the context the backend built, never against what the model wrote.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final

from portfolio_rag.rag.errors import GenerationUnavailableError

#: A JSON object, possibly wrapped in a Markdown fence by a model that could
#: not resist. Tolerated on input because it is common, cheap to strip and
#: unambiguous; anything less structured than this is a failed generation.
_FENCED_JSON: Final = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)

#: Labels as the context mints them.
_LABEL: Final = re.compile(r"^S\d+$")


@dataclass(frozen=True, slots=True)
class GroundedAnswerDraft:
    """What the model claims: an answer, and the labels it says it used.

    A *draft* because nothing has been verified yet. It becomes an answer only
    after its labels have been checked against the real context.
    """

    answer: str
    """What the model wrote. May be empty: a provider that returned the
    requested object with no answer text has made a decision, not an error."""

    source_labels: tuple[str, ...]


def parse_generation(text: str) -> GroundedAnswerDraft:
    """Parse a provider's reply into a draft answer.

    Raises :class:`~portfolio_rag.rag.errors.GenerationUnavailableError` when
    the reply is not the object that was asked for.

    **An empty answer is not a failed generation.** A provider that returned
    the requested object and left the answer blank has made a decision — it
    had nothing to say about these passages — and that is a knowledge outcome,
    not an outage. The draft carries the empty text and
    :mod:`portfolio_rag.rag.service` sends it down the same refusal path as an
    answer nobody could verify. Escalating it instead would turn an honest
    "not covered" into a 503, which is both a worse answer and a misleading
    diagnosis: the provider was reachable and did reply.
    """
    payload = _load_object(text)

    answer = payload.get("answer")
    # Missing or null is an *empty* answer. A number or a list is not: that is
    # the wrong shape for the field, which is a broken reply rather than a
    # model declining to say anything.
    if answer is not None and not isinstance(answer, str):
        raise GenerationUnavailableError

    return GroundedAnswerDraft(
        answer=(answer or "").strip(),
        source_labels=_parse_labels(payload.get("sources")),
    )


def _load_object(text: str) -> dict[str, Any]:
    fenced = _FENCED_JSON.match(text)
    candidate = fenced.group("body") if fenced else text.strip()
    if not candidate:
        raise GenerationUnavailableError

    try:
        payload = json.loads(candidate)
    except ValueError as exc:
        raise GenerationUnavailableError from exc
    if not isinstance(payload, dict):
        raise GenerationUnavailableError
    return payload


def _parse_labels(raw: object) -> tuple[str, ...]:
    """Normalize claimed labels, keeping every claim the model actually made.

    Case and stray brackets are forgiven — ``"s1"`` and ``"[S1]"`` are plainly
    the same claim. Anything that is not label-shaped is kept *verbatim* rather
    than discarded, so that citation validation can report it as unknown. A
    parser that quietly drops bad labels makes a model look better behaved than
    it is.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise GenerationUnavailableError

    labels: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise GenerationUnavailableError
        cleaned = entry.strip().strip("[]").strip()
        candidate = cleaned.upper()
        labels.append(candidate if _LABEL.match(candidate) else cleaned)
    return tuple(labels)
