"""Reading the model's reply back into something the backend can check.

The other half of :mod:`portfolio_rag.rag.prompt`. The prompt asks for one JSON
object with an answer, a list of source labels and a support verdict; this
parses exactly that,
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
from enum import StrEnum
from typing import Any, Final

from portfolio_rag.rag.errors import (
    GenerationFailure,
    GenerationFailureCategory,
    GenerationUnavailableError,
)

#: A JSON object, possibly wrapped in a Markdown fence by a model that could
#: not resist. Tolerated on input because it is common, cheap to strip and
#: unambiguous; anything less structured than this is a failed generation.
_FENCED_JSON: Final = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)

#: Labels as the context mints them.
_LABEL: Final = re.compile(r"^S\d+$")


class SupportVerdict(StrEnum):
    """How the model says its answer relates to the passages it was given.

    A statement about *this answer* and *these passages* — not about the
    question, and not about the kind of document a passage came from. A
    project passage that says what the answer says is ``STATED``; the same
    passage stretched to a different subject, a wider scope or another time
    is ``INFERRED``.
    """

    STATED = "stated"
    """The passages directly carry what the answer says."""

    INFERRED = "inferred"
    """The answer needed a conclusion, a wider scope, a generalization or the
    correction of an assumption the passages do not confirm."""

    NONE = "none"
    """The passages do not hold enough for the answer."""


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

    support: SupportVerdict | None = None
    """The model's verdict on its own answer. ``None`` when the reply does not
    carry one — which is not a verdict in the answer's favour, and is never
    read as one.

    Like the labels, this is a *claim*, and it is believed in one direction
    only: anything short of ``STATED`` is taken at its word, ``STATED`` earns
    nothing and the answer still has to pass citation validation."""


def _unusable(rule: str) -> GenerationUnavailableError:
    """A reply that broke the contract, and the name of the rule it broke.

    The name is a fixed word chosen here, never a piece of the reply: what a
    model wrote is content, and which check rejected it is not.
    """
    return GenerationUnavailableError(
        failure=GenerationFailure(
            category=GenerationFailureCategory.UNPARSEABLE_OUTPUT, detail=rule
        )
    )


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
        raise _unusable("answer_not_text")

    return GroundedAnswerDraft(
        answer=(answer or "").strip(),
        source_labels=_parse_labels(payload.get("sources")),
        support=_parse_support(payload.get("support")),
    )


def _parse_support(raw: object) -> SupportVerdict | None:
    """Read the verdict, or report that there is none.

    Missing or null is *no verdict*: the reply is otherwise well formed, so it
    is not a broken generation, and the caller refuses to publish on it. A
    value that is present and is not one of the three words is a broken reply —
    guessing which way an unknown word was meant is how a refusal gets
    published. Case and surrounding blanks are forgiven, as they are for labels.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise _unusable("support_not_recognized")
    try:
        return SupportVerdict(raw.strip().lower())
    except ValueError as exc:
        raise _unusable("support_not_recognized") from exc


def _load_object(text: str) -> dict[str, Any]:
    fenced = _FENCED_JSON.match(text)
    candidate = fenced.group("body") if fenced else text.strip()
    if not candidate:
        raise _unusable("reply_empty")

    try:
        payload = json.loads(candidate)
    except ValueError as exc:
        raise _unusable("reply_not_json") from exc
    if not isinstance(payload, dict):
        raise _unusable("reply_not_an_object")
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
        raise _unusable("sources_not_a_list")

    labels: list[str] = []
    for entry in raw:
        if not isinstance(entry, str):
            raise _unusable("source_not_text")
        cleaned = entry.strip().strip("[]").strip()
        candidate = cleaned.upper()
        labels.append(candidate if _LABEL.match(candidate) else cleaned)
    return tuple(labels)
