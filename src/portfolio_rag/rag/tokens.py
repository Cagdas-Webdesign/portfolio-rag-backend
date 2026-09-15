"""Estimating how much of a model's context a piece of text will occupy.

**This is an estimate and is named like one.** It is not a tokenizer, it does
not match any provider's count, and no code here pretends otherwise.

Why not an exact count. An exact number requires the tokenizer of the model
that will actually run — a real dependency (``mistral-common`` pulls in a
sizeable tree), pinned per model, wrong the moment the model changes, and
useless for the second provider this architecture exists to allow. The budget
it would feed is itself a conservative guess: the ceiling is set well below any
current context window on cost and latency grounds, not at a hard boundary.
Paying a dependency to compute a precise input to an approximate decision is
not a good trade.

Why this constant. Three characters per token is deliberately *below* what
real tokenizers achieve on the text this corpus contains (English and German
prose with Markdown around it, typically 3.5 to 4 characters per token). Being low
means the estimate over-counts, and over-counting spends context that was
available rather than overflowing a window that was not. The error is bounded
and always in the safe direction — which is the only property a budget needs.

If evaluation ever shows the slack is costing real answer quality, the honest
fix is to measure the ratio against recorded prompts and adjust this constant,
with the measurement written down.
"""

from __future__ import annotations

import math
from typing import Final

#: Characters per token. Deliberately pessimistic — see the module docstring.
CHARS_PER_TOKEN: Final = 3.0


def estimate_tokens(text: str) -> int:
    """Estimate the token count of *text*, rounding up.

    Empty text costs nothing; anything else costs at least one token.
    """
    if not text:
        return 0
    return math.ceil(len(text) / CHARS_PER_TOKEN)
