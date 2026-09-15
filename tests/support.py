"""Small helpers shared by the async-facing tests."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Drive one coroutine to completion, so async tests read synchronously.

    The ports are async because real adapters do I/O; the tests around them are
    not, and threading `async def` through every one of them would add noise
    without adding coverage.
    """
    return asyncio.run(coroutine)
