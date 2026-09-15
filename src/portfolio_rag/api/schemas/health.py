"""Schema for the operational health endpoint."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness answer.

    Intentionally free of internal detail: it reports that the process is up
    and which build is serving, nothing about dependencies or configuration.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(description="Always `ok` — a failing process cannot answer.")
    service: str = Field(description="Service identifier.")
    version: str = Field(description="Version of the running build.")
