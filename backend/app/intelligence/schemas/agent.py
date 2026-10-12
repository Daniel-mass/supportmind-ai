#schemas/agents.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Agent(BaseModel):
    """
    Canonical SupportMind support-agent representation.

    This schema represents agent identity and operational
    grouping only.

    Performance metrics such as workload, processing time,
    handoff rate, and ticket volume are calculated by the
    analytics layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    agent_id: str = Field(..., min_length=1)

    name: str | None = Field(default=None, min_length=1)
    team: str | None = Field(default=None, min_length=1)

    active: bool = True

    source: str = Field(
        default="helpdesk",
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )