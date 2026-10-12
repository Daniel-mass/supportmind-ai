#schemas/evidence.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceScope(str, Enum):
    """Scope at which Help Desk evidence is calculated."""

    TICKET = "ticket"
    AGENT = "agent"
    TEAM = "team"
    ORGANIZATION = "organization"


class EvidenceType(str, Enum):
    """Type of operational evidence produced by the Intelligence Layer."""

    TICKET_METRIC = "ticket_metric"
    WORKFLOW_METRIC = "workflow_metric"
    HANDOFF = "handoff"
    BOTTLENECK = "bottleneck"
    TREND = "trend"
    OPERATIONAL_SIGNAL = "operational_signal"


class IntelligenceEvidence(BaseModel):
    """
    Canonical Help Desk evidence produced by the Intelligence Layer.

    This model represents observable, traceable operational evidence
    used by dashboards and downstream consumers.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    evidence_id: str = Field(..., min_length=1)

    scope: EvidenceScope

    ticket_id: str | None = Field(default=None, min_length=1)
    agent_id: str | None = Field(default=None, min_length=1)
    team_id: str | None = Field(default=None, min_length=1)

    evidence_type: EvidenceType

    metric_name: str = Field(..., min_length=1)

    value: int | float | str | bool

    unit: str | None = Field(default=None, min_length=1)

    source_dataset: str = Field(..., min_length=1)

    source_fields: list[str] = Field(
        ...,
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )