#schemas/ticket.py
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PriorityLevel(str, Enum):
    """Canonical SupportMind priority levels."""

    LOWEST = "Lowest"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    HIGHEST = "Highest"
    BLOCKER = "Blocker"
    UNKNOWN = "unknown"


class Ticket(BaseModel):
    """
    Canonical SupportMind ticket representation.

    This schema represents ticket-level business information.
    Workflow-specific state and event information belongs in
    WorkflowEvent rather than this model.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    ticket_id: str = Field(..., min_length=1)
    project: str = Field(..., min_length=1)
    issue_num: int = Field(..., ge=0)

    issue_type: str = Field(..., min_length=1)
    priority: PriorityLevel = PriorityLevel.UNKNOWN
    status: str = Field(..., min_length=1)

    created_at: datetime
    started_at: datetime | None = None
    ended_at: datetime | None = None
    resolution_date: datetime | None = None

    resolution: str | None = None

    reporter: str | None = None
    assignee: str | None = None

    contributor_count: int = Field(default=0, ge=0)
    comments_count: int = Field(default=0, ge=0)

    source: str = Field(default="helpdesk", min_length=1)

    metadata: dict[str, Any] = Field(default_factory=dict)