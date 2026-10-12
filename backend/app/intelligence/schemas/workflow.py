from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.intelligence.schemas.ticket import PriorityLevel


class WorkflowEvent(BaseModel):
    """
    Canonical SupportMind workflow event.

    Represents one point-in-time operational event associated
    with a support ticket.

    Derived concepts such as time-in-state, transitions,
    handoffs, workflow variants, and lifecycle duration are
    calculated later by the workflow intelligence layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    event_id: str = Field(..., min_length=1)
    ticket_id: str = Field(..., min_length=1)

    timestamp: datetime

    activity: str = Field(..., min_length=1)

    status: str | None = Field(default=None, min_length=1)
    assignee: str | None = Field(default=None, min_length=1)
    team: str | None = Field(default=None, min_length=1)

    priority: PriorityLevel = PriorityLevel.UNKNOWN
    category: str | None = Field(default=None, min_length=1)

    source: str = Field(default="helpdesk", min_length=1)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone(
        cls,
        value: datetime,
    ) -> datetime:
        """
        Workflow timestamps must be timezone-aware.

        The Help Desk data contains UTC timestamps, and the
        Intelligence Layer must preserve their temporal meaning.
        """
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                "timestamp must be timezone-aware"
            )

        return value
