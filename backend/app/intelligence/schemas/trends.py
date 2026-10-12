
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class TrendSeries(BaseModel):
    """One named series of monthly counts."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    label: str = Field(..., min_length=1)
    data: list[int] = Field(default_factory=list)


class MonthlyTrendPoint(BaseModel):
    """Counts for one calendar month."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    month: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}$",
    )
    count: int = Field(..., ge=0)


class TrendCategory(BaseModel):
    """One category's count for a given month."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    month: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}$",
    )
    category: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class OrganizationTrends(BaseModel):
    """Time-series analytics for the organization."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    months: list[str] = Field(default_factory=list)
    ticket_volume: list[MonthlyTrendPoint] = Field(
        default_factory=list
    )
    issue_type_trends: list[TrendCategory] = Field(
        default_factory=list
    )
    priority_trends: list[TrendCategory] = Field(
        default_factory=list
    )
    status_trends: list[TrendCategory] = Field(
        default_factory=list
    )
    total_tickets: int = Field(..., ge=0)
    total_workflow_events: int = Field(..., ge=0)