#schemas/overview.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DistributionItem(BaseModel):
    """
    One category/count pair used by dashboard summaries.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    label: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)


class MetricSummary(BaseModel):
    """
    One numeric operational metric for dashboard display.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    name: str = Field(..., min_length=1)
    value: int | float = Field(...)
    unit: str | None = Field(default=None, min_length=1)


class OverviewSummary(BaseModel):
    """
    Canonical SupportMind organization-level dashboard summary.

    This model represents aggregated Help Desk intelligence
    produced by the analytics layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    total_tickets: int = Field(..., ge=0)
    open_tickets: int = Field(..., ge=0)
    closed_tickets: int = Field(..., ge=0)

    priority_distribution: list[DistributionItem] = Field(
        default_factory=list
    )

    status_distribution: list[DistributionItem] = Field(
        default_factory=list
    )

    workflow_metrics: list[MetricSummary] = Field(
        default_factory=list
    )

    handoff_metrics: list[MetricSummary] = Field(
        default_factory=list
    )

    bottlenecks: list[dict[str, Any]] = Field(
        default_factory=list
    )

    emerging_issues: list[dict[str, Any]] = Field(
        default_factory=list
    )