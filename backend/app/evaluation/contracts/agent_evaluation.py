from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AgentEvaluationCase(BaseModel):
    """Input and expected behavior for one agent evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1)

    input: str = Field(..., min_length=1)

    actual_output: str = Field(..., min_length=1)

    expected_output: str | None = None

    tools_called: list[str] = Field(default_factory=list)

    expected_tools: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentMetricResult(BaseModel):
    """Result produced by one agent evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    metric_name: str = Field(..., min_length=1)

    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    passed: bool | None = None

    reason: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluationResult(BaseModel):
    """Combined result for one evaluated agent case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str

    metrics: list[AgentMetricResult] = Field(
        default_factory=list
    )

    overall_passed: bool | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def metric_count(self) -> int:
        """Return the number of evaluated metrics."""
        return len(self.metrics)

    @property
    def evaluated(self) -> bool:
        """Return whether at least one metric was evaluated."""
        return self.metric_count > 0