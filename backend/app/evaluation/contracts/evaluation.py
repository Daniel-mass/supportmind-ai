from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    """Input data used to evaluate an AI/RAG response."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)

    retrieved_context: list[str] = Field(default_factory=list)

    generated_answer: str = Field(..., min_length=1)

    reference_answer: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetricResult(BaseModel):
    """Result of one evaluation metric."""

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


class RAGEvaluationResult(BaseModel):
    """Aggregated evaluation result for one RAG evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., min_length=1)

    metrics: list[EvaluationMetricResult] = Field(
        default_factory=list
    )

    overall_passed: bool | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def metric_count(self) -> int:
        return len(self.metrics)

    @property
    def evaluated(self) -> bool:
        return self.metric_count > 0