"""
#classification/contracts/state.py
Classification Agent state and contracts.

This module contains data contracts only.

It does NOT:
- load ML models
- execute predictors
- call AutoGen
- call LLMs
- perform classification
- perform business logic

The contracts defined here are used to move classification data safely
between the Classification Agent, Coordinator, and downstream agents.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ClassificationStatus(str, Enum):
    """Lifecycle state of a classification workflow."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ClassificationType(str, Enum):
    """Supported classification signals."""

    CATEGORY = "category"
    PRIORITY = "priority"
    SENTIMENT = "sentiment"
    COMPLEXITY = "complexity"
    TOPIC = "topic"


class PriorityFeatures(BaseModel):
    """
    Structured features required by the trained priority model.

    These fields mirror the feature contract used during model training
    and notebook inference.
    """

    model_config = ConfigDict(extra="forbid")

    issue_proj: str
    issue_type: str
    created_year: int
    created_month: int
    created_dayofweek: int
    created_hour: int
    issue_contr_count: int


class ClassificationPrediction(BaseModel):
    """
    Prediction produced by one classification capability.

    The prediction value is intentionally represented as a string because
    the five existing predictors produce different label/value types.
    """

    model_config = ConfigDict(extra="forbid")

    classification_type: ClassificationType
    value: str
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """
    Combined output of the Classification Agent.

    Each field corresponds to one existing SupportMind classification
    capability.
    """

    model_config = ConfigDict(extra="forbid")

    category: ClassificationPrediction | None = None
    priority: ClassificationPrediction | None = None
    sentiment: ClassificationPrediction | None = None
    complexity: ClassificationPrediction | None = None
    topic: ClassificationPrediction | None = None

    predictions_completed: int = Field(
        default=0,
        ge=0,
    )

    predictions_requested: int = Field(
        default=0,
        ge=0,
    )

    uncertainty_detected: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationError(BaseModel):
    """Represents a classification workflow error."""

    model_config = ConfigDict(extra="forbid")

    source: str
    error_type: str
    message: str
    recoverable: bool = False


class ClassificationState(BaseModel):
    """
    Mutable state for one Classification Agent workflow.

    This state is passed through the classification workflow and eventually
    converted into the result consumed by the Coordinator.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None

    ticket_text: str = Field(
        min_length=1,
    )

    priority_features: PriorityFeatures | None = None

    status: ClassificationStatus = (
        ClassificationStatus.PENDING
    )

    requested_predictions: list[ClassificationType] = Field(
        default_factory=list,
    )

    completed_predictions: list[ClassificationType] = Field(
        default_factory=list,
    )

    predictions: ClassificationResult = Field(
        default_factory=ClassificationResult,
    )

    errors: list[ClassificationError] = Field(
        default_factory=list,
    )

    iteration_count: int = Field(
        default=0,
        ge=0,
    )