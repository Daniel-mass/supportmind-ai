"""
#agents/coordinator/state.py
Coordinator workflow state contracts.
#state.py
This module contains data contracts only.
It does not execute agents, call LLMs, or invoke ML models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.classification_agent.contracts.state import PriorityFeatures

class WorkflowStatus(str, Enum):
    """Lifecycle status of a Coordinator workflow."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_TOOL = "waiting_for_tool"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStep(str, Enum):
    """High-level execution step currently being performed."""

    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    RESOLUTION = "resolution"
    ESCALATION = "escalation"
    GENERATION = "generation"
    COMPLETED = "completed"


class Capability(str, Enum):
    """Capabilities that the Coordinator can request."""

    CLASSIFICATION = "classification"
    RETRIEVAL = "retrieval"
    RESOLUTION = "resolution"
    ESCALATION = "escalation"
    GENERATION = "generation"


class PredictionSignal(BaseModel):
    """
    ML prediction represented as an advisory signal.

    Predictions are not treated as verified facts.
    """

    model_config = ConfigDict(extra="forbid")

    value: str
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassificationSignals(BaseModel):
    """
    Predictions produced by the existing ML/NLP layer.

    These remain separate from verified evidence.
    """

    model_config = ConfigDict(extra="forbid")

    category: PredictionSignal | None = None
    priority: PredictionSignal | None = None
    sentiment: PredictionSignal | None = None
    complexity: PredictionSignal | None = None
    topic: PredictionSignal | None = None


class EvidenceReference(BaseModel):
    """
    Reference to evidence produced by existing SupportMind services.

    Raw evidence generation is handled outside the Coordinator.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source: str
    content: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    """Structured result returned by another SupportMind agent."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    success: bool
    result: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class WorkflowError(BaseModel):
    """Structured workflow error."""

    model_config = ConfigDict(extra="forbid")

    source: str
    error_type: str
    message: str
    recoverable: bool = False


class CoordinatorState(BaseModel):
    """
    Single source of truth for one Coordinator execution.

    The state intentionally separates:
    - ML prediction signals
    - agent results
    - retrieved/verified evidence

    This prevents ML predictions from automatically becoming facts.
    """

    model_config = ConfigDict(extra="forbid")

    request: str = Field(min_length=1)

    status: WorkflowStatus = WorkflowStatus.PENDING

    current_step: WorkflowStep | None = None

    requested_capabilities: list[Capability] = Field(
        default_factory=list
    )

    completed_capabilities: list[Capability] = Field(
        default_factory=list
    )

    predictions: ClassificationSignals = Field(
        default_factory=ClassificationSignals
    )

    agent_results: dict[str, AgentResult] = Field(
        default_factory=dict
    )

    evidence: list[EvidenceReference] = Field(
        default_factory=list
    )

    errors: list[WorkflowError] = Field(
        default_factory=list
    )

    iteration_count: int = Field(
        default=0,
        ge=0,
    )

    final_response: str | None = None
    priority_features: PriorityFeatures | None = None