#escaltion/state.py
from __future__ import annotations


from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field



class EscalationAgentStatus(str, Enum):
    """Lifecycle status of the Escalation Agent."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EscalationType(str, Enum):
    """Types of escalation recommendations."""

    NO_ESCALATION = "no_escalation"
    HUMAN_REVIEW = "human_review"
    TEAM_ESCALATION = "team_escalation"
    SPECIALIST_ESCALATION = "specialist_escalation"
    INCIDENT_ESCALATION = "incident_escalation"
    SLA_ESCALATION = "sla_escalation"
    MANAGER_ESCALATION = "manager_escalation"
    UNKNOWN = "unknown"


class EscalationSignal(BaseModel):
    """
    A normalized operational signal used during escalation reasoning.

    Signals may come from workflow intelligence, SLA analysis,
    ticket analytics, classification, or other deterministic services.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationClassificationSignal(BaseModel):
    """
    Classification signal supplied by the Classification Agent.

    This is intentionally separate from EscalationSignal so that
    classification predictions remain distinguishable from
    operational escalation signals.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationResolutionSignal(BaseModel):
    """
    Resolution information supplied by the Resolution Agent.

    The Escalation Agent uses this as an input signal rather than
    blindly accepting the resolution recommendation.
    """

    resolution_type: str = Field(..., min_length=1)
    recommendation: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    requires_human_review: bool = False
    proposed_actions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationEvidence(BaseModel):
    """
    Evidence supporting or contradicting an escalation recommendation.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_reference: str | None = None
    score: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EscalationRecommendation(BaseModel):
    """
    Final recommendation produced by the Escalation Agent.

    This is a recommendation only. It does not execute an escalation.
    """

    model_config = ConfigDict(extra="forbid")

    escalation_type: EscalationType = EscalationType.UNKNOWN

    should_escalate: bool = False

    reason: str = Field(..., min_length=1)

    rationale: str = Field(..., min_length=1)

    recommended_team: str | None = None

    recommended_path: str | None = None

    evidence_ids: list[str] = Field(default_factory=list)

    confidence: float | None = Field(default=None, ge=0, le=1)

    requires_human_review: bool = False

    proposed_actions: list[str] = Field(default_factory=list)


class EscalationAgentError(BaseModel):
    """Structured error generated during Escalation Agent execution."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1)

    error_type: str = Field(..., min_length=1)

    message: str = Field(..., min_length=1)

    recoverable: bool = False


class EscalationAgentState(BaseModel):
    """
    Complete state passed through the Escalation Agent.

    The state contains deterministic signals and evidence as inputs,
    followed by the agent's escalation recommendation.
    """

    model_config = ConfigDict(extra="forbid")

    request: str = Field(..., min_length=1)

    ticket_id: str | None = None

    classification_signals: list[EscalationClassificationSignal] = Field(
        default_factory=list
    )

    resolution: EscalationResolutionSignal | None = None

    operational_signals: list[EscalationSignal] = Field(
        default_factory=list
    )

    evidence: list[EscalationEvidence] = Field(
        default_factory=list
    )

    recommendation: EscalationRecommendation | None = None

    status: EscalationAgentStatus = EscalationAgentStatus.PENDING

    iteration_count: int = Field(default=0, ge=0)

    uncertainty_detected: bool = False

    uncertainty_reasons: list[str] = Field(
        default_factory=list
    )

    errors: list[EscalationAgentError] = Field(
        default_factory=list
    )

    execution_metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def successful(self) -> bool:
        """Return whether the agent completed successfully."""

        return self.status is EscalationAgentStatus.COMPLETED

    @property
    def has_evidence(self) -> bool:
        """Return whether supporting evidence is available."""

        return len(self.evidence) > 0

    @property
    def has_operational_signals(self) -> bool:
        """Return whether operational signals are available."""

        return len(self.operational_signals) > 0

    @property
    def has_recommendation(self) -> bool:
        """Return whether an escalation recommendation exists."""

        return self.recommendation is not None