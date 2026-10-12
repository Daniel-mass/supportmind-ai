#state.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ResponseComposerAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResponseAudience(str, Enum):
    CUSTOMER = "customer"
    SUPPORT_AGENT = "support_agent"
    INTERNAL = "internal"


class ResponseTone(str, Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CONCISE = "concise"
    TECHNICAL = "technical"


class ResponseClassificationSignal(BaseModel):
    """
    Classification information supplied by the existing
    Classification Agent.

    This agent does not recalculate classification.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    source: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ResponseEvidence(BaseModel):
    """
    Evidence supplied by Retrieval / Graph-RAG.
    """

    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    source_reference: str | None = None
    score: float | None = Field(
        default=None,
        ge=0.0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ResponseResolutionSignal(BaseModel):
    """
    Resolution recommendation supplied by the existing
    Resolution Agent.
    """

    model_config = ConfigDict(extra="forbid")

    resolution_type: str = Field(..., min_length=1)
    recommendation: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    requires_human_review: bool = False
    proposed_actions: list[str] = Field(
        default_factory=list
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ResponseEscalationSignal(BaseModel):
    """
    Escalation recommendation supplied by the existing
    Escalation Agent.
    """

    model_config = ConfigDict(extra="forbid")

    escalation_type: str = Field(..., min_length=1)
    should_escalate: bool
    reason: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    recommended_team: str | None = None
    recommended_path: str | None = None
    evidence_ids: list[str] = Field(
        default_factory=list
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    requires_human_review: bool = False
    proposed_actions: list[str] = Field(
        default_factory=list
    )


class ComposedResponse(BaseModel):
    """
    Final response produced by the Response Composer Agent.

    This is the actual user-facing artifact.
    """

    model_config = ConfigDict(extra="forbid")

    response: str = Field(..., min_length=1)

    audience: ResponseAudience = ResponseAudience.CUSTOMER

    tone: ResponseTone = ResponseTone.PROFESSIONAL

    evidence_ids: list[str] = Field(
        default_factory=list
    )

    escalation_required: bool = False

    human_review_required: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class ResponseComposerAgentError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1)
    error_type: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    recoverable: bool = False


class ResponseComposerAgentState(BaseModel):
    """
    Complete state consumed by the Response Composer Agent.

    All upstream signals must already exist before composition.
    """

    model_config = ConfigDict(extra="forbid")

    request: str = Field(..., min_length=1)

    ticket_id: str | None = None

    audience: ResponseAudience = ResponseAudience.CUSTOMER

    tone: ResponseTone = ResponseTone.PROFESSIONAL

    classification_signals: list[
        ResponseClassificationSignal
    ] = Field(default_factory=list)

    evidence: list[ResponseEvidence] = Field(
        default_factory=list
    )

    resolution: ResponseResolutionSignal | None = None

    escalation: ResponseEscalationSignal | None = None

    composed_response: ComposedResponse | None = None

    status: ResponseComposerAgentStatus = (
        ResponseComposerAgentStatus.PENDING
    )

    iteration_count: int = Field(
        default=0,
        ge=0,
    )

    uncertainty_detected: bool = False

    uncertainty_reasons: list[str] = Field(
        default_factory=list
    )

    errors: list[ResponseComposerAgentError] = Field(
        default_factory=list
    )

    execution_metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    @property
    def successful(self) -> bool:
        return (
            self.status
            is ResponseComposerAgentStatus.COMPLETED
        )

    @property
    def has_evidence(self) -> bool:
        return len(self.evidence) > 0

    @property
    def has_resolution(self) -> bool:
        return self.resolution is not None

    @property
    def has_escalation(self) -> bool:
        return self.escalation is not None

    @property
    def has_response(self) -> bool:
        return self.composed_response is not None