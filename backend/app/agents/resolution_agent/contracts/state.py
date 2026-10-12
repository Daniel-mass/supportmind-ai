from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# Resolution Agent Status
# ============================================================================


class ResolutionAgentStatus(str, Enum):
    """
    Lifecycle status of the Resolution Agent workflow.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Resolution Type
# ============================================================================


class ResolutionType(str, Enum):
    """
    Controlled vocabulary for the type of resolution recommended
    by the Resolution Agent.

    The Resolution Agent recommends an approach; it does not execute
    customer-impacting actions.

    IMPORTANT:
    These values form the stable internal application contract.
    LLM output must use these exact string values.
    """

    INFORMATIONAL = "informational"
    TROUBLESHOOTING = "troubleshooting"
    POLICY_BASED = "policy_based"
    ACCOUNT_ACTION = "account_action"
    BILLING = "billing"
    TECHNICAL = "technical"
    ESCALATION_RECOMMENDED = "escalation_recommended"
    UNKNOWN = "unknown"


# ============================================================================
# Classification Signal
# ============================================================================


class ResolutionClassificationSignal(BaseModel):
    """
    A classification prediction supplied to the Resolution Agent.

    ML predictions are advisory signals. They are not treated as absolute
    truth by the Resolution Agent.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(
        ...,
        min_length=1,
    )

    value: str = Field(
        ...,
        min_length=1,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    source: str = Field(
        ...,
        min_length=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Evidence
# ============================================================================


class ResolutionEvidence(BaseModel):
    """
    Evidence available to the Resolution Agent.

    Evidence may come from retrieval, Graph-RAG, ticket intelligence,
    workflow intelligence, organizational intelligence, or other approved
    SupportMind services.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    evidence_id: str = Field(
        ...,
        min_length=1,
    )

    source: str = Field(
        ...,
        min_length=1,
    )

    content: str = Field(
        ...,
        min_length=1,
    )

    source_reference: str | None = None

    score: float | None = Field(
        default=None,
        ge=0.0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Resolution Recommendation
# ============================================================================


class ResolutionRecommendation(BaseModel):
    """
    Structured resolution recommendation produced by the Resolution Agent.

    This is a recommendation, not an executed action.

    The resolution_type field is intentionally restricted to the
    ResolutionType enum so that the internal application never receives
    an uncontrolled resolution category.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    resolution_type: ResolutionType = (
        ResolutionType.UNKNOWN
    )

    recommendation: str = Field(
        ...,
        min_length=1,
    )

    rationale: str = Field(
        ...,
        min_length=1,
    )

    evidence_ids: list[str] = Field(
        default_factory=list,
    )

    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    requires_human_review: bool = False

    proposed_actions: list[str] = Field(
        default_factory=list,
    )


# ============================================================================
# Resolution Error
# ============================================================================


class ResolutionAgentError(BaseModel):
    """
    Structured error produced during Resolution Agent execution.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source: str = Field(
        ...,
        min_length=1,
    )

    error_type: str = Field(
        ...,
        min_length=1,
    )

    message: str = Field(
        ...,
        min_length=1,
    )

    recoverable: bool = False


# ============================================================================
# Resolution Agent State
# ============================================================================


class ResolutionAgentState(BaseModel):
    """
    Complete state passed through the Resolution Agent workflow.

    The state intentionally contains data contracts rather than service
    instances. Existing SupportMind services will be connected later through
    the Resolution Agent tools layer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # ------------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------------

    request: str = Field(
        ...,
        min_length=1,
    )

    ticket_id: str | None = None

    # ------------------------------------------------------------------------
    # Classification evidence
    # ------------------------------------------------------------------------

    classification_signals: list[
        ResolutionClassificationSignal
    ] = Field(
        default_factory=list,
    )

    # ------------------------------------------------------------------------
    # Retrieved / operational evidence
    # ------------------------------------------------------------------------

    evidence: list[
        ResolutionEvidence
    ] = Field(
        default_factory=list,
    )

    # ------------------------------------------------------------------------
    # Resolution output
    # ------------------------------------------------------------------------

    recommendation: ResolutionRecommendation | None = None

    # ------------------------------------------------------------------------
    # Workflow state
    # ------------------------------------------------------------------------

    status: ResolutionAgentStatus = (
        ResolutionAgentStatus.PENDING
    )

    iteration_count: int = Field(
        default=0,
        ge=0,
    )

    # ------------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------------

    uncertainty_detected: bool = False

    uncertainty_reasons: list[str] = Field(
        default_factory=list,
    )

    # ------------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------------

    errors: list[
        ResolutionAgentError
    ] = Field(
        default_factory=list,
    )

    # ------------------------------------------------------------------------
    # Execution metadata
    # ------------------------------------------------------------------------

    execution_metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def successful(self) -> bool:
        """
        Return whether the Resolution Agent completed successfully.
        """

        return (
            self.status
            == ResolutionAgentStatus.COMPLETED
        )

    @property
    def has_evidence(self) -> bool:
        """
        Return whether any evidence is available for resolution reasoning.
        """

        return len(self.evidence) > 0

    @property
    def has_recommendation(self) -> bool:
        """
        Return whether a resolution recommendation exists.
        """

        return self.recommendation is not None