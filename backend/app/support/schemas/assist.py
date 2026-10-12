from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.agents.coordinator.state import (
    AgentResult,
    Capability,
    ClassificationSignals,
    EvidenceReference,
    WorkflowError,
    WorkflowStatus,
    WorkflowStep,
)


class SupportAssistRequest(BaseModel):
    """
    Public request contract for SupportMind.

    The API accepts only a natural-language support request.
    Workflow capabilities and model-specific features remain
    internal implementation details.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    request: str = Field(
        min_length=1,
        description="Natural-language customer or support request.",
        examples=[
            "My refund hasn't arrived and I've been waiting "
            "for two weeks. Can you check what's happening?"
        ],
    )


class SupportAssistResponseContent(BaseModel):
    """
    Customer-facing response content.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    text: str | None = None


class SupportAssistRetrievalSummary(BaseModel):
    """
    High-level retrieval / Graph-RAG information intended
    for frontend AI-analysis views.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    strategy: str | None = None
    evidence_count: int = Field(
        default=0,
        ge=0,
    )
    grounding_status: str | None = None
    grounding_valid: bool | None = None
    provider: str | None = None
    model_name: str | None = None
    latency_ms: float | None = None


class SupportAssistWorkflowSummary(BaseModel):
    """
    High-level workflow information for frontend visualization.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: WorkflowStatus
    current_step: WorkflowStep | None = None

    completed_capabilities: list[Capability] = Field(
        default_factory=list,
    )

    iteration_count: int = Field(
        ge=0,
    )

    errors: list[WorkflowError] = Field(
        default_factory=list,
    )


class SupportAssistIntelligence(BaseModel):
    """
    Internal AI intelligence exposed intentionally for the
    SupportMind portfolio frontend.

    This allows the frontend to progressively reveal ML,
    RAG, evidence, and agent information without exposing
    raw internal state as the primary customer response.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    predictions: ClassificationSignals

    retrieval: SupportAssistRetrievalSummary

    evidence: list[EvidenceReference] = Field(
        default_factory=list,
    )

    agents: dict[str, AgentResult] = Field(
        default_factory=dict,
    )

    workflow: SupportAssistWorkflowSummary


class SupportAssistResponse(BaseModel):
    """
    Application-level SupportMind response.

    The customer-facing response is separated from the
    AI intelligence/provenance information used by the
    SupportMind frontend.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    request: str

    response: SupportAssistResponseContent

    intelligence: SupportAssistIntelligence