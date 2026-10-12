"""
graph_rag_models.py

Generation-layer contracts for SupportMind AI Graph-RAG.

IMPORTANT
---------
This module does NOT implement retrieval.

Existing retrieval contracts remain responsible for:
    - RetrievalRequest
    - RetrievalResult
    - ContextItem
    - BuiltContext

This module defines only the contracts required after retrieval:
    retrieval evidence
    evidence fusion
    grounded generation
    final Graph-RAG response
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    field_validator,
)

from app.retrieval.models.retrieval_models import RetrievalRequest


# ============================================================================
# Graph-RAG Evidence Source
# ============================================================================


class GraphRAGEvidenceSource(str, Enum):
    """
    Identifies the subsystem that produced a Graph-RAG evidence item.
    """

    RETRIEVAL_DENSE = "retrieval_dense"
    RETRIEVAL_SPARSE = "retrieval_sparse"
    RETRIEVAL_HYBRID = "retrieval_hybrid"
    RETRIEVAL_GRAPH = "retrieval_graph"

    HELPDESK_GRAPH = "helpdesk_graph"

    WORKFLOW = "workflow"
    TICKET_INTELLIGENCE = "ticket_intelligence"
    ORGANIZATION_INTELLIGENCE = "organization_intelligence"


# ============================================================================
# Graph-RAG Evidence Type
# ============================================================================


class GraphRAGEvidenceType(str, Enum):
    """
    Semantic type of evidence supplied to the generation layer.
    """

    DOCUMENT = "document"
    GRAPH_FACT = "graph_fact"
    METRIC = "metric"
    WORKFLOW_EVENT = "workflow_event"
    HANDOFF = "handoff"
    BOTTLENECK = "bottleneck"
    TREND = "trend"
    OPERATIONAL_SIGNAL = "operational_signal"
    TICKET_CONTEXT = "ticket_context"


# ============================================================================
# Grounding Status
# ============================================================================


class GroundingStatus(str, Enum):
    """
    Final groundedness state of a generated response.
    """

    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID = "invalid"


# ============================================================================
# Graph-RAG Request
# ============================================================================


class GraphRAGRequest(BaseModel):
    """
    Request for the generation-stage Graph-RAG pipeline.

    The existing RetrievalRequest is deliberately reused rather than
    duplicating retrieval configuration here.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=1,
        description="Original user question.",
    )

    retrieval_request: RetrievalRequest = Field(
        ...,
        description=(
            "Validated request for the existing retrieval pipeline."
        ),
    )

    ticket_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional Help Desk / Intelligence ticket identifier "
            "used for operational evidence."
        ),
    )

    include_intelligence: bool = Field(
        default=True,
        description=(
            "Whether Intelligence Layer evidence should be added."
        ),
    )

    include_workflow: bool = Field(
        default=True,
        description=(
            "Whether workflow-specific evidence should be added."
        ),
    )

    max_evidence_items: int = Field(
        default=20,
        gt=0,
        description=(
            "Maximum number of unified evidence items supplied "
            "to the generation layer."
        ),
    )

    max_evidence_characters: int = Field(
        default=16000,
        gt=0,
        description=(
            "Maximum total evidence characters supplied "
            "to the generation layer."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional request metadata.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """
        Normalize and reject empty queries.
        """

        value = value.strip()

        if not value:
            raise ValueError("query cannot be empty")

        return value

    @field_validator("ticket_id")
    @classmethod
    def validate_ticket_id(
        cls,
        value: str | None,
    ) -> str | None:
        """
        Normalize the optional ticket identifier.
        """

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError("ticket_id cannot be empty")

        return value


# ============================================================================
# Graph-RAG Evidence
# ============================================================================


class GraphRAGEvidence(BaseModel):
    """
    One normalized evidence item entering Graph-RAG generation.

    Evidence is intentionally independent of the underlying source.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    evidence_id: str = Field(
        ...,
        min_length=2,
        description=(
            "Deterministic evidence identifier used for citations, "
            "for example E1 or E12."
        ),
    )

    source: GraphRAGEvidenceSource = Field(
        ...,
        description="Subsystem that produced the evidence.",
    )

    evidence_type: GraphRAGEvidenceType = Field(
        ...,
        description="Semantic type of the evidence.",
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Human-readable evidence content.",
    )

    source_reference: str = Field(
        ...,
        min_length=1,
        description=(
            "Reference identifying the originating record, node, "
            "ticket, document, or metric."
        ),
    )

    score: StrictFloat | None = Field(
        default=None,
        description=(
            "Optional normalized relevance/confidence score."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provenance metadata.",
    )

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(
        cls,
        value: str,
    ) -> str:
        """
        Evidence IDs must follow the generation-layer citation format.

        Examples:
            E1
            E2
            E12
        """

        value = value.strip()

        if not value.startswith("E"):
            raise ValueError(
                "evidence_id must start with 'E'"
            )

        suffix = value[1:]

        if not suffix.isdigit():
            raise ValueError(
                "evidence_id must use the format E<number>"
            )

        if int(suffix) <= 0:
            raise ValueError(
                "evidence_id number must be greater than zero"
            )

        return value

    @field_validator("text", "source_reference")
    @classmethod
    def validate_required_strings(
        cls,
        value: str,
    ) -> str:
        """
        Normalize required strings and reject whitespace-only values.
        """

        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    @field_validator("score")
    @classmethod
    def validate_score(
        cls,
        value: float | None,
    ) -> float | None:
        """
        Ensure optional scores are finite numeric values.
        """

        if value is None:
            return None

        if isinstance(value, bool):
            raise ValueError("score must be numeric")

        if not math.isfinite(value):
            raise ValueError(
                "score must be finite"
            )

        return float(value)


# ============================================================================
# Graph-RAG Context
# ============================================================================


class GraphRAGContext(BaseModel):
    """
    Unified evidence package supplied to the generation layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=1,
    )

    evidence: list[GraphRAGEvidence] = Field(
        default_factory=list,
    )

    selected_count: int = Field(
        ge=0,
    )

    character_count: int = Field(
        ge=0,
    )

    truncated: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """
        Normalize and validate the context query.
        """

        value = value.strip()

        if not value:
            raise ValueError("query cannot be empty")

        return value

    @field_validator("selected_count")
    @classmethod
    def validate_selected_count(
        cls,
        value: int,
    ) -> int:
        """
        Selected count must be non-negative.
        """

        if isinstance(value, bool):
            raise ValueError(
                "selected_count must be an integer"
            )

        return value

    @field_validator("character_count")
    @classmethod
    def validate_character_count(
        cls,
        value: int,
    ) -> int:
        """
        Character count must be non-negative.
        """

        if isinstance(value, bool):
            raise ValueError(
                "character_count must be an integer"
            )

        return value


# ============================================================================
# Grounding Result
# ============================================================================


class GroundingResult(BaseModel):
    """
    Result of validating a generated answer against supplied evidence.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    status: GroundingStatus

    valid: bool

    cited_evidence_ids: list[str] = Field(
        default_factory=list,
    )

    invalid_citation_ids: list[str] = Field(
        default_factory=list,
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
    )

    issues: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Graph-RAG Response
# ============================================================================


class GraphRAGResponse(BaseModel):
    """
    Final structured result returned by the Graph-RAG generation layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(
        ...,
        min_length=1,
    )

    answer: str = Field(
        ...,
        min_length=1,
    )

    evidence: list[GraphRAGEvidence] = Field(
        default_factory=list,
    )

    grounding: GroundingResult

    provider: str = Field(
        ...,
        min_length=1,
        description="LLM provider used for generation.",
    )

    model_name: str = Field(
        ...,
        min_length=1,
        description="LLM model identifier used for generation.",
    )

    latency_ms: StrictFloat | None = Field(
        default=None,
        description="End-to-end Graph-RAG generation latency.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("query", "answer", "provider", "model_name")
    @classmethod
    def validate_required_strings(
        cls,
        value: str,
    ) -> str:
        """
        Normalize required string fields.
        """

        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    @field_validator("latency_ms")
    @classmethod
    def validate_latency(
        cls,
        value: float | None,
    ) -> float | None:
        """
        Validate optional generation latency.
        """

        if value is None:
            return None

        if isinstance(value, bool):
            raise ValueError(
                "latency_ms must be numeric"
            )

        if not math.isfinite(value):
            raise ValueError(
                "latency_ms must be finite"
            )

        if value < 0:
            raise ValueError(
                "latency_ms cannot be negative"
            )

        return float(value)