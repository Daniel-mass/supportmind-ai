'''retrieval state'''
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.graph_rag.graph_rag_models import GraphRAGResponse
from app.retrieval.context.context_models import BuiltContext
from app.retrieval.models.retrieval_models import RetrievalRequest


# ============================================================================
# Retrieval Agent Status
# ============================================================================


class RetrievalAgentStatus(str, Enum):
    """
    Lifecycle state of a Retrieval Agent execution.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# Retrieval Strategy
# ============================================================================


class RetrievalStrategy(str, Enum):
    """
    Retrieval execution path selected by the Retrieval Agent.
    """

    STANDARD = "standard"
    GRAPH_RAG = "graph_rag"


# ============================================================================
# Retrieval Agent Error
# ============================================================================


class RetrievalAgentError(BaseModel):
    """
    Structured error produced during Retrieval Agent execution.
    """

    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        ...,
        min_length=1,
        description="Component that produced the error.",
    )

    error_type: str = Field(
        ...,
        min_length=1,
        description="Machine-readable error type.",
    )

    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable error message.",
    )

    recoverable: bool = Field(
        default=False,
        description="Whether retry/recovery may be possible.",
    )


# ============================================================================
# Retrieval Agent State
# ============================================================================


class RetrievalAgentState(BaseModel):
    """
    Runtime state for one Retrieval Agent execution.

    The Retrieval Agent orchestrates existing SupportMind retrieval
    and Graph-RAG services. It does not duplicate their internal
    retrieval contracts.

    Existing contracts are therefore stored directly:

        RetrievalRequest
        BuiltContext
        GraphRAGResponse
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # ------------------------------------------------------------------------
    # Request
    # ------------------------------------------------------------------------

    request: RetrievalRequest = Field(
        ...,
        description="Validated request passed to the retrieval layer.",
    )

    use_graph_rag: bool = Field(
        default=False,
        description=(
            "Whether the Retrieval Agent should execute the "
            "Graph-RAG generation path."
        ),
    )

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    status: RetrievalAgentStatus = Field(
        default=RetrievalAgentStatus.PENDING,
    )

    strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.STANDARD,
    )

    iteration_count: int = Field(
        default=0,
        ge=0,
        description="Number of Retrieval Agent execution iterations.",
    )

    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------

    context: BuiltContext | None = Field(
        default=None,
        description=(
            "Generation-ready retrieval context returned by "
            "RetrievalService."
        ),
    )

    graph_rag_response: GraphRAGResponse | None = Field(
        default=None,
        description=(
            "Optional Graph-RAG response returned by GraphRAGService."
        ),
    )

    evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence identifiers exposed to downstream agents."
        ),
    )

    execution_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Retrieval execution metadata such as latency, "
            "candidate counts, reranking information, and strategy."
        ),
    )

    # ------------------------------------------------------------------------
    # Failure / diagnostics
    # ------------------------------------------------------------------------

    errors: list[RetrievalAgentError] = Field(
        default_factory=list,
        description="Structured errors encountered during execution.",
    )

    # ------------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------------

    @property
    def successful(self) -> bool:
        """
        Return whether the Retrieval Agent completed successfully.
        """

        return self.status == RetrievalAgentStatus.COMPLETED

    @property
    def has_results(self) -> bool:
        """
        Return whether the state contains retrieval output.
        """

        return (
            self.context is not None
            or self.graph_rag_response is not None
        )