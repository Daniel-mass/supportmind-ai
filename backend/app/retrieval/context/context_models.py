#context_models.py
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    field_validator,
)


# ============================================================================
# Context Source
# ============================================================================

class ContextSource(str, Enum):
    """
    Identifies where a retrieval/context item originated.

    These values are intentionally explicit because provenance is important
    for SupportMind's downstream reasoning and observability.
    """

    DENSE = "dense"

    # BM25 is the actual sparse retrieval mechanism used by the pipeline.
    BM25 = "bm25"

    # Keep SPARSE as a compatibility/general sparse source.
    SPARSE = "sparse"

    HYBRID = "hybrid"

    GRAPH = "graph"

    RERANKED = "reranked"


# ============================================================================
# Context Item
# ============================================================================

class ContextItem(BaseModel):
    """
    A single retrieval/evidence item.

    This is the common contract shared by:

        Dense Retrieval
        BM25 Retrieval
        Graph Retrieval
        Fusion
        Reranking
        Context Builder
        Generation Layer
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    id: str = Field(min_length=1)

    text: str = Field(min_length=1)

    source: ContextSource

    rank: int = Field(gt=0)

    score: StrictFloat | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("id", "text")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        """
        Normalize required string fields and reject whitespace-only values.
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
        Scores must be numeric and must never accept boolean values.

        bool is a subclass of int in Python, so it requires an explicit check.
        """

        if value is None:
            return None

        if isinstance(value, bool):
            raise ValueError("score must be numeric")

        return float(value)


# ============================================================================
# Context Budget
# ============================================================================

class ContextBudget(BaseModel):
    """
    Controls how much evidence can enter the final LLM context.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    max_items: int = Field(
        default=8,
        gt=0,
    )

    max_characters: int = Field(
        default=12000,
        gt=0,
    )


# ============================================================================
# Context Request
# ============================================================================

class ContextRequest(BaseModel):
    """
    Request passed into the context-building pipeline.

    RetrievalService populates `results` with fused/reranked candidates.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    query: str = Field(
        min_length=1,
    )

    results: list[ContextItem] = Field(
        default_factory=list,
    )

    budget: ContextBudget = Field(
        default_factory=ContextBudget,
    )

    include_metadata: bool = True

    include_provenance: bool = True

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """
        Normalize query and reject empty/whitespace-only queries.
        """

        value = value.strip()

        if not value:
            raise ValueError("query cannot be empty")

        return value


# ============================================================================
# Context Section
# ============================================================================

class ContextSection(BaseModel):
    """
    Structured section of the final generation context.

    Examples:

        Query
        Evidence
        Provenance
        Metadata
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    title: str = Field(
        min_length=1,
    )

    content: str = Field(
        min_length=1,
    )

    @field_validator("title", "content")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        """
        Normalize section strings.
        """

        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value


# ============================================================================
# Built Context
# ============================================================================

class BuiltContext(BaseModel):
    """
    Final generation-ready context produced by ContextBuilder.

    IMPORTANT:

    `selected_items` is the canonical internal field.

    `items` is supported as a compatibility alias because the retrieval
    service/test layer historically used `items`.

    Therefore both are valid:

        BuiltContext(
            selected_items=[...]
        )

    and:

        BuiltContext(
            items=[...]
        )

    Internally they represent the exact same selected evidence.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    query: str = Field(
        min_length=1,
    )

    sections: list[ContextSection] = Field(
        default_factory=list,
    )

    selected_items: list[ContextItem] = Field(
        default_factory=list,
        alias="items",
    )

    total_candidates: int = Field(
        ge=0,
    )

    selected_count: int = Field(
        ge=0,
    )

    character_count: int = Field(
        ge=0,
    )

    truncated: bool = False

    # ------------------------------------------------------------------
    # Compatibility property
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[ContextItem]:
        """
        Backwards-compatible access to selected evidence.

        This allows existing retrieval-service tests/components to use:

            context.items

        while the canonical field remains:

            context.selected_items
        """

        return self.selected_items

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        """
        Normalize and validate the final query.
        """

        value = value.strip()

        if not value:
            raise ValueError("query cannot be empty")

        return value

    @field_validator("selected_count")
    @classmethod
    def selected_count_is_consistent(
        cls,
        value: int,
        info,
    ) -> int:
        """
        Ensure selected_count exactly matches the selected evidence count.
        """

        selected_items = info.data.get("selected_items")

        if (
            selected_items is not None
            and value != len(selected_items)
        ):
            raise ValueError(
                "selected_count must match len(selected_items)"
            )

        return value