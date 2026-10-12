"""
orchestration_schema.py

Schemas used by the SupportMind AI end-to-end
ingestion orchestration layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StageMetric(BaseModel):
    """
    Metrics for one orchestration stage.
    """

    stage: str
    status: str
    duration_seconds: float = 0.0
    items_in: int = 0
    items_out: int = 0
    error: str | None = None


class OrchestrationResult(BaseModel):
    """
    Final result of an end-to-end ingestion run.
    """

    success: bool = False

    started_at: datetime = Field(
        default_factory=datetime.utcnow
    )

    completed_at: datetime | None = None

    source_directory: str

    documents_discovered: int = 0

    documents_processed: int = 0

    documents_failed: int = 0

    chunks_created: int = 0

    vectors_created: int = 0

    collection_name: str | None = None

    vector_provider: str | None = None

    embedding_provider: str | None = None

    embedding_dimension: int | None = None

    stages: list[StageMetric] = Field(
        default_factory=list
    )

    errors: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )